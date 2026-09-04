"""P9 매트릭스 스윕 CLI: 탐지기(5) × 온보드 등급(3) × 지연보상(2) 통계 MC.

정밀 지도 가정(Tier 2: 계통 정합 오차 없음) 하에서, 탐지기 선택(mAP 차이의 실측 결과인
보정 σ·오검출률·τ)과 연산 등급(τ 스케일)이 착륙 CEP에 미치는 영향을 전 조합으로 정량화.
- mAP→σ는 수식 유도가 아니라 각 탐지기의 개루프 보정 실측(measurement_model*.json)을 쓴다.
- 등급 τ 스케일은 p7b_cpu_bench.json의 DMIPS 비율(차세대 HPSC ×1은 같은 자릿수 가정).
- τ = 상수(스케일된 중앙값, 계약 §2.4 스윕 규칙), 1 Hz 직렬, n=200/조건.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mc import bootstrap_cep_ci, cep, extras_summary, result_meta, run_mc  # noqa: E402

# 탐지기 사양: 측정 모델 파일(+by_precision 키), τ 출처, mAP 출처 (전부 results 실측)
DETECTORS = [
    {"key": "n_int8", "meas": "results/measurement_model.json", "prec": None,
     "tau_file": "results/tau_ort_cpu_int8.json",
     "map_file": "results/p5_det.json", "map_path": ("int8", "mAP50_95")},
    {"key": "n_fp32", "meas": "results/measurement_model.json", "prec": "fp32",
     "tau_file": "results/tau_ort_cpu_fp32.json",
     "map_file": "results/p5_det.json", "map_path": ("fp32", "mAP50_95")},
    {"key": "s_int8", "meas": "results/measurement_model_s.json", "prec": None,
     "tau_file": "results/s/tau_ort_cpu_int8.json",
     "map_file": "results/p5_det_s.json", "map_path": ("int8", "mAP50_95")},
    {"key": "s_fp32", "meas": "results/measurement_model_s.json", "prec": "fp32",
     "tau_file": "results/s/tau_ort_cpu_fp32.json",
     "map_file": "results/p5_det_s.json", "map_path": ("fp32", "mAP50_95")},
    {"key": "classic_pca", "meas": "results/measurement_model_classic.json", "prec": None,
     "tau_file": "results/p7c_det_compare.json",
     "tau_path": ("entries", "classic_pca_prior", "tau", "median_s"),
     "map_file": "results/p7c_det_compare.json",
     "map_path": ("entries", "classic_pca_prior", "mAP50_95")},
]


def _dig(d: dict, path: tuple[str, ...]):
    for k in path:
        d = d[k]
    return d


def derive_meas_file(src: Path, prec: str | None, out: Path) -> dict:
    """보정 모델(또는 by_precision 항목)에서 σ·오검출 통계를 뽑아 파생 파일로 저장."""
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    if prec is not None:
        d = d["by_precision"][prec]
    m = {"sigma_xyz_m": d["sigma_xyz_m"], "fp_rate_est": d["fp_rate_est"],
         "fp_offset_med_m": d["fp_offset_med_m"],
         "source": f"{src}" + (f"#by_precision.{prec}" if prec else "")}
    out.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m


def load_compute_classes(bench_path: Path) -> list[dict]:
    """온보드 등급 → τ 스케일 (p7b_cpu_bench.json DMIPS 비율)."""
    b = json.loads(Path(bench_path).read_text(encoding="utf-8"))["speedup_vs_reference"]
    gr740 = next(v for k, v in b.items() if "GR740" in k)["dmips_x"]
    hr5000 = next(v for k, v in b.items() if "HR5000" in k)["dmips_x"]
    return [
        {"key": "next_gen", "label": "차세대(HPSC급)", "tau_scale": 1.0,
         "note": "벤치 1스레드 ≈ HPSC 칩 목표(같은 자릿수 가정, onboard_cpu_emulation.md §1)"},
        {"key": "current_gen", "label": "현세대(GR740급)", "tau_scale": float(gr740),
         "note": "DMIPS 비율"},
        {"key": "legacy", "label": "구세대(HR5000/SLIM 세대급)", "tau_scale": float(hr5000),
         "note": "DMIPS 비율"},
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0, help="seed0 (시드 seed0..seed0+n-1)")
    ap.add_argument("--n-runs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bench", default="results/p7b_cpu_bench.json")
    ap.add_argument("--out", default="results/p9_matrix.json")
    ap.add_argument("--meas-dir", default="results", help="파생 측정 모델 파일 위치")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)
    classes = load_compute_classes(Path(args.bench))
    rng = np.random.default_rng(args.seed)

    dets = []
    for spec in DETECTORS:
        meas_out = Path(args.meas_dir) / f"p9_meas_{spec['key']}.json"
        m = derive_meas_file(Path(spec["meas"]), spec["prec"], meas_out)
        tau_d = json.loads(Path(spec["tau_file"]).read_text(encoding="utf-8"))
        tau_s = float(_dig(tau_d, spec["tau_path"]) if "tau_path" in spec
                      else tau_d["median_s"])
        map_d = json.loads(Path(spec["map_file"]).read_text(encoding="utf-8"))
        try:
            map5095 = float(_dig(map_d, spec["map_path"]))
        except (KeyError, TypeError):
            map5095 = None
        dets.append({**spec, "meas_file": str(meas_out), "sigma_xyz_m": m["sigma_xyz_m"],
                     "fp_rate": m["fp_rate_est"], "fp_offset_m": m["fp_offset_med_m"],
                     "tau_base_s": tau_s, "mAP50_95": map5095})
        print(f"{spec['key']}: σ={[round(v, 1) for v in m['sigma_xyz_m']]} "
              f"fp={m['fp_rate_est']:.3f} τ={tau_s * 1e3:.1f}ms mAP={map5095}")

    conditions = []
    n_total = len(dets) * len(classes) * 2
    k = 0
    for det in dets:
        for cls in classes:
            tau = det["tau_base_s"] * cls["tau_scale"]
            for comp in (True, False):
                k += 1
                t0 = time.perf_counter()
                cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # 깊은 복사
                cfg["measurement"]["mode"] = "calibrated"
                cfg["measurement"]["file"] = det["meas_file"]
                cfg["tau"]["serial"] = True
                xy, v, extras = run_mc(
                    cfg, args.n_runs, args.workers, seed0=args.seed,
                    tau=tau, fp_rate=det["fp_rate"], delay_comp=comp,
                    measurement="stat")
                ci = bootstrap_cep_ci(xy, int(cfg["mc"]["bootstrap_n"]), rng)
                cond = {
                    "detector": det["key"], "class": cls["key"], "delay_comp": comp,
                    "tau_s": tau, "cep_m": cep(xy), "cep_ci95_m": list(ci),
                    "landing_v_mean_mps": float(np.mean(v)),
                    **extras_summary(extras),
                }
                conditions.append(cond)
                print(f"[{k}/{n_total}] {det['key']:11s} {cls['key']:11s} "
                      f"comp={'on ' if comp else 'off'} τ={tau:7.2f}s "
                      f"CEP {cond['cep_m']:8.1f} m  meas {cond['mean_n_meas']:5.1f}  "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    out = {
        "meta": result_meta(args.config),
        "params": {"n_runs": args.n_runs, "seed0": args.seed, "serial": True,
                   "camera_rate_hz": float(base_cfg["camera"]["rate_hz"]),
                   "tau_mode": "constant(스케일된 중앙값)",
                   "assumption": "정밀 지도(계통 정합 오차 없음) — Tier 2 통계 MC",
                   "classes": classes,
                   "detectors": [{k2: d[k2] for k2 in
                                  ("key", "meas_file", "sigma_xyz_m", "fp_rate",
                                   "fp_offset_m", "tau_base_s", "mAP50_95")}
                                 for d in dets]},
        "conditions": conditions,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"done: {args.out} ({len(conditions)} conditions)")


if __name__ == "__main__":
    main()
