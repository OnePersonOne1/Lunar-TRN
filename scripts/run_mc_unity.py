"""Unity 실런 몬테카를로 CLI (P8): 프레임별 실측 τ(wallclock)로 n회 폐루프.

통계 MC(p7b_baseline)와 달리 τ·측정 오차에 모델 가정이 없다 — 렌더→탐지→연관→PnP를
실제로 돌리고 걸린 시간을 그 측정의 τ로 쓴다. Unity 렌더 서버(Play) 필요, 직렬 실행 전용.
시드별 체크포인트(jsonl)로 중단 후 재시작 시 완료 시드는 건너뛴다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mc import bootstrap_cep_ci, cep, error_ellipse_95, result_meta  # noqa: E402

MEAS_KEYS = ("t_c", "h_true_m", "n_det", "n_match", "n_inliers", "pnp_err_m",
             "valid", "tau_wallclock_s", "tau_det_s", "reproj_err_px")
RETRY_N = 3
RETRY_WAIT_S = 5.0


def load_done_seeds(ckpt: Path) -> set[int]:
    """체크포인트 jsonl에서 완료된 시드 집합 (파일 없으면 빈 집합)."""
    if not ckpt.exists():
        return set()
    done: set[int] = set()
    for line in ckpt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            done.add(int(json.loads(line)["seed"]))
    return done


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def append_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def dedupe_meas(rows: list[dict]) -> list[dict]:
    """(seed, t_c) 중복 시 마지막 항목 우선 (중단 후 재실행한 시드의 잔여 행 제거)."""
    by_key = {(r["seed"], r["t_c"]): r for r in rows}
    return list(by_key.values())


def aggregate(cfg: dict, runs: list[dict], meas_rows: list[dict], params: dict,
              failed_seeds: list[int], config_path: Path) -> dict:
    """완료 런 목록 → 결과 dict (CEP·CI·타원·R95·τ 분포·게이트 통계).

    발산 런(landing_xy 비유한 — 필터 발산으로 미착륙)은 CEP·타원에서 제외하고
    n_diverged/diverged_seeds로 따로 보고한다.
    """
    xy_all = np.asarray([[np.nan, np.nan] if r["landing_xy_m"] is None
                         else r["landing_xy_m"] for r in runs], dtype=float)
    finite = np.isfinite(xy_all).all(axis=1)
    diverged_seeds = [runs[i]["seed"] for i in np.flatnonzero(~finite)]
    landed = [r for r, f in zip(runs, finite) if f]
    xy = xy_all[finite]
    radii = np.linalg.norm(xy, axis=1)
    cep_m = cep(xy)
    rng = np.random.default_rng(0)
    ci = bootstrap_cep_ci(xy, int(cfg["mc"]["bootstrap_n"]), rng)
    r95 = float(np.percentile(radii, 95))
    done_seeds = {r["seed"] for r in runs}
    taus = np.asarray([m["tau_wallclock_s"] for m in dedupe_meas(meas_rows)
                       if m["seed"] in done_seeds], dtype=float)
    return {
        "meta": result_meta(config_path),
        "params": {**params, "n_runs": len(runs)},
        "n_landed": len(landed),
        "n_diverged": len(diverged_seeds),
        "diverged_seeds": diverged_seeds,
        "cep_m": cep_m,
        "cep_ci95_m": list(ci),
        "ellipse95": error_ellipse_95(xy),
        "r95_m": r95,
        "r95_over_cep": r95 / cep_m,  # 등방 가우시안이면 ≈ 2.08
        "landing_v_mean_mps": float(np.mean([r["landing_v_mps"] for r in landed])),
        "mean_n_meas": float(np.mean([r["n_meas"] for r in runs])),
        "mean_n_dropped": float(np.mean([r["n_dropped"] for r in runs])),
        "mean_gate_accept": float(np.nanmean(
            [np.nan if r["gate_accept"] is None else r["gate_accept"] for r in runs])),
        "mean_wall_s": float(np.mean([r["wall_s"] for r in runs])),
        "tau_wallclock_s": {
            "n_frames": int(taus.size),
            "median": float(np.median(taus)) if taus.size else None,
            "p95": float(np.percentile(taus, 95)) if taus.size else None,
            "max": float(taus.max()) if taus.size else None,
            "std": float(taus.std()) if taus.size else None,
        },
        "failed_seeds": failed_seeds,
        "landing_xy_m": xy.tolist(),
    }


def run_one(cfg: dict, seed: int, tau: float | str, detector: str) -> tuple[dict, list[dict]]:
    """폐루프 1회 → (체크포인트 레코드, meas jsonl 행들). 예외는 호출자에서 처리."""
    from sim.loop import run_closed_loop

    t0 = time.perf_counter()
    res = run_closed_loop(cfg, seed, tau=tau, measurement="unity",
                          detector_path=detector, frames_dir=None)
    wall = time.perf_counter() - t0
    taus = np.asarray([m["tau_wallclock_s"] for m in res["meas_log"]], dtype=float)
    gl = res["gate_log"]
    rec = {
        "seed": seed,
        "landing_xy_m": [float(v) for v in res["landing_xy"]],
        "landing_error_m": float(np.linalg.norm(res["landing_xy"])),
        "landing_v_mps": float(res["landing_v"]),
        "n_meas": int(res["n_meas"]),
        "n_dropped": int(res["n_dropped"]),
        "gate_accept": (sum(e["accepted"] for e in gl) / len(gl)) if gl else None,
        "tau_wallclock_median_s": float(np.median(taus)) if taus.size else None,
        "tau_wallclock_p95_s": float(np.percentile(taus, 95)) if taus.size else None,
        "tau_wallclock_max_s": float(taus.max()) if taus.size else None,
        "wall_s": wall,
    }
    meas_rows = [{"seed": seed, **{k: m.get(k) for k in MEAS_KEYS}} for m in res["meas_log"]]
    return rec, meas_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n-runs", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--detector", default="runs/export/crater_int8_ort.onnx")
    ap.add_argument("--tau", default="wallclock", help='"wallclock" 또는 고정값[s]')
    ap.add_argument("--ckpt", default="results/p8_unity_mc_runs.jsonl")
    ap.add_argument("--out", default="results/p8_unity_mc.json")
    ap.add_argument("--meas-log-out", default="results/p8_unity_mc_meas.jsonl")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    tau: float | str = args.tau if args.tau == "wallclock" else float(args.tau)
    ckpt, meas_out = Path(args.ckpt), Path(args.meas_log_out)

    done = load_done_seeds(ckpt)
    if done:
        print(f"체크포인트 재개: 완료 시드 {len(done)}개 건너뜀", flush=True)
    failed: list[int] = []
    seeds = [args.seed0 + i for i in range(args.n_runs)]
    for k, seed in enumerate(seeds, start=1):
        if seed in done:
            continue
        rec = None
        for attempt in range(1, RETRY_N + 1):
            try:
                rec, meas_rows = run_one(cfg, seed, tau, args.detector)
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # UnityRenderError·소켓 오류 등 — 시드 단위 재시도
                print(f"[{k}/{len(seeds)}] seed {seed} 시도 {attempt}/{RETRY_N} 실패: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                if attempt == RETRY_N:
                    traceback.print_exc()
                else:
                    time.sleep(RETRY_WAIT_S)
        if rec is None:
            failed.append(seed)
            continue
        append_jsonl(meas_out, meas_rows)  # 완료 표식(ckpt)은 마지막에 — 재개 안전
        append_jsonl(ckpt, [rec])
        tau_ms = (rec["tau_wallclock_median_s"] or 0.0) * 1e3
        print(f"[{k}/{len(seeds)}] seed {seed}  err {rec['landing_error_m']:7.1f} m  "
              f"τ_med {tau_ms:6.1f} ms  wall {rec['wall_s']:5.1f} s", flush=True)

    runs = [r for r in load_jsonl(ckpt) if args.seed0 <= r["seed"] < args.seed0 + args.n_runs]
    if not runs:
        raise SystemExit("완료된 런이 없다 — Unity 서버(Play) 상태를 확인하라")
    out = aggregate(cfg, runs, load_jsonl(meas_out),
                    params={"measurement": "unity", "tau": args.tau,
                            "detector": args.detector, "seed0": args.seed0,
                            "cpu_threads": int(cfg["bench"]["cpu_threads"])},
                    failed_seeds=failed, config_path=Path(args.config))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    t = out["tau_wallclock_s"]
    print(f"완료 {out['params']['n_runs']}런 (실패 {len(failed)}): "
          f"CEP {out['cep_m']:.1f} m  CI [{out['cep_ci95_m'][0]:.1f}, {out['cep_ci95_m'][1]:.1f}]  "
          f"τ_med {t['median'] * 1e3:.1f} ms → {args.out}", flush=True)


if __name__ == "__main__":
    main()
