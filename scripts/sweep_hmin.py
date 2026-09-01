"""h_min 후보별 CEP 트레이드오프 CLI: 밴드별 실측 σ·이상치율로 MC → CEP vs h_min.

p5_sigma_vs_altitude.json의 프레임별 오차에서 각 후보 밴드 [h_min, h_max]의
0 기준 로버스트 σ(3σ 클리핑)·fp_rate를 산출해 그 값으로 MC(n_runs)를 돌린다.
→ results/p5_hmin_tradeoff.json + figs/p5_hmin_tradeoff.png.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mc import bootstrap_cep_ci, cep, result_meta, run_mc  # noqa: E402


def band_stats(frames: np.ndarray, h_min: float, h_max: float) -> dict | None:
    """밴드 내 프레임 오차의 로버스트 σ(0 기준)·fp_rate·표본 수."""
    m = (frames[:, 0] >= h_min) & (frames[:, 0] <= h_max)
    e = frames[m, 1:4]
    if len(e) < 20:
        return None
    sigma_mad = 1.4826 * np.median(np.abs(e), axis=0)
    inlier = np.all(np.abs(e) <= 3.0 * np.maximum(sigma_mad, 1e-9), axis=1)
    return {
        "sigma_xyz_m": np.sqrt(np.mean(e[inlier] ** 2, axis=0)).tolist(),
        "fp_rate": float((~inlier).mean()),
        "n_frames": int(len(e)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alt-json", default="results/p5_sigma_vs_altitude.json")
    ap.add_argument("--h-candidates", type=float, nargs="+",
                    default=[17000, 18000, 19000, 20000, 21000, 22000, 23000])
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--n-runs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scratch-dir", default=None,
                    help="후보별 측정모델 json 저장 위치 (기본: results/hmin_sweep)")
    ap.add_argument("--out", default="results/p5_hmin_tradeoff.json")
    ap.add_argument("--fig", default="figs/p5_hmin_tradeoff.png")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg0 = yaml.safe_load(fh)
    alt = json.loads(Path(args.alt_json).read_text(encoding="utf-8"))
    frames = np.asarray(alt["frames"], dtype=float)
    h_max = float(cfg0["trn_band"]["h_max_m"])
    scratch = Path(args.scratch_dir or "results/hmin_sweep")
    scratch.mkdir(parents=True, exist_ok=True)

    conds = []
    for h_min in args.h_candidates:
        st = band_stats(frames, h_min, h_max)
        if st is None:
            print(f"h_min {h_min:g}: 표본 부족 — 건너뜀")
            continue
        meas_file = scratch / f"measurement_h{int(h_min)}.json"
        meas_file.write_text(json.dumps({"sigma_xyz_m": st["sigma_xyz_m"]}), encoding="utf-8")
        cfg = copy.deepcopy(cfg0)
        cfg["trn_band"]["h_min_m"] = float(h_min)
        cfg["measurement"]["file"] = str(meas_file)
        for comp in (True, False):
            xy, _v, _ = run_mc(cfg, args.n_runs, args.workers, args.seed,
                            tau=args.tau, fp_rate=st["fp_rate"], delay_comp=comp)
            xy = np.asarray(xy)
            r50 = cep(xy)
            lo, hi = bootstrap_cep_ci(xy, n_boot=int(cfg["mc"]["bootstrap_n"]),
                                      rng=np.random.default_rng(args.seed))
            conds.append({
                "h_min_m": float(h_min), "delay_comp": comp,
                "sigma_xyz_m": [round(v, 1) for v in st["sigma_xyz_m"]],
                "fp_rate": round(st["fp_rate"], 4), "n_frames_calib": st["n_frames"],
                "cep_m": round(float(r50), 1), "cep_ci95_m": [round(float(lo), 1), round(float(hi), 1)],
            })
            print(f"h_min {h_min/1e3:g} km comp={comp}: CEP {r50:.1f} m "
                  f"(σ={np.round(st['sigma_xyz_m'],0)}, fp={st['fp_rate']:.3f})", flush=True)

    out = {"meta": result_meta(args.config), "tau_s": args.tau, "n_runs": args.n_runs,
           "h_max_m": h_max, "conditions": conds,
           "note": "밴드별 실측 σ·fp_rate를 주입한 통계 MC. iid 가정 — 실런의 시간상관 편향은 미포함."}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {args.out}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for comp, lab, mk in ((True, "delay compensation", "o"), (False, "no compensation", "s")):
        sel = [c for c in conds if c["delay_comp"] == comp]
        h = [c["h_min_m"] / 1e3 for c in sel]
        y = [c["cep_m"] for c in sel]
        yerr = np.array([[c["cep_m"] - c["cep_ci95_m"][0] for c in sel],
                         [c["cep_ci95_m"][1] - c["cep_m"] for c in sel]])
        ax.errorbar(h, y, yerr=yerr, marker=mk, capsize=3, label=lab)
    ax.set_xlabel("TRN band lower bound h_min [km]")
    ax.set_ylabel("landing CEP [m]")
    ax.set_yscale("log")
    ax.set_title(f"CEP vs h_min (band-specific measured sigma, tau={args.tau:g} s, n={args.n_runs})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=150)
    print(f"fig: {args.fig}")


if __name__ == "__main__":
    main()
