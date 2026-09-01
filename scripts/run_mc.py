"""몬테카를로 배치 CLI: 단일 조건 n회 실행 → CEP·95% 타원·부트스트랩 CI json (+옵션 산포도)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mc import (  # noqa: E402
    bootstrap_cep_ci, cep, error_ellipse_95, extras_summary, result_meta, run_mc,
)
from sim.measurement import measurement_R  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/mc.json")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--tau-file", default=None,
                    help="results/tau_*.json 경로 — median_s를 τ로 사용 (--tau보다 우선)")
    ap.add_argument("--fp-rate", type=float, default=None)
    ap.add_argument("--fp-offset", type=float, default=None,
                    help="오검출 오프셋 [m] 강제 (기본: calibrated 파일 > config)")
    ap.add_argument("--delay-comp", choices=["on", "off"], default=None)
    ap.add_argument("--serial", choices=["on", "off"], default=None,
                    help="직렬 처리 모델(tau.serial) 강제 (기본 config)")
    ap.add_argument("--measurement-file", default=None,
                    help="calibrated 측정 모델 파일 강제 (기본 config measurement.file — s 모델 비교용)")
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--fig", default=None, help="착륙 산포도 png 경로 (옵션)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    n_runs = args.n_runs if args.n_runs is not None else int(cfg["mc"]["n_runs"])
    workers = args.workers if args.workers is not None else int(cfg["mc"]["workers"])
    delay_comp = None if args.delay_comp is None else args.delay_comp == "on"
    if args.serial is not None:
        cfg["tau"]["serial"] = args.serial == "on"
    if args.measurement_file is not None:
        cfg["measurement"]["file"] = args.measurement_file
    tau = args.tau
    if args.tau_file is not None:
        tau = float(json.loads(Path(args.tau_file).read_text(encoding="utf-8"))["median_s"])
    rng = np.random.default_rng(args.seed)

    xy, v, extras = run_mc(
        cfg, n_runs, workers, seed0=args.seed,
        tau=tau, fp_rate=args.fp_rate, fp_offset=args.fp_offset, delay_comp=delay_comp,
    )
    lo, hi = bootstrap_cep_ci(xy, int(cfg["mc"]["bootstrap_n"]), rng)
    out = {
        "meta": result_meta(args.config),
        "params": {
            "seed": args.seed, "tau": tau, "tau_file": args.tau_file,
            "fp_rate": args.fp_rate, "fp_offset": args.fp_offset,
            "delay_comp": delay_comp, "n_runs": n_runs,
            "serial": bool(cfg["tau"].get("serial", False)),
            "measurement_file": cfg["measurement"]["file"],
        },
        "assumed_measurement_stats": bool(measurement_R(cfg)[1]),
        "cep_m": cep(xy),
        "cep_ci95_m": [lo, hi],
        "ellipse95": error_ellipse_95(xy),
        "landing_v_mean_mps": float(v.mean()),
        **extras_summary(extras),
        "landing_xy_m": xy.tolist(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("cep_m", "cep_ci95_m", "ellipse95")}, indent=2))
    print(f"saved: {out_path}")

    if args.fig:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        ell = out["ellipse95"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.5)
        ax.add_patch(Ellipse(ell["center"], 2 * ell["semi_axes_m"][0], 2 * ell["semi_axes_m"][1],
                             angle=np.degrees(ell["angle_rad"]), fill=False, color="r",
                             label="95% ellipse"))
        ax.plot(0, 0, "r*", markersize=12)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        stats_tag = "assumed" if measurement_R(cfg)[1] else "calibrated"
        ax.set_title(f"Landing dispersion, n={n_runs} ({stats_tag} measurement stats)")
        fig.tight_layout()
        fig_path = Path(args.fig)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150)
        print(f"fig: {fig_path}")


if __name__ == "__main__":
    main()
