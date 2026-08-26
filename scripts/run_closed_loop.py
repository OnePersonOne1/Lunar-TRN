"""단일 폐루프 시뮬레이션 실행 CLI: 결과 json + (옵션) 궤적·추정오차 그림."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.loop import run_closed_loop  # noqa: E402
from sim.mc import result_meta  # noqa: E402


def _title(base: str, assumed: bool) -> str:
    return f"{base} (assumed measurement stats)" if assumed else base


def make_figs(res: dict, prefix: Path) -> list[str]:
    """figs/{prefix}_trajectory.png(3면도), {prefix}_est_error.png 생성."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    tr = res["traj_true"]
    est = res["traj_est"]
    assumed = res["meas_assumed"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    planes = [(0, 1, "East [m]", "North [m]"), (0, 2, "East [m]", "Up [m]"), (1, 2, "North [m]", "Up [m]")]
    for ax, (i, j, xl, yl) in zip(axes, planes):
        ax.plot(tr[:, i], tr[:, j], label="true")
        if est is not None:
            ax.plot(est[:, i], est[:, j], "--", label="estimate")
        ax.plot(0.0, 0.0, "r*", markersize=12, label="target")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle(_title("Closed-loop trajectory", assumed))
    fig.tight_layout()
    p = prefix.parent / f"{prefix.name}_trajectory.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    if est is not None:
        t = res["traj_t"]
        err = res["est_error"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for i, lab in enumerate(["E", "N", "U"]):
            axes[0].plot(t, err[:, i], label=f"pos {lab}")
            axes[1].plot(t, err[:, 3 + i], label=f"vel {lab}")
        axes[0].set_ylabel("position error [m]")
        axes[1].set_ylabel("velocity error [m/s]")
        axes[1].set_xlabel("t [s]")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
        fig.suptitle(_title("EKF estimation error", assumed))
        fig.tight_layout()
        p = prefix.parent / f"{prefix.name}_est_error.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(str(p))
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/closed_loop.json")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--fp-rate", type=float, default=None)
    ap.add_argument("--delay-comp", choices=["on", "off"], default=None)
    ap.add_argument("--measurement", choices=["stat", "truth", "unity"], default="stat")
    ap.add_argument("--figs-prefix", default=None, help="예: figs/p1 → figs/p1_trajectory.png 등")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    delay_comp = None if args.delay_comp is None else args.delay_comp == "on"

    res = run_closed_loop(
        cfg, args.seed, tau=args.tau, fp_rate=args.fp_rate,
        delay_comp=delay_comp, measurement=args.measurement,
    )

    n_meas = len(res["gate_log"])
    n_rej = sum(1 for e in res["gate_log"] if not e["accepted"])
    out = {
        "meta": result_meta(args.config),
        "params": {
            "seed": args.seed, "tau": args.tau, "fp_rate": args.fp_rate,
            "delay_comp": delay_comp, "measurement": args.measurement,
            "assumed_measurement_stats": res["meas_assumed"],
        },
        "landing_xy_m": res["landing_xy"].tolist(),
        "landing_error_m": float(np.linalg.norm(res["landing_xy"])),
        "landing_v_mps": res["landing_v"],
        "t_land_s": res["t_land"],
        "n_measurements": n_meas,
        "n_gate_rejected": n_rej,
        "final_est_error_m": (
            None if res["est_error"] is None
            else float(np.linalg.norm(res["est_error"][-1, :3]))
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.figs_prefix:
        prefix = Path(args.figs_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for p in make_figs(res, prefix):
            print(f"fig: {p}")


if __name__ == "__main__":
    main()
