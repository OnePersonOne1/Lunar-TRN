"""τ 스윕 CLI: tau.sweep_s 각 값 × (보상/미보상) MC → CEP 곡선 json·그림."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mc import bootstrap_cep_ci, cep, result_meta, run_mc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/p1_tau_sweep.json")
    ap.add_argument("--fig", default="figs/p1_tau_sweep.png")
    ap.add_argument("--n-runs", type=int, default=None, help="τ·조건당 시행 수 (기본 mc.n_runs)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    n_runs = args.n_runs if args.n_runs is not None else int(cfg["mc"]["n_runs"])
    workers = args.workers if args.workers is not None else int(cfg["mc"]["workers"])
    taus = [float(t) for t in cfg["tau"]["sweep_s"]]
    n_boot = int(cfg["mc"]["bootstrap_n"])
    rng = np.random.default_rng(args.seed)

    curves: dict[str, dict] = {}
    assumed = True
    for comp in (True, False):
        key = "comp" if comp else "uncomp"
        ceps, ci_lo, ci_hi = [], [], []
        for tau in taus:
            xy, _ = run_mc(cfg, n_runs, workers, seed0=args.seed, tau=tau, delay_comp=comp)
            c = cep(xy)
            lo, hi = bootstrap_cep_ci(xy, n_boot, rng)
            ceps.append(c)
            ci_lo.append(lo)
            ci_hi.append(hi)
            print(f"tau={tau:>5.2f}s {key:>6}: CEP={c:8.2f} m  CI=[{lo:.2f}, {hi:.2f}]", flush=True)
        curves[key] = {"cep_m": ceps, "ci95_lo_m": ci_lo, "ci95_hi_m": ci_hi}

    out = {
        "meta": result_meta(args.config),
        "assumed_measurement_stats": assumed,
        "tau_s": taus,
        "n_runs_per_point": n_runs,
        "delay_comp": curves["comp"],
        "no_delay_comp": curves["uncomp"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, marker in (("comp", "delay compensation", "o"), ("uncomp", "no compensation", "s")):
        c = curves[key]
        yerr = np.vstack([
            np.array(c["cep_m"]) - np.array(c["ci95_lo_m"]),
            np.array(c["ci95_hi_m"]) - np.array(c["cep_m"]),
        ])
        ax.errorbar(taus, c["cep_m"], yerr=yerr, marker=marker, capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("inference latency τ [s]")
    ax.set_ylabel("CEP [m]")
    ax.set_title(f"Landing CEP vs τ, n={n_runs}/point (assumed measurement stats)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"fig: {fig_path}")


if __name__ == "__main__":
    main()
