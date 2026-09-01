"""오검출률 스윕 CLI (연구계획서 산출물 ④): measurement.fp_sweep 각 값 MC → CEP 곡선.

조건: τ = 실측 median(기본 n INT8 CPU, --tau-file), delay_comp on, serial on,
오프셋 = calibrated 파일의 fp_offset_med_m(measurement.mode=calibrated일 때).
출력: 조건별 CEP·CI·게이트 기각 비율·수락 측정 수 json + CEP(좌)·기각 비율(우) 그림.
보정 오검출률(measurement 파일의 fp_rate_est) 위치에 세로선.
"""
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
    ap.add_argument("--out", default="results/p7b_fp_sweep.json")
    ap.add_argument("--fig", default="figs/p7b_cep_vs_fp.png")
    ap.add_argument("--tau-file", default="results/tau_ort_cpu_int8.json",
                    help="τ로 쓸 실측 파일 (median_s)")
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["tau"]["serial"] = True
    n_runs = args.n_runs if args.n_runs is not None else int(cfg["mc"]["n_runs"])
    workers = args.workers if args.workers is not None else int(cfg["mc"]["workers"])
    n_boot = int(cfg["mc"]["bootstrap_n"])
    tau = float(json.loads(Path(args.tau_file).read_text(encoding="utf-8"))["median_s"])
    fp_values = [float(v) for v in cfg["measurement"]["fp_sweep"]]
    rng = np.random.default_rng(args.seed)
    assumed = measurement_R(cfg)[1]

    # 보정 오검출률(세로선 기준값) — calibrated 파일이 있으면 fp_rate_est
    fp_calibrated = None
    model_path = Path(cfg["measurement"]["file"])
    if model_path.exists():
        model = json.loads(model_path.read_text(encoding="utf-8"))
        fp_calibrated = model.get("fp_rate_est")

    conditions: list[dict] = []
    fp_offset_used = None
    for fp in fp_values:
        xy, _, extras = run_mc(cfg, n_runs, workers, seed0=args.seed,
                               tau=tau, fp_rate=fp, delay_comp=True)
        fp_offset_used = extras["fp_offset_used_m"]
        lo, hi = bootstrap_cep_ci(xy, n_boot, rng)
        summ = extras_summary(extras)
        cond = {
            "fp_rate": fp,
            "cep_m": cep(xy),
            "cep_ci95_m": [lo, hi],
            "ellipse95": error_ellipse_95(xy),
            "n_runs": n_runs,
            "gate_reject_ratio": 1.0 - summ["mean_gate_accept"],
            "mean_n_accept": summ["mean_n_accept"],
            "mean_n_meas": summ["mean_n_meas"],
            "mean_n_dropped": summ["mean_n_dropped"],
        }
        conditions.append(cond)
        print(f"fp={fp:.2f}: CEP={cond['cep_m']:8.2f} m  CI=[{lo:.2f}, {hi:.2f}]  "
              f"reject={cond['gate_reject_ratio']:.3f}  accept={cond['mean_n_accept']:.1f}",
              flush=True)

    out = {
        "meta": result_meta(args.config),
        "assumed_measurement_stats": assumed,
        "serial": True,
        "delay_comp": True,
        "tau_s": tau,
        "tau_file": args.tau_file,
        "fp_offset_used_m": fp_offset_used,
        "fp_rate_calibrated": fp_calibrated,
        "n_runs_per_point": n_runs,
        "conditions": conditions,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps = [c["fp_rate"] for c in conditions]
    ceps = [c["cep_m"] for c in conditions]
    yerr = np.vstack([
        np.array(ceps) - np.array([c["cep_ci95_m"][0] for c in conditions]),
        np.array([c["cep_ci95_m"][1] for c in conditions]) - np.array(ceps),
    ])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(fps, ceps, yerr=yerr, marker="o", capsize=3, color="tab:blue", label="CEP")
    ax.set_xlabel("false-detection rate")
    ax.set_ylabel("CEP [m]", color="tab:blue")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(fps, [c["gate_reject_ratio"] for c in conditions], marker="s", ls="--",
             color="tab:red", label="gate reject ratio")
    ax2.set_ylabel("gate reject ratio", color="tab:red")

    if fp_calibrated is not None:
        ax.axvline(fp_calibrated, color="gray", ls=":", lw=1.5)
        ax.annotate(f"calibrated {fp_calibrated:.3f}", xy=(fp_calibrated, ax.get_ylim()[1]),
                    xytext=(3, -12), textcoords="offset points", fontsize=8, color="gray")
    stats_tag = "assumed" if assumed else "calibrated"
    ax.set_title(f"Landing CEP vs false-detection rate (τ={tau * 1e3:.0f} ms, serial, comp on), "
                 f"n={n_runs}/point ({stats_tag} measurement stats)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"fig: {fig_path}")


if __name__ == "__main__":
    main()
