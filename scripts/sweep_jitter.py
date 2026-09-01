"""타임스탬프 지터 스윕 CLI: 보상 성립 조건 ① — t_c 오차 1σ vs CEP.

지연 보상은 촬영시각 t_c로 되감아 보정한다. t_c에 오차(1σ = tau.t_c_jitter_sweep_s)가
있으면 잘못된 시점으로 되감아 보상 품질이 떨어진다. 조건: τ = 실측 median(--tau-file,
기본 n INT8 CPU), serial on, comp on. 그림에 무지터 보상(하한)·미보상(상한, 같은 τ의
p7b_tau_serial_compoff.json) 참조선을 넣는다. IMU 스텝(10 ms) 미만 지터는 스냅샷
반올림에 흡수되어 효과가 없어야 정상이다.
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
    ap.add_argument("--out", default="results/p7b_jitter_sweep.json")
    ap.add_argument("--fig", default="figs/p7b_cep_vs_jitter.png")
    ap.add_argument("--tau-file", default="results/tau_ort_cpu_int8.json")
    ap.add_argument("--comp-off-ref", default="results/p7b_tau_serial_compoff.json",
                    help="미보상 참조선(같은 τ 라벨) — 없으면 생략")
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
    jitters = [float(v) for v in cfg["tau"]["t_c_jitter_sweep_s"]]
    rng = np.random.default_rng(args.seed)
    assumed = measurement_R(cfg)[1]

    conditions = []
    for jit in jitters:
        xy, _, extras = run_mc(cfg, n_runs, workers, seed0=args.seed,
                               tau=tau, delay_comp=True, t_c_jitter=jit)
        lo, hi = bootstrap_cep_ci(xy, n_boot, rng)
        summ = extras_summary(extras)
        conditions.append({
            "t_c_jitter_s": jit, "cep_m": cep(xy), "cep_ci95_m": [lo, hi],
            "ellipse95": error_ellipse_95(xy), "n_runs": n_runs,
            "mean_gate_accept": summ["mean_gate_accept"],
            "mean_n_meas": summ["mean_n_meas"],
        })
        print(f"jitter={jit * 1e3:6.0f} ms: CEP={conditions[-1]['cep_m']:8.2f} m  "
              f"CI=[{lo:.2f}, {hi:.2f}]  gate={summ['mean_gate_accept']:.3f}", flush=True)

    comp_off_cep = None
    ref_path = Path(args.comp_off_ref)
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        near = min(ref["conditions"], key=lambda c: abs(c["tau_s"] - tau))
        if abs(near["tau_s"] - tau) < 0.01:
            comp_off_cep = near["cep_m"]

    out = {
        "meta": result_meta(args.config),
        "assumed_measurement_stats": assumed,
        "serial": True, "delay_comp": True,
        "tau_s": tau, "tau_file": args.tau_file,
        "comp_off_ref_cep_m": comp_off_cep,
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

    xs = [c["t_c_jitter_s"] * 1e3 for c in conditions]
    ys = [c["cep_m"] for c in conditions]
    yerr = np.vstack([
        np.array(ys) - np.array([c["cep_ci95_m"][0] for c in conditions]),
        np.array([c["cep_ci95_m"][1] for c in conditions]) - np.array(ys)])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, color="tab:blue",
                label="comp ON + timestamp jitter")
    ax.axhline(ys[0], color="tab:blue", ls=":", lw=1.2, alpha=0.7)
    ax.annotate("no-jitter comp ON", (xs[-1], ys[0]), fontsize=8, color="tab:blue",
                textcoords="offset points", xytext=(-90, 5))
    if comp_off_cep is not None:
        ax.axhline(comp_off_cep, color="tab:red", ls="--", lw=1.2)
        ax.annotate("comp OFF (same τ)", (xs[0], comp_off_cep), fontsize=8,
                    color="tab:red", textcoords="offset points", xytext=(4, 5))
    imu_dt_ms = 1e3 / float(cfg["imu"]["rate_hz"])
    ax.axvline(imu_dt_ms, color="gray", ls=":", lw=1.0)
    ax.annotate("IMU step", (imu_dt_ms, ax.get_ylim()[1]), fontsize=8, color="gray",
                textcoords="offset points", xytext=(3, -14))
    stats_tag = "assumed" if assumed else "calibrated"
    ax.set_xlabel("capture-timestamp error 1σ [ms]")
    ax.set_ylabel("CEP [m]")
    ax.set_title(f"Delay compensation vs timestamp error (τ={tau * 1e3:.0f} ms, serial), "
                 f"n={n_runs}/point ({stats_tag} stats)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"fig: {fig_path}")


if __name__ == "__main__":
    main()
