"""카메라 레이트 스윕 CLI: 보상 성립 조건 ③ — 프레임 주기 vs 드롭 임계.

camera.rate_sweep_hz 각 레이트 × 실측 CPU τ 4점(n/s × FP32/INT8, tau.measured_points의
CPU 라벨)을 serial on + comp on으로 MC. 레이트가 오르면 프레임 주기(=드롭 임계)가
τ 쪽으로 내려와, 보상을 켠 채로도 양자화·경량화(τ 감소)가 CEP를 가르는지 본다.
가정: 프레임당 측정 σ는 레이트와 무관(calibrated 값 그대로) — json note에 명기.
레이트는 imu.rate_hz의 약수여야 한다(스텝/프레임 정수).
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
    ap.add_argument("--out", default="results/p7b_rate_sweep.json")
    ap.add_argument("--fig", default="figs/p7b_cep_vs_rate.png")
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["tau"]["serial"] = True
    n_runs = args.n_runs if args.n_runs is not None else int(cfg["mc"]["n_runs"])
    workers = args.workers if args.workers is not None else int(cfg["mc"]["workers"])
    n_boot = int(cfg["mc"]["bootstrap_n"])
    imu_hz = float(cfg["imu"]["rate_hz"])
    rates = [float(r) for r in cfg["camera"]["rate_sweep_hz"]]
    rng = np.random.default_rng(args.seed)
    assumed = measurement_R(cfg)[1]

    points = []  # (label, tau_s) — 실측 CPU 지점만
    for mp in cfg["tau"]["measured_points"]:
        if "CPU" not in mp["label"]:
            continue
        med = float(json.loads(Path(mp["file"]).read_text(encoding="utf-8"))["median_s"])
        points.append((mp["label"], med))

    conditions = []
    for rate in rates:
        if abs(imu_hz / rate - round(imu_hz / rate)) > 1e-9:
            raise SystemExit(f"rate {rate} Hz가 imu.rate_hz({imu_hz})의 약수가 아니다")
        cfg_r = json.loads(json.dumps(cfg))  # per-rate 사본
        cfg_r["camera"]["rate_hz"] = rate
        for label, tau in points:
            xy, _, extras = run_mc(cfg_r, n_runs, workers, seed0=args.seed,
                                   tau=tau, delay_comp=True)
            lo, hi = bootstrap_cep_ci(xy, n_boot, rng)
            summ = extras_summary(extras)
            drop_ratio = summ["mean_n_dropped"] / max(
                summ["mean_n_meas"] + summ["mean_n_dropped"], 1e-9)
            conditions.append({
                "rate_hz": rate, "label": label, "tau_s": tau,
                "cep_m": cep(xy), "cep_ci95_m": [lo, hi],
                "ellipse95": error_ellipse_95(xy), "n_runs": n_runs,
                "mean_n_meas": summ["mean_n_meas"],
                "mean_n_dropped": summ["mean_n_dropped"],
                "drop_ratio": drop_ratio,
                "mean_gate_accept": summ["mean_gate_accept"],
            })
            print(f"rate={rate:3.0f} Hz  {label:>11}  τ={tau * 1e3:5.1f} ms: "
                  f"CEP={conditions[-1]['cep_m']:8.2f} m  CI=[{lo:.2f}, {hi:.2f}]  "
                  f"drop={drop_ratio:.2f}", flush=True)

    out = {
        "meta": result_meta(args.config),
        "assumed_measurement_stats": assumed,
        "serial": True, "delay_comp": True,
        "note": "프레임당 측정 σ는 레이트와 무관하다고 가정(calibrated 값 그대로)",
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

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {"n INT8 CPU": "tab:blue", "n FP32 CPU": "tab:cyan",
              "s INT8 CPU": "tab:red", "s FP32 CPU": "tab:orange"}
    for label, _ in points:
        rows = [c for c in conditions if c["label"] == label]
        xs = [c["rate_hz"] for c in rows]
        ys = [c["cep_m"] for c in rows]
        yerr = np.vstack([
            np.array(ys) - np.array([c["cep_ci95_m"][0] for c in rows]),
            np.array([c["cep_ci95_m"][1] for c in rows]) - np.array(ys)])
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3,
                    color=colors.get(label), label=label)
        for c in rows:
            if c["drop_ratio"] > 0.01:
                ax.annotate(f"drop {c['drop_ratio']:.0%}",
                            (c["rate_hz"], c["cep_m"]), fontsize=7,
                            color=colors.get(label),
                            textcoords="offset points", xytext=(5, 5))
    stats_tag = "assumed" if assumed else "calibrated"
    ax.set_xlabel("camera rate [Hz] (frame period = drop threshold)")
    ax.set_ylabel("CEP [m]")
    ax.set_xticks([float(r) for r in rates])
    ax.set_title(f"CEP vs camera rate — comp ON, serial (calibrated τ points), "
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
