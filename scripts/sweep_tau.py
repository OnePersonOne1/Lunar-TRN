"""τ 스윕 CLI.

기본(레거시): tau.sweep_s × (보상/미보상) → 기존 스키마 json·그림 (P1/P7 재현용).
--serial on 또는 --comp on/off 또는 --measured-points: 조건 리스트 모드(P7b) —
격자(tau.sweep_serial_s | sweep_s) + 실측 점(tau.measured_points의 median_s),
조건별 CEP·CI·타원·측정 수·드롭 수·게이트 수락률을 기록하고
CEP(좌)·밴드 내 측정 수(우) 이중 축 그림을 그린다. 미보상 곡선은 그리지 않는다.
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


def _legacy_sweep(cfg: dict, args, n_runs: int, workers: int, n_boot: int) -> None:
    """기존 동작: sweep_s × comp/uncomp, 기존 스키마·그림."""
    taus = [float(t) for t in cfg["tau"]["sweep_s"]]
    rng = np.random.default_rng(args.seed)
    curves: dict[str, dict] = {}
    assumed = measurement_R(cfg)[1]
    for comp in (True, False):
        key = "comp" if comp else "uncomp"
        ceps, ci_lo, ci_hi = [], [], []
        for tau in taus:
            xy, _, _ = run_mc(cfg, n_runs, workers, seed0=args.seed, tau=tau, delay_comp=comp)
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
    stats_tag = "assumed" if assumed else "calibrated"
    ax.set_title(f"Landing CEP vs τ, n={n_runs}/point ({stats_tag} measurement stats)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"fig: {fig_path}")


def _conditions_sweep(cfg: dict, args, n_runs: int, workers: int, n_boot: int) -> None:
    """P7b 모드: 격자+실측 점 조건 리스트, 단일 comp 조건, 직렬 처리 통계 포함."""
    serial = args.serial == "on"
    cfg["tau"]["serial"] = serial
    comp = args.comp != "off"  # on(기본) | off
    grid_key = "sweep_serial_s" if serial and "sweep_serial_s" in cfg["tau"] else "sweep_s"
    points: list[dict] = [{"tau_s": float(t), "label": None} for t in cfg["tau"][grid_key]]
    if args.measured_points:
        for mp in cfg["tau"]["measured_points"]:
            data = json.loads(Path(mp["file"]).read_text(encoding="utf-8"))
            points.append({"tau_s": float(data["median_s"]), "label": str(mp["label"]),
                           "file": mp["file"]})
    points.sort(key=lambda p: p["tau_s"])

    rng = np.random.default_rng(args.seed)
    assumed = measurement_R(cfg)[1]
    conditions: list[dict] = []
    fp_offset_used = None
    for p in points:
        xy, _, extras = run_mc(cfg, n_runs, workers, seed0=args.seed,
                               tau=p["tau_s"], delay_comp=comp)
        fp_offset_used = extras["fp_offset_used_m"]
        lo, hi = bootstrap_cep_ci(xy, n_boot, rng)
        summ = extras_summary(extras)
        cond = {
            "tau_s": p["tau_s"],
            "label": p["label"],
            "tau_file": p.get("file"),
            "cep_m": cep(xy),
            "cep_ci95_m": [lo, hi],
            "ellipse95": error_ellipse_95(xy),
            "n_runs": n_runs,
            "mean_n_meas": summ["mean_n_meas"],
            "mean_n_dropped": summ["mean_n_dropped"],
            "mean_gate_accept": summ["mean_gate_accept"],
            "delta_v_mps": summ["delta_v_mps"],
        }
        conditions.append(cond)
        tag = p["label"] or "grid"
        print(f"tau={p['tau_s']:>7.4f}s [{tag}]: CEP={cond['cep_m']:8.2f} m "
              f"CI=[{lo:.2f}, {hi:.2f}]  n_meas={cond['mean_n_meas']:.1f} "
              f"dropped={cond['mean_n_dropped']:.1f}", flush=True)

    out = {
        "meta": result_meta(args.config),
        "assumed_measurement_stats": assumed,
        "serial": serial,
        "delay_comp": comp,
        "fp_offset_used_m": fp_offset_used,
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

    taus = [c["tau_s"] for c in conditions]
    ceps = [c["cep_m"] for c in conditions]
    yerr = np.vstack([
        np.array(ceps) - np.array([c["cep_ci95_m"][0] for c in conditions]),
        np.array([c["cep_ci95_m"][1] for c in conditions]) - np.array(ceps),
    ])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(taus, ceps, yerr=yerr, marker="o", capsize=3, color="tab:blue",
                label="CEP (delay comp, serial)" if serial else "CEP (delay comp)")
    ax.set_xscale("log")
    ax.set_xlabel("inference latency τ [s]")
    ax.set_ylabel("CEP [m]", color="tab:blue")
    ax.grid(True, which="both", alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(taus, [c["mean_n_meas"] for c in conditions], marker="s", ls="--",
             color="tab:orange", label="in-band measurements")
    ax2.set_ylabel("mean measurements in TRN band", color="tab:orange")

    frame_period = 1.0 / float(cfg["camera"]["rate_hz"])
    ax.axvline(frame_period, color="gray", ls=":", lw=1.5)
    ax.annotate(f"frame period {frame_period:g}s", xy=(frame_period, ax.get_ylim()[1]),
                xytext=(3, -12), textcoords="offset points", fontsize=8, color="gray")
    for c in conditions:
        if c["label"]:
            ax.annotate(c["label"], xy=(c["tau_s"], c["cep_m"]),
                        xytext=(4, 6), textcoords="offset points", fontsize=8, rotation=30)
    stats_tag = "assumed" if assumed else "calibrated"
    ax.set_title(f"Landing CEP vs τ (serial={'on' if serial else 'off'}, "
                 f"comp={'on' if comp else 'off'}), n={n_runs}/point "
                 f"({stats_tag} measurement stats)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"fig: {fig_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/p1_tau_sweep.json")
    ap.add_argument("--fig", default="figs/p1_tau_sweep.png")
    ap.add_argument("--n-runs", type=int, default=None, help="τ·조건당 시행 수 (기본 mc.n_runs)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--serial", choices=["on", "off"], default="off",
                    help="직렬 처리 모델 — on이면 조건 리스트 모드(tau.sweep_serial_s)")
    ap.add_argument("--comp", choices=["on", "off", "both"], default="both",
                    help="지연 보상 — both(기본)는 레거시 두 곡선 모드")
    ap.add_argument("--measured-points", action="store_true",
                    help="tau.measured_points(실측 median)를 조건에 추가 — 조건 리스트 모드")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    n_runs = args.n_runs if args.n_runs is not None else int(cfg["mc"]["n_runs"])
    workers = args.workers if args.workers is not None else int(cfg["mc"]["workers"])
    n_boot = int(cfg["mc"]["bootstrap_n"])

    if args.serial == "on" or args.comp != "both" or args.measured_points:
        if args.comp == "both":
            raise SystemExit("조건 리스트 모드에서는 --comp on|off 를 명시해라 (미보상 곡선은 그리지 않는다)")
        _conditions_sweep(cfg, args, n_runs, workers, n_boot)
    else:
        _legacy_sweep(cfg, args, n_runs, workers, n_boot)


if __name__ == "__main__":
    main()
