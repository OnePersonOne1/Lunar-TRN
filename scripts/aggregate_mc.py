"""P7 MC 집계 CLI: results/p7_mc_*.json → p7_mc.json + CEP vs τ·산포도 그림.

--emit-commands: 조건별 run_mc.py 실행 명령 세트만 출력한다(사람이 tmux에서 돌린다).
모든 그림 제목에 "preliminary, n=…; full MC scheduled Oct"를 넣는다.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mc import result_meta  # noqa: E402


def measured_tau_medians() -> dict[str, float]:
    """P2/P5 벤치 산출물에서 실측 τ median을 모은다 (n 3종 + s CPU INT8 비교점)."""
    paths = {
        "trt_int8": "results/tau_trt_int8.json",
        "ort_cpu_fp32": "results/tau_ort_cpu_fp32.json",
        "ort_cpu_int8": "results/tau_ort_cpu_int8.json",
        "s_ort_cpu_int8": "results/s/tau_ort_cpu_int8.json",
    }
    out = {}
    for key, path in paths.items():
        p = Path(path)
        if p.exists():
            out[key] = float(json.loads(p.read_text(encoding="utf-8"))["median_s"])
    return out


def emit_commands(cfg: dict, seed: int) -> None:
    taus = sorted(set(float(t) for t in cfg["tau"]["sweep_s"]) | set(measured_tau_medians().values()))
    py = ".venv\\Scripts\\python"
    print("# P7 예비 MC 명령 세트 (조건별 순차 실행; 워커는 run 내부 병렬)")
    for tau in taus:
        for comp in ("on", "off"):
            out = f"results/p7_mc_tau{tau:g}_comp{comp}.json"
            print(f"{py} scripts\\run_mc.py --config config.yaml --seed {seed} "
                  f"--tau {tau:g} --delay-comp {comp} --out {out}")
    print(f"{py} scripts\\aggregate_mc.py --config config.yaml --seed {seed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/p7_mc.json")
    ap.add_argument("--pattern", default="results/p7_mc_*.json")
    ap.add_argument("--fig-dir", default="figs")
    ap.add_argument("--emit-commands", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if args.emit_commands:
        emit_commands(cfg, args.seed)
        return

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"{args.pattern} 파일이 없다. --emit-commands로 명령을 먼저 뽑아 실행해라.")
    conds = []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        conds.append({
            "file": f,
            "tau_s": d["params"]["tau"],
            "delay_comp": d["params"]["delay_comp"],
            "n_runs": d["params"]["n_runs"],
            "cep_m": d["cep_m"],
            "cep_ci95_m": d["cep_ci95_m"],
            "ellipse95": d["ellipse95"],
            "assumed_measurement_stats": not Path(cfg["measurement"]["file"]).exists(),
            "landing_xy_m": d["landing_xy_m"],
        })
    n_runs = conds[0]["n_runs"]
    subtitle = f"preliminary, n={n_runs}; full MC scheduled Oct"
    assumed = any(c["assumed_measurement_stats"] for c in conds)
    if assumed:
        subtitle += " (assumed measurement stats)"

    out = {"meta": result_meta(args.config), "subtitle": subtitle,
           "conditions": [{k: v for k, v in c.items() if k != "landing_xy_m"} for c in conds]}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path} ({len(conds)} conditions)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- CEP vs τ
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for comp, label, marker in ((True, "delay compensation", "o"), (False, "no compensation", "s")):
        sel = sorted((c for c in conds if c["delay_comp"] == comp and c["tau_s"] is not None),
                     key=lambda c: c["tau_s"])
        if not sel:
            continue
        taus = [c["tau_s"] for c in sel]
        ceps = [c["cep_m"] for c in sel]
        yerr = np.array([[c["cep_m"] - c["cep_ci95_m"][0] for c in sel],
                         [c["cep_ci95_m"][1] - c["cep_m"] for c in sel]])
        ax.errorbar(taus, ceps, yerr=yerr, marker=marker, capsize=3, label=label)
    colors = {"trt_int8": "tab:green", "ort_cpu_fp32": "tab:red", "ort_cpu_int8": "tab:purple",
              "s_ort_cpu_int8": "tab:brown"}
    for key, tau in measured_tau_medians().items():
        ax.axvline(tau, linestyle=":", color=colors.get(key, "gray"), alpha=0.8,
                   label=f"measured {key} ({tau*1e3:.0f} ms)")
    ax.set_xscale("log")
    ax.set_xlabel("inference latency τ [s]")
    ax.set_ylabel("CEP [m]")
    ax.set_title(f"Landing CEP vs τ — {subtitle}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "p7_cep_vs_tau.png", dpi=150)
    print(f"fig: {fig_dir / 'p7_cep_vs_tau.png'}")

    # --- 조건별 산포도
    for c in conds:
        xy = np.asarray(c["landing_xy_m"])
        tag = f"tau{c['tau_s']:g}_comp{'on' if c['delay_comp'] else 'off'}"
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(xy[:, 0], xy[:, 1], s=6, alpha=0.5)
        ax.plot(0, 0, "r*", markersize=12)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{tag} — CEP {c['cep_m']:.1f} m\n{subtitle}", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"p7_scatter_{tag}.png", dpi=150)
        plt.close(fig)
    print(f"fig: {fig_dir}/p7_scatter_*.png ({len(conds)}장)")


if __name__ == "__main__":
    main()
