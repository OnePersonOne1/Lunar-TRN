"""τ 스케일링 요약 CLI: 양자화·모델 변경·보상 유무 효과 표 + 온보드 등급 환산 그림.

입력(전부 기존 results): p7b_tau_serial.json(보상), p7b_tau_serial_compoff.json(미보상),
p7b_cpu_bench.json(등급별 성능비). 출력: results/p7b_tau_scaling.json,
figs/p7b_tau_scaling.png, stdout 마크다운 표.
환산 τ = 실측 τ × 벤치 성능비(1차 근사 — 메모리·SIMD 차이 미반영, 문서 §2 규칙 준수:
스윕 격자(5 s) 밖은 외삽하지 않고 "성립 불가"로만 판정한다.
RTG4급 FPGA 구간(0.3~1.5 s)은 docs/onboard_cpu_emulation.md §1b의 문헌 기반 1차 추정.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mc import result_meta  # noqa: E402

CPU_LABELS = ["n INT8 CPU", "n FP32 CPU", "s INT8 CPU", "s FP32 CPU"]
FPGA_EST_S = (0.3, 1.5)  # RTG4급 FPGA+CNN INT8 1차 추정 (docs/onboard_cpu_emulation.md §1b)


def ci_separated(ci_a: list[float], ci_b: list[float]) -> bool:
    """부트스트랩 95% CI가 겹치지 않으면 True (효과 유의로 취급)."""
    return ci_a[0] > ci_b[1] or ci_b[0] > ci_a[1]


def verdict(tau_scaled: float, frame_period_s: float, grid_max_s: float) -> str:
    """환산 τ의 판정. 격자 밖은 외삽하지 않는다."""
    if tau_scaled <= frame_period_s:
        return "평탄 구간 — 보상 시 흡수"
    if tau_scaled <= grid_max_s:
        return "드롭 영역 — 측정 손실로 악화"
    return "격자 밖 — 성립 불가(외삽 금지)"


def _cond_map(path: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(c.get("label") or f"grid {c['tau_s']:g}s"): c for c in d["conditions"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # CLI 규약 통일용
    ap.add_argument("--out", default="results/p7b_tau_scaling.json")
    ap.add_argument("--fig", default="figs/p7b_tau_scaling.png")
    ap.add_argument("--comp-on", default="results/p7b_tau_serial.json")
    ap.add_argument("--comp-off", default="results/p7b_tau_serial_compoff.json")
    ap.add_argument("--bench", default="results/p7b_cpu_bench.json")
    args = ap.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    frame_period = 1.0 / float(cfg["camera"]["rate_hz"])
    grid_max = max(float(t) for t in cfg["tau"]["sweep_serial_s"])

    on, off = _cond_map(args.comp_on), _cond_map(args.comp_off)
    bench = json.loads(Path(args.bench).read_text(encoding="utf-8"))
    sp = bench["speedup_vs_reference"]

    # 표 1: 현재 실측(차세대 CPU 프록시)에서 양자화·모델·보상 효과
    rows = []
    for lab in CPU_LABELS:
        a, b = on[lab], off[lab]
        rows.append({
            "config": lab, "tau_ms": a["tau_s"] * 1e3,
            "cep_comp_on_m": a["cep_m"], "ci_on": a["cep_ci95_m"],
            "cep_comp_off_m": b["cep_m"], "ci_off": b["cep_ci95_m"],
        })

    def eff(lab_hi: str, lab_lo: str, name: str) -> dict:
        hi, lo = off[lab_hi], off[lab_lo]
        return {
            "effect": name,
            "from": lab_hi, "to": lab_lo,
            "tau_gain_ms": (hi["tau_s"] - lo["tau_s"]) * 1e3,
            "cep_off_from_m": hi["cep_m"], "cep_off_to_m": lo["cep_m"],
            "cep_off_delta_pct": 100.0 * (lo["cep_m"] - hi["cep_m"]) / hi["cep_m"],
            "significant_ci95": ci_separated(hi["cep_ci95_m"], lo["cep_ci95_m"]),
            "cep_on_from_m": on[lab_hi]["cep_m"], "cep_on_to_m": on[lab_lo]["cep_m"],
        }

    effects = [
        eff("n FP32 CPU", "n INT8 CPU", "양자화 (n)"),
        eff("s FP32 CPU", "s INT8 CPU", "양자화 (s)"),
        eff("s INT8 CPU", "n INT8 CPU", "모델 경량화 (INT8)"),
        eff("s FP32 CPU", "n FP32 CPU", "모델 경량화 (FP32)"),
    ]

    # 표 2: 온보드 등급 환산 (벤치 성능비 × 실측 τ)
    classes = [
        {"name": "차세대 HPSC급 (벤치 동급 자릿수 — 가정 ×1)", "ratio": [1.0, 1.0]},
        {"name": "GR740급", "ratio": sorted([
            sp["Gaisler GR740 (quad LEON4FT, 250 MHz)"]["coremark_x"],
            sp["Gaisler GR740 (quad LEON4FT, 250 MHz)"]["dmips_x"]])},
        {"name": "HR5000/RAD750급", "ratio": sorted([
            sp["BAE RAD750 (200 MHz)"]["dmips_x"],
            sp["JAXA HR5000 계열 (MIPS64 5Kf, 200 MHz)"]["dmips_x"]])},
    ]
    scaled = []
    for cl in classes:
        for lab in CPU_LABELS:
            t = on[lab]["tau_s"]
            lo_s, hi_s = t * cl["ratio"][0], t * cl["ratio"][1]
            scaled.append({
                "class": cl["name"], "config": lab,
                "tau_scaled_s": [lo_s, hi_s],
                "verdict": verdict(hi_s, frame_period, grid_max),
            })

    out = {
        "meta": result_meta(args.config),
        "note": ("환산 τ = 실측 τ × 1스레드 벤치 성능비(1차 근사). 격자 밖 외삽 금지. "
                 "RTG4급 FPGA 0.3~1.5 s는 docs/onboard_cpu_emulation.md §1b 문헌 기반 추정"),
        "frame_period_s": frame_period, "grid_max_s": grid_max,
        "fpga_est_s": list(FPGA_EST_S),
        "measured_table": rows, "effects": effects, "scaled_classes": scaled,
        "sources": {"comp_on": args.comp_on, "comp_off": args.comp_off, "bench": args.bench},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("| 구성 | τ [ms] | 보상 CEP [m] | 미보상 CEP [m] |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['config']} | {r['tau_ms']:.1f} | {r['cep_comp_on_m']:.1f} "
              f"[{r['ci_on'][0]:.0f},{r['ci_on'][1]:.0f}] | {r['cep_comp_off_m']:.1f} "
              f"[{r['ci_off'][0]:.0f},{r['ci_off'][1]:.0f}] |")
    print()
    print("| 효과 | τ 이득 [ms] | 미보상 CEP 변화 | 유의(CI 분리) |")
    print("|---|---|---|---|")
    for e in effects:
        print(f"| {e['effect']} | {e['tau_gain_ms']:.1f} | {e['cep_off_from_m']:.1f}"
              f"→{e['cep_off_to_m']:.1f} ({e['cep_off_delta_pct']:+.0f}%) "
              f"| {'예' if e['significant_ci95'] else '아니오'} |")
    print(f"\nsaved: {out_path}")

    # 그림: CEP vs τ (보상/미보상) + 실측점 + 프레임 주기 + FPGA 추정 구간
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cmap, color, name in ((on, "tab:blue", "delay comp ON"),
                              (off, "tab:red", "delay comp OFF")):
        conds = sorted(cmap.values(), key=lambda c: c["tau_s"])
        taus = [c["tau_s"] for c in conds]
        ceps = [c["cep_m"] for c in conds]
        yerr = np.vstack([
            np.array(ceps) - np.array([c["cep_ci95_m"][0] for c in conds]),
            np.array([c["cep_ci95_m"][1] for c in conds]) - np.array(ceps)])
        ax.errorbar(taus, ceps, yerr=yerr, marker="o", ms=4, capsize=3,
                    color=color, label=name)
        for i, c in enumerate(conds):
            if c.get("label") in CPU_LABELS:
                up = CPU_LABELS.index(c["label"]) % 2 == 0  # 인접 라벨 상하 교차 배치
                ax.annotate(c["label"], (c["tau_s"], c["cep_m"]),
                            textcoords="offset points",
                            xytext=(4, 7 if up else -13), fontsize=7, color=color)
    ax.axvline(frame_period, color="gray", ls="--", lw=1.2)
    ax.annotate("frame period", xy=(frame_period, 0.985), xycoords=("data", "axes fraction"),
                xytext=(4, -6), textcoords="offset points", fontsize=8, color="gray")
    ax.axvspan(*FPGA_EST_S, alpha=0.10, color="tab:green")
    ax.annotate("RTG4-class FPGA+CNN (est.)", xy=(FPGA_EST_S[0] * 1.05, 0.90),
                xycoords=("data", "axes fraction"), fontsize=8, color="tab:green")
    hr = sp["JAXA HR5000 계열 (MIPS64 5Kf, 200 MHz)"]["dmips_x"]
    ax.annotate(f"legacy CPU classes: τ×68~{hr:.0f} → 13~25 s (off-grid, infeasible) →",
                xy=(0.99, 0.03), xycoords="axes fraction", ha="right", fontsize=8,
                color="dimgray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("τ [s] (serial capture model)")
    ax.set_ylabel("CEP [m]")
    ax.set_title("Landing CEP vs τ — onboard-class mapping "
                 "(calibrated stats, n=200/point)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"fig: {fig_path}")


if __name__ == "__main__":
    main()
