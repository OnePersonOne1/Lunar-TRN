"""P9 매트릭스 집계 CLI: 탐지기 × 온보드 등급 CEP 히트맵 + mAP vs CEP 산점도.

숫자는 results/p9_matrix.json에서만 읽는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INK, GRAY = "#3B4148", "#8A93A0"
CLS_COL = {"next_gen": "#1452C7", "current_gen": "#D16608", "legacy": "#8A5EC7"}
DET_ORDER = ["n_int8", "n_fp32", "s_int8", "s_fp32", "classic_pca"]
DET_LABEL = {"n_int8": "YOLO n INT8", "n_fp32": "YOLO n FP32", "s_int8": "YOLO s INT8",
             "s_fp32": "YOLO s FP32", "classic_pca": "classic PCA"}
CLS_ORDER = ["next_gen", "current_gen", "legacy"]
CLS_LABEL = {"next_gen": "next-gen\n(HPSC-class, x1)",
             "current_gen": "current-gen\n(GR740-class, x96)",
             "legacy": "legacy\n(HR5000-class, x127)"}


def heatmap(d: dict, comp: bool, ax) -> None:
    import matplotlib.pyplot as plt  # noqa: F401

    conds = {(c["detector"], c["class"]): c for c in d["conditions"]
             if c["delay_comp"] == comp}
    M = np.array([[conds[(det, cls)]["cep_m"] for cls in CLS_ORDER] for det in DET_ORDER])
    ax.imshow(np.log10(M), cmap="YlOrRd", aspect="auto",
              vmin=np.log10(80), vmax=np.log10(6000))
    for i, det in enumerate(DET_ORDER):
        for j, cls in enumerate(CLS_ORDER):
            c = conds[(det, cls)]
            txt = f"{c['cep_m']:.0f}" if c["cep_m"] < 10000 else f"{c['cep_m']/1e3:.1f}k"
            ax.text(j, i - 0.12, txt, ha="center", va="center", fontsize=11,
                    color=INK, fontweight="bold")
            ax.text(j, i + 0.24, f"meas {c['mean_n_meas']:.0f}", ha="center",
                    va="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(CLS_ORDER)), [CLS_LABEL[c] for c in CLS_ORDER], fontsize=9)
    ax.set_yticks(range(len(DET_ORDER)), [DET_LABEL[k] for k in DET_ORDER], fontsize=9)
    ax.set_title(f"CEP [m], delay comp. {'ON' if comp else 'OFF'}",
                 fontsize=11, color=INK)


def fig_heatmaps(d: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, comp in zip(axes, (True, False)):
        heatmap(d, comp, ax)
    fig.suptitle("Landing CEP: detector x onboard compute class "
                 f"(precision-map assumption, n={d['params']['n_runs']}/cell, 1 Hz serial)",
                 fontsize=12, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"fig: {out}")


def fig_map_vs_cep(d: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = {dd["key"]: dd["mAP50_95"] for dd in d["params"]["detectors"]}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cls in CLS_ORDER:
        pts = [(maps[c["detector"]], c["cep_m"], c["detector"])
               for c in d["conditions"]
               if c["class"] == cls and c["delay_comp"] and maps[c["detector"]] is not None]
        xs, ys, _ = zip(*pts)
        ax.plot(xs, ys, "o", ms=8, color=CLS_COL[cls], alpha=0.85,
                label=CLS_LABEL[cls].replace("\n", " "))
    # 차세대 YOLO 4종은 한 점처럼 겹침 — 군집 라벨 하나로
    yolo_next = [(maps[c["detector"]], c["cep_m"]) for c in d["conditions"]
                 if c["class"] == "next_gen" and c["delay_comp"]
                 and c["detector"] != "classic_pca"]
    xs, ys = zip(*yolo_next)
    ax.annotate(f"YOLO n/s x FP32/INT8\n({min(ys):.0f}-{max(ys):.0f} m)",
                (float(np.mean(xs)), float(np.mean(ys))), xytext=(-30, 22),
                textcoords="offset points", ha="right", fontsize=9, color=INK)
    classic_next = next(c for c in d["conditions"] if c["class"] == "next_gen"
                        and c["delay_comp"] and c["detector"] == "classic_pca")
    ax.annotate("classic PCA", (maps["classic_pca"], classic_next["cep_m"]),
                xytext=(10, -4), textcoords="offset points", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("detector mAP50-95 (open-loop)")
    ax.set_ylabel("landing CEP [m] (log)")
    ax.grid(True, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("mAP alone does not order landing accuracy — compute class does too\n"
                 "(delay comp. ON, precision-map assumption)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"fig: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # 결정론 — 미사용
    ap.add_argument("--matrix", default="results/p9_matrix.json")
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()

    d = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    fig_heatmaps(d, Path(args.out) / "p9_matrix_heatmap.png")
    fig_map_vs_cep(d, Path(args.out) / "p9_map_vs_cep.png")


if __name__ == "__main__":
    main()
