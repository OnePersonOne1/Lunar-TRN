"""슬라이드용 다이어그램 생성 CLI (P8): 파이프라인·직렬 처리/지연 보상 모식도.

출력: figs/slides/slide_05_pipeline.png, figs/slides/slide_06_delay_comp.png.
수치 라벨은 results 파일에서만 읽는다(τ median). 폰트는 Windows 한글(Malgun Gothic).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BOX_FC = "#eef3fb"
BOX_EC = "#3a6ea5"
ACCENT = "#c0392b"
GREEN = "#1e8449"
GRAY = "#666666"


def _box(ax, x, y, w, h, title, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=BOX_FC, ec=BOX_EC, lw=1.6))
    cy = y + h * (0.62 if sub else 0.5)
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=13, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.24, sub, ha="center", va="center",
                fontsize=8.5, color=GRAY)


def _arrow(ax, x0, y0, x1, y1, label=None, color="black", ls="-", lift=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=16, color=color, ls=ls, lw=1.4))
    if label:
        ax.annotate(label, ((x0 + x1) / 2, max(y0, y1)), textcoords="offset points",
                    xytext=(0, lift), ha="center", fontsize=8.5, color=color)


def pipeline(out: Path) -> None:
    """핵심 블록 6개 + 화살표만. 블록 제목 외 텍스트 없음."""
    fig, ax = plt.subplots(figsize=(12.5, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 24)
    ax.axis("off")

    w, h, y = 13.5, 8.5, 10.0
    xs = [2, 18.5, 35, 51.5, 68, 84.5]
    titles = ["3-DOF 동역학", "Unity 렌더", "YOLO11n INT8", "PnP\n(지도 및 위치 매칭)", "EKF", "ZEM/ZEV 유도"]
    for x, t in zip(xs, titles):
        _box(ax, x, y, w, h, t)
    for i in range(5):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    # 피드백 루프: 유도 → 동역학 (추력, 아래), 동역학 → EKF (IMU, 위) — 라벨 없음
    ax.add_patch(FancyArrowPatch((xs[5] + w / 2, y), (xs[0] + w / 2, y),
                                 arrowstyle="-|>", mutation_scale=16, color=ACCENT,
                                 lw=1.6, connectionstyle="arc3,rad=-0.18"))
    ax.add_patch(FancyArrowPatch((xs[0] + w / 2, y + h), (xs[4] + w / 2, y + h),
                                 arrowstyle="-|>", mutation_scale=14, color=GRAY,
                                 lw=1.2, ls="--", connectionstyle="arc3,rad=-0.12"))

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def delay_comp(out: Path) -> None:
    """지연 보상(되감기-재전파) 단독 다이어그램 — 직렬 처리 패널은 슬라이드에서 제외."""
    fig, ax2 = plt.subplots(figsize=(12.5, 3.4))

    ax2.set_xlim(-0.3, 8.8)
    ax2.set_ylim(-2.15, 2.1)
    ax2.axis("off")
    ax2.set_title("지연 보상 — 도착한 측정을 촬영 시각으로 되감아 보정 후 재전파 (성립 조건: 정확한 t_c)",
                  fontsize=11.5, loc="left")
    ax2.axhline(0, color="black", lw=1)
    t_c, t_arr = 2.0, 4.5
    for t in range(9):
        ax2.plot([t, t], [-0.09, 0.09], color="black", lw=1)
    # 링버퍼 (스냅샷 칸)
    for i in range(16):
        x = 0.5 + i * 0.47
        fc = "#fdebd0" if 2.0 <= x <= 4.5 else "#f2f3f4"
        ax2.add_patch(Rectangle((x, -1.5), 0.42, 0.42, fc=fc, ec=GRAY, lw=0.7))
    ax2.text(0.45, -1.9, r"링버퍼: (t, $\hat{x}$, P, a_IMU) 스냅샷 — 색칠 구간 = 재전파 대상",
             fontsize=8.5, color=GRAY)
    ax2.plot(t_c, 0, "o", color=GREEN, ms=9, zorder=5)
    ax2.text(t_c, 0.28, "촬영 t_c", ha="center", fontsize=9, color=GREEN)
    ax2.plot(t_arr, 0, "s", color=BOX_EC, ms=8, zorder=5)
    ax2.text(t_arr, 0.28, "z 도착 (t_c + τ)", ha="center", fontsize=9, color=BOX_EC)
    ax2.add_patch(FancyArrowPatch((t_arr - 0.1, 0.58), (t_c + 0.1, 0.58), arrowstyle="-|>",
                                  mutation_scale=15, color=ACCENT, lw=1.5,
                                  connectionstyle="arc3,rad=0.5"))
    ax2.text((t_c + t_arr) / 2, 1.72, "① t_c 스냅샷으로 되감기 → 보정", ha="center",
             fontsize=9.5, color=ACCENT)
    ax2.add_patch(FancyArrowPatch((t_c, -0.45), (t_arr + 1.6, -0.45), arrowstyle="-|>",
                                  mutation_scale=15, color=GREEN, lw=1.5))
    ax2.text((t_c + t_arr + 1.6) / 2, -0.85, "② 저장된 IMU 입력으로 현재까지 재전파",
             ha="center", fontsize=9.5, color=GREEN)

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # CLI 규약 통일용
    ap.add_argument("--out-dir", default="figs/slides")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / "slide_05_pipeline.png"
    pipeline(p1)
    print(f"fig: {p1}")
    p2 = out_dir / "slide_06_delay_comp.png"
    delay_comp(p2)
    print(f"fig: {p2}")


if __name__ == "__main__":
    main()
