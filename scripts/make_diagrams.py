"""슬라이드용 다이어그램 생성 CLI (P8): 파이프라인·직렬 처리/지연 보상 모식도.

출력: figs/slides/slide_05_pipeline.png, figs/slides/slide_06_serial_model.png.
수치 라벨은 results 파일에서만 읽는다(τ median). 폰트는 Windows 한글(Malgun Gothic).
"""
from __future__ import annotations

import argparse
import json
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
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=11, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.24, sub, ha="center", va="center",
                fontsize=8.5, color=GRAY)


def _arrow(ax, x0, y0, x1, y1, label=None, color="black", ls="-", lift=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=16, color=color, ls=ls, lw=1.4))
    if label:
        ax.annotate(label, ((x0 + x1) / 2, max(y0, y1)), textcoords="offset points",
                    xytext=(0, lift), ha="center", fontsize=8.5, color=color)


def pipeline(tau_ms: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    w, h, y = 13.5, 9.0, 14.5
    xs = [2, 18.5, 35, 51.5, 68, 84.5]
    _box(ax, xs[0], y, w, h, "3-DOF 동역학", "RK4 · 참값 상태")
    _box(ax, xs[1], y, w, h, "Unity 렌더", "센서 모사 (DEM·WAC)")
    _box(ax, xs[2], y, w, h, "YOLO11n INT8", f"크레이터 탐지 · τ={tau_ms:.0f} ms 실측")
    _box(ax, xs[3], y, w, h, "연관 · PnP", "카탈로그 D≥1 km")
    _box(ax, xs[4], y, w, h, "EKF 6-state", "지연 보상 + χ² 게이트")
    _box(ax, xs[5], y, w, h, "ZEM/ZEV 유도", "무제약 해석해")

    labels = ["참값 pose", "영상", "탐지 박스", "z = r_PnP", r"$\hat{x}=[\hat{r};\hat{v}]$"]
    for i in range(5):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2, labels[i])

    # 피드백: 유도 → 동역학 (추력, 아래로), 동역학 → EKF (IMU, 위로)
    ax.add_patch(FancyArrowPatch((xs[5] + w / 2, y), (xs[0] + w / 2, y),
                                 arrowstyle="-|>", mutation_scale=16, color=ACCENT,
                                 lw=1.6, connectionstyle="arc3,rad=-0.18"))
    ax.text((xs[0] + xs[5] + w) / 2, 1.2, "추력 명령 a_T (100 Hz)", ha="center",
            fontsize=9.5, color=ACCENT)
    ax.add_patch(FancyArrowPatch((xs[0] + w / 2, y + h), (xs[4] + w / 2, y + h),
                                 arrowstyle="-|>", mutation_scale=14, color=GRAY,
                                 lw=1.2, ls="--", connectionstyle="arc3,rad=-0.12"))
    ax.text((xs[0] + xs[4] + w) / 2, 27.6, "IMU 100 Hz (백색잡음)", ha="center",
            fontsize=9, color=GRAY)

    # τ 구간 브레이스
    ax.plot([xs[2], xs[4] - 1.2], [y - 1.6, y - 1.6], color=GREEN, lw=1.4)
    ax.text((xs[2] + xs[4]) / 2, y - 3.6, "τ: 촬영 → 측정 z 도착 (이 동안 촬영 불가 = 직렬 모델)",
            ha="center", fontsize=9.5, color=GREEN)

    fig.suptitle("폐루프 시뮬레이터 파이프라인 — 인식→항법→유도를 착륙 CEP로 평가", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def serial_model(out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 6.0), height_ratios=[1, 1.15])

    # ── 패널 A: 직렬 촬영 모델 (τ=2.5 s 예시, 주기 1 s)
    ax1.set_xlim(-0.3, 8.8)
    ax1.set_ylim(-1.15, 1.9)
    ax1.axis("off")
    ax1.set_title("직렬 처리 모델 — τ가 프레임 주기를 넘으면 촬영 기회를 잃는다 (예: τ=2.5 s, 1 Hz)",
                  fontsize=11.5, loc="left")
    ax1.axhline(0, color="black", lw=1)
    for t in range(9):
        ax1.plot([t, t], [-0.09, 0.09], color="black", lw=1)
        ax1.text(t, -0.42, f"{t}", ha="center", fontsize=8.5)
    ax1.text(8.28, -0.42, "t [s]", fontsize=8.5)
    for cap, color in ((0, GREEN), (3, GREEN), (6, GREEN)):
        ax1.plot(cap, 0, "o", color=color, ms=9, zorder=5)
        ax1.add_patch(Rectangle((cap, 0.32), 2.5, 0.42, fc="#d5e8d4", ec=GREEN, lw=1.2))
        ax1.text(cap + 1.25, 0.53, "처리 중 (busy)", ha="center", va="center",
                 fontsize=8.5, color=GREEN)
        ax1.annotate("", xy=(cap + 2.5, 0.14), xytext=(cap + 2.5, 0.32),
                     arrowprops=dict(arrowstyle="-|>", color=GREEN))
        ax1.text(cap + 2.5, 1.0, "z 도착", ha="center", fontsize=8, color=GREEN)
    for drop in (1, 2, 4, 5, 7, 8):
        ax1.plot(drop, 0, "x", color=ACCENT, ms=10, mew=2.2, zorder=5)
    ax1.plot([], [], "o", color=GREEN, label="촬영")
    ax1.plot([], [], "x", color=ACCENT, mew=2.2, label="드롭 (n_dropped)")
    ax1.legend(loc="upper right", fontsize=9, frameon=False)

    # ── 패널 B: 지연 보상 (링버퍼 재전파)
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
    ap.add_argument("--tau-file", default="results/tau_ort_cpu_int8.json")
    ap.add_argument("--out-dir", default="figs/slides")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tau_ms = float(json.loads(Path(args.tau_file).read_text(encoding="utf-8"))["median_s"]) * 1e3

    p1 = out_dir / "slide_05_pipeline.png"
    pipeline(tau_ms, p1)
    print(f"fig: {p1}")
    p2 = out_dir / "slide_06_serial_model.png"
    serial_model(p2)
    print(f"fig: {p2}")


if __name__ == "__main__":
    main()
