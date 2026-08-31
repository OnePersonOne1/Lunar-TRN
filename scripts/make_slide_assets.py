"""슬라이드용 시연 자산 생성 CLI: 텔레메트리 최종 그래프(PNG) + 오버레이/착륙 영상(mp4).

- 텔레메트리 그래프: frames/traj_demo.csv(참값 궤적) → 고도/속력/동서 3패널 최종 상태.
  Unity Display 4(TelemetryView)와 같은 데이터·구성을 발표 품질로 다시 그린다.
- Display 3 영상: frames/p6/*.png(P6 실런 탐지 오버레이, 1장=1초)를 시연 배속(timescale)에
  맞는 재생 속도로 인코딩.
- Display 2 영상: frames/demo/demo_*.png(Unity captureFrames 산출)가 있으면 인코딩.
ffmpeg는 imageio-ffmpeg 내장 바이너리를 쓴다.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Unity TelemetryView와 같은 계열의 패널 색 (단일 시리즈 패널 — 텍스트 라벨로 식별)
COL_ALT = "#1452C7"
COL_SPD = "#D16608"
COL_EAST = "#087A29"
COL_BAND = "#B9C2CE"


def load_traj(path: Path) -> dict[str, np.ndarray]:
    d = np.genfromtxt(path, delimiter=",", names=True)
    spd = np.sqrt(d["vx"] ** 2 + d["vy"] ** 2 + d["vz"] ** 2)
    return {"t": d["t"], "alt_km": d["z"] / 1000.0, "spd": spd, "east_km": d["x"] / 1000.0}


def make_telemetry_fig(traj: dict, cfg: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = traj["t"]
    band = cfg["trn_band"]
    h_lo, h_hi = float(band["h_min_m"]) / 1000.0, float(band["h_max_m"]) / 1000.0
    in_band = (traj["alt_km"] >= h_lo) & (traj["alt_km"] <= h_hi)
    t_band = (float(t[in_band][0]), float(t[in_band][-1])) if in_band.any() else None

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    panels = [
        (traj["alt_km"], "Altitude [km]", COL_ALT),
        (traj["spd"], "Speed [m/s]", COL_SPD),
        (traj["east_km"], "East [km]", COL_EAST),
    ]
    for ax, (y, label, col) in zip(axes, panels):
        ax.plot(t, y, color=col, linewidth=2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if t_band is not None:
            ax.axvspan(*t_band, color=COL_BAND, alpha=0.25, zorder=0)
    if t_band is not None:
        axes[0].text(
            (t_band[0] + t_band[1]) / 2.0, axes[0].get_ylim()[1] * 0.93,
            "TRN band", ha="center", va="top", fontsize=10, color="#3B4148",
        )
    axes[0].annotate(
        f"touchdown t={t[-1]:.0f} s", xy=(t[-1], traj["alt_km"][-1]),
        xytext=(-8, 14), textcoords="offset points", ha="right", fontsize=10,
        color="#3B4148",
    )
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Closed-loop descent telemetry (truth trajectory)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"fig: {out}")


def encode_video(
    frames_glob: str, in_fps: float, out_fps: int, out: Path, start_number: int = 0
) -> None:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-framerate", f"{in_fps:g}", "-start_number", str(start_number),
        "-i", frames_glob, "-r", str(out_fps), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"ffmpeg 실패: {out}")
    print(f"video: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # 결정론 산출물 — 시드 미사용
    ap.add_argument("--out", default="figs/slides", help="출력 디렉터리")
    ap.add_argument("--traj", default="frames/traj_demo.csv")
    ap.add_argument("--overlay-dir", default="frames/p6")
    ap.add_argument("--capture-dir", default="frames/demo")
    ap.add_argument("--timescale", type=float, default=8.0,
                    help="시연 배속 — 오버레이(1장=1초)의 재생 프레임레이트")
    ap.add_argument("--fps", type=int, default=30, help="출력 영상 프레임레이트")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    out_dir = Path(args.out)

    # 1) Display 4: 텔레메트리 최종 그래프
    make_telemetry_fig(load_traj(Path(args.traj)), cfg, out_dir / "display4_telemetry.png")

    # 2) Display 3: 탐지 오버레이 영상 (1장 = 시뮬 1초 → timescale 배속 재생)
    if list(Path(args.overlay_dir).glob("[0-9]*.png")):
        encode_video(
            str(Path(args.overlay_dir) / "%05d.png"), args.timescale, args.fps,
            out_dir / "display3_detection.mp4",
        )
    else:
        print(f"skip: {args.overlay_dir}에 오버레이 프레임 없음")

    # 3) Display 2: 착륙 추적 캡처 영상 (Unity captureFrames 산출이 있을 때)
    if list(Path(args.capture_dir).glob("demo_*.png")):
        encode_video(
            str(Path(args.capture_dir) / "demo_%05d.png"), float(args.fps), args.fps,
            out_dir / "display2_landing.mp4",
        )
    else:
        print(f"skip: {args.capture_dir}에 캡처 프레임 없음 (unity/README.md 시연 재생 참고)")


if __name__ == "__main__":
    main()
