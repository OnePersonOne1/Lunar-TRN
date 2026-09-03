"""슬라이드용 시연 자산 생성 CLI: 텔레메트리 최종 그래프(PNG) + 오버레이/착륙 영상(mp4).

- 텔레메트리 그래프: frames/traj_demo.csv(참값 궤적) → 고도/속력/동서 3패널 최종 상태.
  Unity Display 4(TelemetryView)와 같은 데이터·구성을 발표 품질로 다시 그린다.
- Display 3 영상: frames/p6/*.png(P6 실런 탐지 오버레이, 1장=1초)를 시연 배속(timescale)에
  맞는 재생 속도로 인코딩.
- Unity 캡처 영상: frames/demo/의 d2_*.jpg(착륙 추적)·d3_*.jpg(오버레이 동기 재생)·
  d4_*.png(텔레메트리) — TrajectoryPlayback captureFrames가 세 화면을 동시 저장 →
  display{2,3,4}_*.mp4로 각각 인코딩 (구 demo_*.png 단일 캡처도 하위 호환 지원).
ffmpeg는 imageio-ffmpeg 내장 바이너리를 쓴다.
"""
from __future__ import annotations

import argparse
import json
import shutil
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


# 슬라이드 숫자 매니페스트: (라벨, 파일, 점표기 경로, 포맷) — 스크립트가 만든 값만.
# 경로는 dict 키를 "."로, 리스트는 [i] 또는 [tau=값]/[label=값]/[fp=값] 조건으로.
SUMMARY = [
    ("val 프레임 수", "p5_det.json", "int8.n_frames", "{:.0f}"),
    ("탐지 mAP50-95 (n INT8)", "p5_det.json", "int8.mAP50_95", "{:.3f}"),
    ("탐지 mAP50-95 (n FP32)", "p5_det.json", "fp32.mAP50_95", "{:.3f}"),
    ("측정 σ 수평 [m] (n INT8)", "measurement_model.json", "sigma_xyz_m[0]", "{:.1f}"),
    ("보정 오검출률 (n INT8)", "measurement_model.json", "fp_rate_est", "{:.3f}"),
    ("τ n INT8 CPU [ms]", "tau_ort_cpu_int8.json", "median_s", "{:.1f}", 1e3),
    ("τ n FP32 CPU [ms]", "tau_ort_cpu_fp32.json", "median_s", "{:.1f}", 1e3),
    ("τ s INT8 CPU [ms]", "s/tau_ort_cpu_int8.json", "median_s", "{:.1f}", 1e3),
    ("τ GPU INT8 [ms]", "tau_trt_int8.json", "median_s", "{:.1f}", 1e3),
    ("CPU 1스레드 CoreMark", "p7b_cpu_bench.json", "coremark.iterations_per_sec", "{:.0f}"),
    ("HR5000 대비 배율", "p7b_cpu_bench.json",
     "speedup_vs_reference.JAXA HR5000 계열 (MIPS64 5Kf, 200 MHz).dmips_x", "{:.0f}x"),
    ("P6 실런 착륙 오차 [m]", "p6_closed_loop.json", "landing_error_m", "{:.1f}"),
    ("P6 연착륙 속도 [m/s]", "p6_closed_loop.json", "landing_v_mps", "{:.2f}"),
    ("CEP 보상 τ≤1s [m]", "p7b_baseline.json", "cep_m", "{:.1f}"),
    ("CEP 미보상 n INT8 [m]", "p7b_tau_serial_compoff.json",
     "conditions[label=n INT8 CPU].cep_m", "{:.1f}"),
    ("CEP 미보상 s FP32 [m]", "p7b_tau_serial_compoff.json",
     "conditions[label=s FP32 CPU].cep_m", "{:.1f}"),
    ("오검출 0.3 시 CEP [m]", "p7b_fp_sweep.json", "conditions[fp=0.3].cep_m", "{:.1f}"),
    ("5Hz n INT8 CEP [m]", "p7b_rate_sweep.json",
     "conditions[rate5_nint8].cep_m", "{:.1f}"),
    ("5Hz n FP32 CEP [m]", "p7b_rate_sweep.json",
     "conditions[rate5_nfp32].cep_m", "{:.1f}"),
    ("지터 200ms 시 CEP [m]", "p7b_jitter_sweep.json",
     "conditions[jit=0.2].cep_m", "{:.1f}"),
    ("ΔV truth [m/s]", "p7b_deltav.json", "delta_v_mps", "{:.1f}"),
    ("고전 PCA mAP50-95", "p7c_det_compare.json", "entries.classic_pca_prior.mAP50_95", "{:.3f}"),
    ("고전 PCA recall", "p7c_det_compare.json", "entries.classic_pca_prior.recall", "{:.3f}"),
    ("τ 고전 PCA [ms]", "p7c_det_compare.json",
     "entries.classic_pca_prior.tau.median_s", "{:.1f}", 1e3),
    ("측정 σ 수평 [m] (고전)", "measurement_model_classic.json", "sigma_xyz_m[0]", "{:.1f}"),
    ("보정 오검출률 (고전)", "measurement_model_classic.json", "fp_rate_est", "{:.3f}"),
    ("CEP 고전 PCA [m]", "p7c_cep_compare.json", "conditions[label=classic_pca].cep_m", "{:.1f}"),
    ("CEP YOLO INT8 (동일 조건) [m]", "p7c_cep_compare.json",
     "conditions[label=yolo_int8].cep_m", "{:.1f}"),
    ("실런 MC CEP [m]", "p8_unity_mc.json", "cep_m", "{:.1f}"),
    ("실런 MC CEP CI 하한 [m]", "p8_unity_mc.json", "cep_ci95_m[0]", "{:.1f}"),
    ("실런 MC CEP CI 상한 [m]", "p8_unity_mc.json", "cep_ci95_m[1]", "{:.1f}"),
    ("실런 τ 중앙값 [ms]", "p8_unity_mc.json", "tau_wallclock_s.median", "{:.1f}", 1e3),
    ("실런 MC 바이어스(평균 오프셋 크기) [m]", "p8_summary.json", "unity_bias_norm_m", "{:.1f}"),
    ("실런 MC 발산 런 수 (/200)", "p8_unity_mc.json", "n_diverged", "{:.0f}"),
]


def _dig(obj, path: str):
    """점표기 경로로 중첩 dict/list에서 값을 꺼낸다. 특수 조건 리스트 셀렉터 지원."""
    import re

    for part in re.findall(r"[^.\[\]]+|\[[^\]]+\]", path):
        if part.startswith("["):
            sel = part[1:-1]
            if sel.isdigit():
                obj = obj[int(sel)]
            elif sel == "rate5_nint8":
                obj = next(c for c in obj if c["rate_hz"] == 5 and c["label"] == "n INT8 CPU")
            elif sel == "rate5_nfp32":
                obj = next(c for c in obj if c["rate_hz"] == 5 and c["label"] == "n FP32 CPU")
            elif "=" in sel:
                k, v = sel.split("=", 1)
                key = {"label": "label", "fp": "fp_rate", "tau": "tau_s",
                       "jit": "t_c_jitter_s"}[k]
                obj = next(c for c in obj if str(c[key]) == v or
                           (isinstance(c[key], float) and abs(c[key] - float(v)) < 1e-6))
            else:
                raise KeyError(sel)
        elif re.fullmatch(r"[a-zA-Z0-9_ ().μ/-]+\[\d+\]", part):
            pass  # 안 옴
        else:
            obj = obj[part]
    return obj


def make_results_summary(results_dir: Path, out: Path) -> None:
    """slides 숫자 매니페스트 → docs/results_summary.md (숫자 | 출처 | 시각 | git hash)."""
    rows = []
    for item in SUMMARY:
        label, fname, path, fmt = item[0], item[1], item[2], item[3]
        scale = item[4] if len(item) > 4 else 1.0
        fpath = results_dir / fname
        if not fpath.exists():
            rows.append((label, "TBD", fname, "—", "—"))
            continue
        d = json.loads(fpath.read_text(encoding="utf-8"))
        try:
            # path에 [i] 인덱스가 섞인 sigma_xyz_m[0] 같은 경우 처리
            if "[" in path and "]" in path and path.split("[")[0].replace("_", "").isalnum() \
                    and path.endswith("]") and path[path.rfind("[") + 1:-1].isdigit():
                base, idx = path[:path.rfind("[")], int(path[path.rfind("[") + 1:-1])
                val = _dig(d, base)[idx]
            else:
                val = _dig(d, path)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                val = fmt.format(val * scale)
            else:
                val = fmt.format(val)
        except (KeyError, StopIteration, IndexError, TypeError) as e:
            val = f"TBD ({type(e).__name__})"
        meta = d.get("meta", {})
        gh = (meta.get("git_hash") or "—")[:8]
        ts = (meta.get("timestamp") or "—")[:19]
        rows.append((label, val, fname, ts, gh))

    lines = [
        "# 슬라이드 숫자 요약 (자동 생성 — scripts/make_slide_assets.py)",
        "",
        "모든 값은 results/*.json에서 스크립트가 추출한다. 손으로 고치지 말 것.",
        "TBD = 해당 결과 파일/필드 부재. 구 assumed 수치는 매니페스트에서 제외됨.",
        "",
        "| 항목 | 값 | 출처 파일 | 생성 시각(UTC) | git |",
        "|---|---|---|---|---|",
    ]
    for label, val, fname, ts, gh in rows:
        lines.append(f"| {label} | {val} | `{fname}` | {ts} | `{gh}` |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary: {out} ({len(rows)} rows)")


# 슬라이드용 그림 복사: (원본, 슬라이드 파일명)
SLIDE_FIGS = [
    # slide_05/06은 scripts/make_diagrams.py가 figs/slides/에 직접 생성한다(여기서 복사 안 함).
    ("figs/p7b_tau_scaling.png", "slide_10_cep_vs_tau.png"),
    ("figs/p7b_cep_vs_fp.png", "slide_11_cep_vs_fp.png"),
    ("figs/p7b_cep_vs_jitter.png", "slide_12a_jitter.png"),
    ("figs/p7b_cep_vs_rate.png", "slide_12b_rate.png"),
    ("figs/p7b_pnp_error_hist_s.png", "slide_07_pnp_hist_s.png"),
    ("figs/p7c_det_compare.png", "slide_13a_classic_det.png"),
    ("figs/p7c_cep_compare.png", "slide_13b_classic_cep.png"),
]


def copy_slide_figs(root: Path, out_dir: Path) -> None:
    for src, dst in SLIDE_FIGS:
        sp = root / src
        if not sp.exists():
            print(f"skip fig: {src} 없음")
            continue
        try:
            shutil.copy2(sp, out_dir / dst)
            print(f"fig copy: {dst}")
        except PermissionError:
            print(f"skip fig: {dst} 잠김(뷰어에서 열림?) — 나중에 재실행")


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
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--summary-out", default="docs/results_summary.md")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    out_dir = Path(args.out)
    root = Path(__file__).resolve().parents[1]

    # 0) 슬라이드 숫자 요약 + 슬라이드 그림 복사
    make_results_summary(Path(args.results_dir), root / args.summary_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_slide_figs(root, out_dir)

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

    # 3) Unity 동시 캡처 영상 (captureFrames 산출이 있을 때): d2/d3/d4 → mp4 3종
    cap = Path(args.capture_dir)
    seqs = [
        ("d2_*.jpg", "d2_%05d.jpg", "display2_landing.mp4"),
        ("d3_*.jpg", "d3_%05d.jpg", "display3_detection_synced.mp4"),
        ("d4_*.png", "d4_%05d.png", "display4_telemetry.mp4"),
        ("demo_*.png", "demo_%05d.png", "display2_landing.mp4"),  # 구 단일 캡처 호환
    ]
    done = set()
    for glob_pat, ff_pat, out_name in seqs:
        if out_name in done or not list(cap.glob(glob_pat)):
            continue
        encode_video(str(cap / ff_pat), float(args.fps), args.fps, out_dir / out_name)
        done.add(out_name)
    if not done:
        print(f"skip: {cap}에 캡처 프레임 없음 (unity/README.md 시연 재생 참고)")


if __name__ == "__main__":
    main()
