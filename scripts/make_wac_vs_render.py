"""실제 WAC 모자이크 vs Unity 렌더 비교 그림 CLI.

데이터셋 프레임(포즈 기록)과 같은 지면 발자국을 원본 텍스처(data/processed/texture_L.png,
LROC WAC 모자이크 크롭 = 실제 위성 영상 산출물)에서 잘라 나란히 놓는다.
도메인 갭(100 m/px 업샘플·재조명) 설명용 슬라이드 자산. → figs/wac_vs_render.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pick_frame(poses_csv: Path, h_target_m: float, split: str) -> dict:
    """목표 고도에 가장 가까운 프레임 한 장."""
    with poses_csv.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == split]
    return min(rows, key=lambda r: abs(float(r["z"]) - h_target_m))


def crop_texture(tex: np.ndarray, meta: dict, x_m: float, y_m: float,
                 half_m: float) -> np.ndarray:
    """텍스처(row0=north, L 원점=중심)에서 (x, y) 중심 half_m 반폭 정사각 크롭."""
    h_px, w_px = tex.shape[:2]
    east_m = meta["size_m"]["east"]
    north_m = meta["size_m"]["north"]
    ppm_x, ppm_y = w_px / east_m, h_px / north_m
    cx = w_px / 2 + x_m * ppm_x
    cy = h_px / 2 - y_m * ppm_y          # row0 = north
    x0, x1 = int(cx - half_m * ppm_x), int(cx + half_m * ppm_x)
    y0, y1 = int(cy - half_m * ppm_y), int(cy + half_m * ppm_y)
    x0, y0 = max(x0, 0), max(y0, 0)
    return tex[y0:y1, x0:x1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # CLI 규약 통일용
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--texture", default="data/processed/texture_L.png")
    ap.add_argument("--meta", default="data/processed/heightmap_meta.json")
    ap.add_argument("--h-target", type=float, default=25000.0, help="비교 프레임 고도 [m]")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="figs/wac_vs_render.png")
    args = ap.parse_args()

    import cv2
    import yaml

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    fov_v = np.deg2rad(float(cfg["camera"]["fov_v_deg"]))
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))

    ds = Path(args.dataset)
    fr = pick_frame(ds / "poses.csv", args.h_target, args.split)
    x, y, h = float(fr["x"]), float(fr["y"]), float(fr["z"])
    tid = fr["traj_id"] if str(fr["traj_id"]).startswith("traj") else f"traj{fr['traj_id']}"
    stem = f"{tid}_{int(fr['frame_id']):05d}"
    img_path = ds / "images" / args.split / f"{stem}.png"
    render = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

    tex = cv2.cvtColor(cv2.imread(args.texture), cv2.COLOR_BGR2RGB)
    half_m = h * np.tan(fov_v / 2)       # nadir 정사각 발자국 반폭
    crop = crop_texture(tex, meta, x, y, half_m)
    crop = cv2.resize(crop, (render.shape[1], render.shape[0]),
                      interpolation=cv2.INTER_NEAREST)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    axes[0].imshow(crop)
    axes[0].set_title("LROC WAC 모자이크 (실제 위성 영상, 100 m/px)", fontsize=11)
    axes[1].imshow(render)
    axes[1].set_title("Unity 렌더 (데이터셋 프레임, DEM 재조명)", fontsize=11)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"같은 지면 발자국 {2 * half_m / 1e3:.1f} km — 고도 {h / 1e3:.1f} km, "
        f"프레임 {stem}, 태양 고도 {float(fr['sun_el_deg']):.0f}° "
        f"(모자이크는 촬영 당시 조명 고정, 렌더는 DEM 기반 재조명)", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"frame {stem}  h={h:.0f} m  footprint={2 * half_m / 1e3:.1f} km")
    print(f"fig: {out}")


if __name__ == "__main__":
    main()
