"""카탈로그 투영 기반 YOLO 자동 라벨 생성 (계약 §2.3).

bbox = 투영된 원(직경 D)의 외접 사각형(nadir 가정). 포함 조건: 중심이 화면 안,
투영 직경 ≥ p_min px, bbox의 50% 이상이 화면 안(경계로 클리핑). 라벨 `0 cx cy w h`(정규화).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.camera import K_cam, project  # noqa: E402

MIN_VISIBLE_FRACTION = 0.5  # 계약 §2.3: bbox의 50% 이상이 화면 안


def label_frame(catalog: np.ndarray, r: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """한 프레임의 YOLO 라벨을 만든다.

    catalog: (N,4) [x, y, z, D] (L, m). r: 카메라 위치.
    반환: (labels (M,4) 정규화 [cx, cy, w, h], ids (M,) — catalog 행 인덱스).
    """
    W = float(cfg["camera"]["W"])
    H = float(cfg["camera"]["H"])
    p_min = float(cfg["catalog"]["p_min_px"])
    f = K_cam(cfg)[0, 0]

    uv, z_C, valid = project(catalog[:, :3], r, cfg)
    labels, ids = [], []
    for i in np.flatnonzero(valid):
        u, v = uv[i]
        if not (0.0 <= u < W and 0.0 <= v < H):  # 중심이 화면 안
            continue
        d_px = f * catalog[i, 3] / z_C[i]  # 투영 직경 (nadir 가정)
        if d_px < p_min:
            continue
        x0, y0 = u - d_px / 2.0, v - d_px / 2.0
        x1, y1 = u + d_px / 2.0, v + d_px / 2.0
        cx0, cy0 = max(x0, 0.0), max(y0, 0.0)
        cx1, cy1 = min(x1, W), min(y1, H)
        if (cx1 - cx0) * (cy1 - cy0) < MIN_VISIBLE_FRACTION * d_px * d_px:
            continue
        labels.append([
            (cx0 + cx1) / 2.0 / W, (cy0 + cy1) / 2.0 / H,
            (cx1 - cx0) / W, (cy1 - cy0) / H,
        ])
        ids.append(i)
    if not labels:
        return np.empty((0, 4)), np.empty(0, dtype=int)
    return np.asarray(labels), np.asarray(ids, dtype=int)


def write_yolo_txt(path: Path, labels: np.ndarray) -> None:
    lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in labels]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def draw_overlay(
    img: np.ndarray | None, catalog: np.ndarray, ids: np.ndarray, r: np.ndarray, cfg: dict
) -> np.ndarray:
    """라벨된 크레이터 원과 id를 그린 오버레이 이미지 (img가 None이면 검정 배경)."""
    import cv2

    W, H = int(cfg["camera"]["W"]), int(cfg["camera"]["H"])
    canvas = img.copy() if img is not None else np.zeros((H, W, 3), dtype=np.uint8)
    f = K_cam(cfg)[0, 0]
    uv, z_C, _ = project(catalog[:, :3], r, cfg)
    for i in ids:
        u, v = uv[i]
        rad = int(round(f * catalog[i, 3] / z_C[i] / 2.0))
        cv2.circle(canvas, (int(round(u)), int(round(v))), rad, (0, 255, 0), 2)
        cv2.putText(canvas, str(int(i)), (int(u) + 4, int(v) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # CLI 규약 통일용, 여기서는 미사용
    ap.add_argument("--catalog", default="data/processed/catalog_L.csv")
    ap.add_argument("--poses", required=True, help="poses.csv (t, x, y, z, sun_az_deg, sun_el_deg)")
    ap.add_argument("--out", required=True, help="라벨 txt 출력 디렉터리")
    ap.add_argument("--traj-id", default="traj0")
    ap.add_argument("--frames-dir", default=None, help="렌더 프레임 디렉터리(있으면 overlay에 사용)")
    ap.add_argument("--overlay-every", type=int, default=0, help="k>0이면 k프레임마다 overlay PNG")
    args = ap.parse_args()

    import cv2

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cat = np.genfromtxt(args.catalog, delimiter=",", names=True)
    catalog = np.column_stack([cat["x"], cat["y"], cat["z"], cat["D"]])
    poses = np.genfromtxt(args.poses, delimiter=",", names=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(len(poses)):
        r = np.array([poses["x"][k], poses["y"][k], poses["z"][k]])
        labels, ids = label_frame(catalog, r, cfg)
        stem = f"{args.traj_id}_{k:05d}"
        write_yolo_txt(out_dir / f"{stem}.txt", labels)
        if args.overlay_every > 0 and k % args.overlay_every == 0:
            img = None
            if args.frames_dir:
                p = Path(args.frames_dir) / f"{stem}.png"
                img = cv2.imread(str(p)) if p.exists() else None
            cv2.imwrite(str(out_dir / f"{stem}_overlay.png"),
                        draw_overlay(img, catalog, ids, r, cfg))
    print(f"labels: {len(poses)} frames → {out_dir}")


if __name__ == "__main__":
    main()
