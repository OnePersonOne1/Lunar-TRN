"""데이터셋 생성 CLI: 공칭 궤적(TRN 구간) pose + 태양각 무작위 → Unity 렌더 + 자동 라벨.

train/val은 궤적(반복) 단위로 분리한다(프레임 무작위 분리 금지). Unity 서버가 떠 있어야 한다.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.labeler import draw_overlay, label_frame, write_yolo_txt  # noqa: E402
from sim.loop import run_closed_loop  # noqa: E402
from unity.client import RenderClient  # noqa: E402

N_OVERLAY_FIGS = 6


def band_poses(cfg: dict, seed: int) -> np.ndarray:
    """공칭 궤적에서 TRN 구간의 카메라 주기 pose (K,4: t, x, y, z)."""
    res = run_closed_loop(cfg, seed, measurement="truth")
    traj, t = res["traj_true"], res["traj_t"]
    spf = int(round(cfg["imu"]["rate_hz"] / cfg["camera"]["rate_hz"]))
    idx = np.arange(0, len(traj), spf)
    h = traj[idx, 2]
    band = (h >= float(cfg["trn_band"]["h_min_m"])) & (h <= float(cfg["trn_band"]["h_max_m"]))
    sel = idx[band]
    return np.column_stack([t[sel], traj[sel, 0], traj[sel, 1], traj[sel, 2]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--catalog", default="data/processed/catalog_L.csv")
    ap.add_argument("--figs-dir", default="figs")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    rng = np.random.default_rng(args.seed)
    ds = cfg["dataset"]
    n_frames = int(ds["n_frames"])
    val_fraction = float(ds["val_fraction"])
    el_lo, el_hi = (float(v) for v in ds["sun_el_deg"])
    az_lo, az_hi = (float(v) for v in ds["sun_az_deg"])

    cat = np.genfromtxt(args.catalog, delimiter=",", names=True)
    catalog = np.column_stack([cat["x"], cat["y"], cat["z"], cat["D"]])

    poses = band_poses(cfg, args.seed)
    if len(poses) == 0:
        raise SystemExit("TRN 구간에 카메라 프레임이 없다. trn_band/scenario를 확인해라.")
    n_traj = math.ceil(n_frames / len(poses))
    n_val_traj = max(1, round(n_traj * val_fraction)) if n_traj > 1 else 0

    out_dir = Path(args.out)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    client = RenderClient(cfg)
    rows = []
    made = 0
    overlay_every = max(1, n_frames // N_OVERLAY_FIGS)
    for traj_id in range(n_traj):
        split = "val" if traj_id >= n_traj - n_val_traj else "train"
        for k in range(len(poses)):
            if made >= n_frames:
                break
            t, x, y, z = poses[k]
            r = np.array([x, y, z])
            sun_az = float(rng.uniform(az_lo, az_hi))
            sun_el = float(rng.uniform(el_lo, el_hi))
            img = client.render(r, sun_az, sun_el, frame_id=made, t=t)
            stem = f"traj{traj_id}_{k:05d}"
            cv2.imwrite(str(out_dir / "images" / split / f"{stem}.png"), img)
            labels, ids = label_frame(catalog, r, cfg)
            write_yolo_txt(out_dir / "labels" / split / f"{stem}.txt", labels)
            rows.append([traj_id, k, t, x, y, z, sun_az, sun_el, split])
            if made % overlay_every == 0 and made // overlay_every < N_OVERLAY_FIGS:
                cv2.imwrite(str(figs_dir / f"p5_overlay_{made // overlay_every}.png"),
                            draw_overlay(img, catalog, ids, r, cfg))
            made += 1
            if made % 50 == 0:
                print(f"{made}/{n_frames}", flush=True)
    client.close()

    with open(out_dir / "poses.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["traj_id", "frame_id", "t", "x", "y", "z", "sun_az_deg", "sun_el_deg", "split"])
        w.writerows(rows)

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\nnames:\n  0: crater\n",
        encoding="utf-8",
    )
    print(f"dataset: {made} frames ({n_traj} traj, val {n_val_traj} traj) → {out_dir}")
    print(f"dataset.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
