"""고전 베이스라인 학습 CLI(P7c): train 프레임 → PCA 템플릿 탐지기 runs/classic/pca_crater.npz.

YOLO와 같은 train 분할·라벨을 쓴다. 결과 results/p7c_classic_train.json(기저 설명분산,
임계 튜닝 F1 등). 이어서 scripts/eval_classic.py로 val mAP·τ를 잰다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.classic import train_pca_detector  # noqa: E402
from sim.mc import result_meta  # noqa: E402


def load_labels_px(txt: Path, W: int, H: int) -> np.ndarray:
    """YOLO txt → (m, 4) px 박스."""
    if not txt.exists():
        return np.empty((0, 4))
    rows = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            _, cx, cy, w, h = (float(v) for v in parts)
            rows.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return np.asarray(rows) if rows else np.empty((0, 4))


def iter_split(dataset: Path, split: str, cfg: dict):
    """(stem, img, gts_px)를 한 장씩 읽어 넘긴다 — 학습셋 전체를 메모리에 올리지 않는다."""
    import cv2

    W, H = int(cfg["camera"]["W"]), int(cfg["camera"]["H"])
    for p in sorted((dataset / "images" / split).glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        yield p.stem, img, load_labels_px(dataset / "labels" / split / f"{p.stem}.txt", W, H)


def load_split(dataset: Path, split: str, cfg: dict) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    imgs, gts, stems = [], [], []
    for stem, img, g in iter_split(dataset, split, cfg):
        imgs.append(img)
        gts.append(g)
        stems.append(stem)
    return imgs, gts, stems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--model-out", default=None, help="기본 config classic.model")
    ap.add_argument("--out", default="results/p7c_classic_train.json")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    c = cfg["classic"]
    rng = np.random.default_rng(args.seed)
    model_out = Path(args.model_out or c["model"])

    def frames():
        for i, (_, img, g) in enumerate(iter_split(Path(args.dataset), "train", cfg)):
            if i % 100 == 0:
                print(f"  frame {i}", flush=True)
            yield img, g

    t0 = time.perf_counter()
    det, info = train_pca_detector(frames(), c, rng)
    info["train_wall_s"] = time.perf_counter() - t0
    det.save(model_out)

    # 폐루프용 사본: EKF 예측 고도와 카탈로그 D_min이 픽셀 직경 하한을 정하므로
    # (d = f·D/h, 최악은 h = h_max) 그보다 미세한 스케일은 탐색하지 않는다.
    from perception.camera import K_cam

    f_px = float(K_cam(cfg)[0, 0])
    d_min_prior = f_px * float(cfg["catalog"]["D_min_m"]) / float(cfg["trn_band"]["h_max_m"])
    prior_out = model_out.with_name(model_out.stem + "_prior" + model_out.suffix)
    det.with_diameter_range(d_min_prior, c["d_max_px"]).save(prior_out)
    info["prior_model"] = str(prior_out)
    info["prior_d_min_px"] = d_min_prior
    print(json.dumps(info, ensure_ascii=False, indent=1), flush=True)

    out = {"meta": result_meta(args.config), "model": str(model_out),
           "params": {k: v for k, v in c.items() if k != "model"}, **info}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {model_out}, {out_path}")


if __name__ == "__main__":
    main()
