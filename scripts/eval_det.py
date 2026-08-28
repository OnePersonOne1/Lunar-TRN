"""탐지기 평가 CLI: val mAP·정밀도·재현율·FP율(FP32 vs INT8) → results/p5_det.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mc import result_meta  # noqa: E402

IOU_TP = 0.5


def _load_labels(txt: Path, W: int, H: int) -> np.ndarray:
    if not txt.exists():
        return np.empty((0, 4))
    rows = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            _, cx, cy, w, h = (float(v) for v in parts)
            rows.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return np.asarray(rows) if rows else np.empty((0, 4))


def _fp_per_frame(model_path: str, cfg: dict, dataset: Path) -> dict:
    """config conf에서 프레임당 오검출 수·정밀도·재현율 (val)."""
    import cv2

    from perception.bench_tau import _iou_matrix
    from perception.detect import Detector

    det = Detector(model_path, cfg)
    W, H = int(cfg["camera"]["W"]), int(cfg["camera"]["H"])
    imgs = sorted((dataset / "images" / "val").glob("*.png"))
    tp = fp = fn = 0
    for p in imgs:
        boxes = det.detect(cv2.imread(str(p)))[:, :4]
        gt = _load_labels(dataset / "labels" / "val" / f"{p.stem}.txt", W, H)
        if len(boxes) == 0:
            fn += len(gt)
            continue
        if len(gt) == 0:
            fp += len(boxes)
            continue
        m = _iou_matrix(boxes, gt)
        matched_gt: set[int] = set()
        for i in range(len(boxes)):
            j = int(m[i].argmax())
            if m[i, j] >= IOU_TP and j not in matched_gt:
                tp += 1
                matched_gt.add(j)
            else:
                fp += 1
        fn += len(gt) - len(matched_gt)
    n = max(len(imgs), 1)
    return {
        "n_frames": len(imgs), "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "fp_per_frame": fp / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp32", default="runs/train/crater/weights/best.pt")
    ap.add_argument("--int8", default="runs/export/crater_int8_ort.onnx")
    ap.add_argument("--data", default="data/dataset/dataset.yaml")
    ap.add_argument("--out", default="results/p5_det.json")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    dataset = Path(args.data).parent

    from ultralytics import YOLO

    out: dict = {"meta": result_meta(args.config), "conf": cfg["detector"]["conf"]}
    for key, path in (("fp32", args.fp32), ("int8", args.int8)):
        # .onnx는 CPU 전용 onnxruntime — CUDA 텐서 바인딩이 안 되므로 device를 CPU로 고정
        device = "cpu" if str(path).endswith(".onnx") else None
        metrics = YOLO(path).val(
            data=args.data, imgsz=int(cfg["detector"]["imgsz"]), split="val", verbose=False,
            **({"device": device} if device else {}),
        )
        entry = {
            "model": path,
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        }
        entry.update(_fp_per_frame(path, cfg, dataset))
        out[key] = entry
        print(f"{key}: mAP50={entry['mAP50']:.3f} mAP50-95={entry['mAP50_95']:.3f} "
              f"P={entry['precision']:.3f} R={entry['recall']:.3f} FP/frame={entry['fp_per_frame']:.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")
    print("이어서 학습 모델 τ 벤치 재실행:")
    print(f"  .venv\\Scripts\\python perception\\bench_tau.py --config {args.config} "
          f"--model {args.fp32} --out results --fig figs/p5_tau_hist.png")


if __name__ == "__main__":
    main()
