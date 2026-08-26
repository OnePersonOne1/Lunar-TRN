"""INT8 PTQ 내보내기 CLI: best.pt → ONNX → (TRT explicit INT8 엔진, ORT CPU INT8 ONNX).

캘리브레이션은 train 이미지 bench.calib_images장. 같은 이미지를 data/calib/에도 복사한다.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.bench_tau import build_trt_engine, quantize_onnx_int8  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="runs/train/crater/weights/best.pt")
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--out", default="runs/export")
    args = ap.parse_args()

    import cv2

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    imgsz = int(cfg["detector"]["imgsz"])
    n_calib = int(cfg["bench"]["calib_images"])

    train_imgs = sorted((Path(args.dataset) / "images" / "train").glob("*.png"))[:n_calib]
    if len(train_imgs) < n_calib:
        print(f"경고: train 이미지 {len(train_imgs)}장 < calib_images {n_calib}")
    calib_dir = Path("data/calib")
    calib_dir.mkdir(parents=True, exist_ok=True)
    for p in train_imgs:
        shutil.copy2(p, calib_dir / p.name)
    calib_imgs = [cv2.imread(str(p)) for p in train_imgs]
    calib_imgs = [im for im in calib_imgs if im is not None]
    print(f"calib: {len(calib_imgs)}장 → data/calib/", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(args.model)
    onnx_path = model.export(format="onnx", imgsz=imgsz, batch=1, dynamic=False, device="cpu")
    fp32_onnx = out_dir / "crater_fp32.onnx"
    shutil.copy2(onnx_path, fp32_onnx)
    print(f"onnx fp32: {fp32_onnx}", flush=True)

    ort_int8 = out_dir / "crater_int8_ort.onnx"
    quantize_onnx_int8(str(fp32_onnx), str(ort_int8), calib_imgs, imgsz, "u8s8")
    print(f"ort int8: {ort_int8}", flush=True)

    trt_int8_onnx = out_dir / "crater_int8_trt.onnx"
    quantize_onnx_int8(str(fp32_onnx), str(trt_int8_onnx), calib_imgs, imgsz, "s8s8_sym",
                       ("/model.23/", "/model.10/"))
    for prec, src in (("fp32", fp32_onnx), ("int8", trt_int8_onnx)):
        eng = out_dir / f"crater_{prec}.engine"
        method = build_trt_engine(str(src), str(eng))
        print(f"trt {prec}: {eng} ({method})", flush=True)


if __name__ == "__main__":
    main()
