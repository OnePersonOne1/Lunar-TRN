"""YOLO 계열 경량 크레이터 탐지 추론 래퍼 (torch .pt / ONNX Runtime / TensorRT 엔진).

반환 박스는 letterbox 좌표가 아니라 원본 이미지 픽셀 좌표다(정사각 W=H 렌더 전제:
입력이 이미 imgsz 정사각이면 letterbox는 항등).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.bench_tau import OrtRunner, TrtRunner  # noqa: E402


class Detector:
    """backend: "torch"(.pt) | "ort"(.onnx) | "trt"(.engine)."""

    def __init__(self, model_path: str, cfg: dict, backend: str | None = None) -> None:
        suffix = Path(model_path).suffix
        if backend is None:
            backend = {".pt": "torch", ".onnx": "ort", ".engine": "trt"}.get(suffix, "torch")
        self.backend = backend
        imgsz = int(cfg["detector"]["imgsz"])
        conf = float(cfg["detector"]["conf"])
        if backend == "torch":
            from ultralytics import YOLO

            self._yolo = YOLO(model_path)
            self._imgsz = imgsz
            self._conf = conf
        elif backend == "ort":
            self._runner = OrtRunner(model_path, int(cfg["bench"]["cpu_threads"]), imgsz, conf)
        elif backend == "trt":
            self._runner = TrtRunner(model_path, imgsz, conf)
        else:
            raise ValueError(f"알 수 없는 backend: {backend}")

    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        """(n, 6) [x0, y0, x1, y1, conf, cls] float32."""
        if self.backend == "torch":
            res = self._yolo.predict(img_bgr, imgsz=self._imgsz, conf=self._conf, verbose=False)[0]
            return res.boxes.data.cpu().numpy().astype(np.float32)
        det = self._runner.detect(img_bgr)
        return det.cpu().numpy().astype(np.float32)

    def centers(self, img_bgr: np.ndarray) -> np.ndarray:
        """탐지 박스 중심 (n, 2) px."""
        d = self.detect(img_bgr)
        if len(d) == 0:
            return np.empty((0, 2))
        return np.column_stack([(d[:, 0] + d[:, 2]) / 2.0, (d[:, 1] + d[:, 3]) / 2.0])
