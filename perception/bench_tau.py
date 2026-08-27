"""추론 지연 τ 벤치마크: TensorRT(FP32/INT8)·ONNX Runtime CPU(FP32/INT8) 지연 분포를 results/tau_*.json에 기록.

τ_det = 전처리 + 추론 + NMS. 출력 스키마는 sim.measurement.TauSampler(empirical)가 읽는다:
{"samples_s": [...], "median_s", "p95_s", "mean_s", "std_s", "meta": {...}}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mc import result_meta  # noqa: E402

IOU_MATCH_MIN = 0.1  # 매칭 후보로 인정할 최소 IoU (sanity 통계용)


# ---------------------------------------------------------------- 이미지 준비

def temporary_images(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """data/calib/가 없을 때 쓰는 임시 이미지: ultralytics 내장 샘플(bus, zidane)을 크롭·플립 증강."""
    from ultralytics.utils import ASSETS

    bases = [cv2.imread(str(p)) for p in sorted(ASSETS.glob("*.jpg"))]
    bases = [b for b in bases if b is not None]
    if not bases:
        raise RuntimeError("ultralytics 내장 샘플 이미지를 찾지 못했다")
    out = []
    for i in range(n):
        img = bases[i % len(bases)]
        h, w = img.shape[:2]
        scale = float(rng.uniform(0.6, 1.0))
        ch, cw = int(h * scale), int(w * scale)
        y0 = int(rng.integers(0, h - ch + 1))
        x0 = int(rng.integers(0, w - cw + 1))
        crop = img[y0 : y0 + ch, x0 : x0 + cw]
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        out.append(np.ascontiguousarray(crop))
    return out


def load_calib_images(cfg: dict, rng: np.random.Generator) -> tuple[list[np.ndarray], str]:
    """data/calib/ 이미지 로드, 없으면 임시 이미지 생성. (images, calib_source) 반환."""
    n = int(cfg["bench"]["calib_images"])
    calib_dir = Path("data/calib")
    if calib_dir.is_dir():
        paths = sorted(list(calib_dir.glob("*.png")) + list(calib_dir.glob("*.jpg")))[:n]
        imgs = [cv2.imread(str(p)) for p in paths]
        imgs = [i for i in imgs if i is not None]
        if imgs:
            return imgs, "data/calib"
    return temporary_images(n, rng), "temporary"


def preprocess(img: np.ndarray, imgsz: int) -> np.ndarray:
    """letterbox → (1,3,imgsz,imgsz) float32 [0,1] CHW."""
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(x[None])


def _nms(pred_np: np.ndarray, conf: float, device: str):
    import torch
    try:
        from ultralytics.utils.nms import non_max_suppression  # ultralytics >= 8.4
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    pred = torch.from_numpy(pred_np) if isinstance(pred_np, np.ndarray) else pred_np
    return non_max_suppression(pred.to(device), conf_thres=conf)[0]


# ---------------------------------------------------------------- 러너

class OrtRunner:
    """ONNX Runtime CPU 러너. intra_op_num_threads = config bench.cpu_threads."""

    def __init__(self, onnx_path: str, threads: int, imgsz: int, conf: float) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf = conf

    def detect(self, img: np.ndarray):
        x = preprocess(img, self.imgsz)
        y = self.sess.run(None, {self.input_name: x})[0]
        return _nms(y, self.conf, "cpu")


class TrtRunner:
    """TensorRT 엔진 러너 (torch CUDA 텐서를 IO 버퍼로 사용)."""

    def __init__(self, engine_path: str, imgsz: int, conf: float) -> None:
        import tensorrt as trt
        import torch

        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.imgsz = imgsz
        self.conf = conf
        self.io: dict[str, torch.Tensor] = {}
        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.ctx.get_tensor_shape(name))
            dtype = {"DataType.FLOAT": torch.float32, "DataType.HALF": torch.float16,
                     "DataType.INT32": torch.int32}.get(str(self.engine.get_tensor_dtype(name)), torch.float32)
            buf = torch.empty(shape, dtype=dtype, device="cuda")
            self.io[name] = buf
            self.ctx.set_tensor_address(name, buf.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        self.stream = torch.cuda.current_stream().cuda_stream

    def detect(self, img: np.ndarray):
        x = preprocess(img, self.imgsz)
        self.io[self.input_name].copy_(self.torch.from_numpy(x))
        self.ctx.execute_async_v3(self.stream)
        det = _nms(self.io[self.output_name].float(), self.conf, "cuda")
        self.torch.cuda.synchronize()
        return det.cpu()


# ---------------------------------------------------------------- TensorRT 빌드

def build_trt_engine(onnx_path: str, engine_path: str, fp16: bool = False) -> str:
    """ONNX → TRT 엔진 직렬화. INT8은 QDQ 양자화 ONNX를 넘기면 explicit quantization으로 빌드된다.

    (TensorRT 11에서 implicit INT8 캘리브레이션 API(BuilderFlag.INT8)가 제거되어
    ORT static 양자화와 같은 QDQ ONNX를 공유하는 explicit 경로만 지원한다.)
    fp16=True면 FP32 ONNX에서 BuilderFlag.FP16으로 half-precision 커널을 허용한다.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    try:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
    except (AttributeError, TypeError):
        network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX 파싱 실패: {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if fp16 and hasattr(trt.BuilderFlag, "FP16"):
        # TRT <= 10. TRT 11은 플래그가 제거되어 FP16 ONNX(모델 자료형)로 정밀도를 지정한다.
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT 엔진 빌드 실패")
    Path(engine_path).write_bytes(serialized)
    if fp16:
        return "fp16"
    return "explicit_qdq" if "int8" in Path(onnx_path).stem else "fp32"


# ---------------------------------------------------------------- ORT INT8 양자화

def quantize_onnx_int8(
    onnx_path: str,
    out_path: str,
    calib_imgs: list[np.ndarray],
    imgsz: int,
    variant: str,
    exclude_patterns: tuple[str, ...] = ("/model.23/",),
) -> None:
    """static QDQ INT8 양자화. 탐지 헤드(model.23, bbox 디코드)는 제외 —
    헤드까지 양자화하면 박스 출력이 무너져 탐지가 사라진다.

    variant:
      "u8s8": activation QUInt8(비대칭) — x86 CPU(VNNI)에서 빠름. ORT CPU용.
      "s8s8_sym": activation QInt8 대칭 — TensorRT explicit INT8은 zero-point 0만 허용.
        TRT용은 attention 블록(model.10)도 제외한다(0차원 상수 per-channel Q/DQ를 TRT가 거부).
    """
    import onnx as onnx_lib
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    import onnxruntime as ort

    pre_path = str(Path(out_path).with_suffix(".pre.onnx"))
    quant_pre_process(onnx_path, pre_path)

    graph = onnx_lib.load(pre_path).graph
    exclude = [n.name for n in graph.node if any(p in n.name for p in exclude_patterns)]

    input_name = ort.InferenceSession(
        pre_path, providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.it = iter([{input_name: preprocess(im, imgsz)} for im in calib_imgs])

        def get_next(self):
            return next(self.it, None)

    act_type = QuantType.QUInt8 if variant == "u8s8" else QuantType.QInt8
    extra = {
        "ActivationSymmetric": variant == "s8s8_sym",
        "WeightSymmetric": True,
        # TRT 11은 Int32 bias DequantizeLinear를 거부 → TRT 변형은 bias를 FP로 유지
        "QuantizeBias": variant != "s8s8_sym",
    }
    quantize_static(
        pre_path, out_path, _Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=act_type,
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=exclude,
        extra_options=extra,
    )
    Path(pre_path).unlink(missing_ok=True)


# ---------------------------------------------------------------- 벤치·sanity

def bench(runner, images: list[np.ndarray], warmup: int, n_iter: int) -> list[float]:
    for i in range(warmup):
        runner.detect(images[i % len(images)])
    times = []
    for i in range(n_iter):
        t0 = time.perf_counter()
        runner.detect(images[i % len(images)])
        times.append(time.perf_counter() - t0)
    return times


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def sanity_iou(runner_a, runner_b, images: list[np.ndarray]) -> dict:
    """같은 이미지에서 FP32 vs INT8 박스 매칭 IoU 평균 (합격 ≥ 0.9)."""
    ious, n_a, n_b, n_match = [], 0, 0, 0
    for img in images:
        da = runner_a.detect(img).cpu().numpy()
        db = runner_b.detect(img).cpu().numpy()
        n_a += len(da)
        n_b += len(db)
        if len(da) == 0 or len(db) == 0:
            continue
        m = _iou_matrix(da, db)
        while m.size and m.max() > IOU_MATCH_MIN:
            i, j = np.unravel_index(m.argmax(), m.shape)
            ious.append(float(m[i, j]))
            n_match += 1
            m = np.delete(np.delete(m, i, 0), j, 1)
    mean_iou = float(np.mean(ious)) if ious else None
    passed = mean_iou is not None and mean_iou >= 0.9
    out = {"mean_iou": mean_iou, "n_boxes_fp32": n_a, "n_boxes_int8": n_b,
           "n_matched": n_match, "passed": passed}
    if not passed:
        out["failure_cause_estimate"] = (
            "탐지 자체가 없음 — 임시(비달표면) 이미지에서 사전학습 모델 반응 부족"
            if mean_iou is None
            else "양자화 민감 레이어로 인한 박스 위치/신뢰도 변형 — 캘리브레이션 데이터가 "
                 "임시 이미지라면 실제 도메인 이미지로 재캘리브레이션 필요"
        )
    return out


# ---------------------------------------------------------------- meta·저장

def bench_meta(cfg: dict, config_path: str, extra: dict) -> dict:
    import onnxruntime as ort
    import torch
    import ultralytics

    meta = result_meta(config_path)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        driver = None
    try:
        power = subprocess.check_output(["powercfg", "/getactivescheme"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        power = None
    try:
        import tensorrt as trt

        trt_ver = trt.__version__
    except ImportError:
        trt_ver = None
    meta.update({
        "gpu": gpu, "driver_version": driver, "power_scheme": power,
        "cpu_threads": int(cfg["bench"]["cpu_threads"]),
        "versions": {"torch": torch.__version__, "tensorrt": trt_ver,
                     "onnxruntime": ort.__version__, "ultralytics": ultralytics.__version__},
        "model": cfg["detector"]["model"], "imgsz": int(cfg["detector"]["imgsz"]),
        "warmup": int(cfg["bench"]["warmup"]), "n_iter": int(cfg["bench"]["n_iter"]),
    })
    meta.update(extra)
    return meta


def save_tau_json(path: Path, samples: list[float], meta: dict) -> None:
    arr = np.asarray(samples)
    payload = {
        "samples_s": [float(s) for s in samples],
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std()),
        "meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {path}  median={payload['median_s']*1e3:.2f} ms  p95={payload['p95_s']*1e3:.2f} ms",
          flush=True)


def make_hist_fig(result_files: dict[str, Path], fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(result_files)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, (label, path) in zip(axes, result_files.items()):
        if not path.exists():
            ax.set_title(f"{label} (missing)")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        s = np.asarray(data["samples_s"]) * 1e3
        ax.hist(s, bins=50)
        ax.axvline(data["median_s"] * 1e3, color="r", linestyle="--",
                   label=f"median {data['median_s']*1e3:.2f} ms")
        ax.set_title(label)
        ax.set_xlabel("τ_det [ms]")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("τ_det = preprocess + inference + NMS (batch 1)")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"fig: {fig_path}", flush=True)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results", help="tau_*.json 출력 디렉터리")
    ap.add_argument("--fig", default="figs/p2_tau_hist.png")
    ap.add_argument("--n-sanity", type=int, default=20)
    ap.add_argument("--model", default=None, help="config detector.model 대신 쓸 가중치 (P5 학습 모델)")
    ap.add_argument("--backends", default="all",
                    help="쉼표 구분 부분 실행 (trt_fp32,trt_fp16,trt_int8,ort_cpu_fp32,ort_cpu_int8). all이면 전부")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    imgsz = int(cfg["detector"]["imgsz"])
    conf = float(cfg["detector"]["conf"])
    warmup, n_iter = int(cfg["bench"]["warmup"]), int(cfg["bench"]["n_iter"])
    threads = int(cfg["bench"]["cpu_threads"])

    from ultralytics import YOLO

    model = YOLO(args.model or cfg["detector"]["model"])
    onnx_path = model.export(format="onnx", imgsz=imgsz, batch=1, dynamic=False, device="cpu")
    print(f"onnx: {onnx_path}", flush=True)

    calib_imgs, calib_source = load_calib_images(cfg, rng)
    bench_imgs = temporary_images(args.n_sanity, rng)
    print(f"calib: {calib_source} ({len(calib_imgs)}장)", flush=True)

    files = {
        "trt_fp32": out_dir / "tau_trt_fp32.json",
        "trt_fp16": out_dir / "tau_trt_fp16.json",
        "trt_int8": out_dir / "tau_trt_int8.json",
        "ort_cpu_fp32": out_dir / "tau_ort_cpu_fp32.json",
        "ort_cpu_int8": out_dir / "tau_ort_cpu_int8.json",
    }
    selected = set(files) if args.backends == "all" else set(args.backends.split(","))
    unknown = selected - set(files)
    if unknown:
        raise SystemExit(f"--backends에 알 수 없는 키: {sorted(unknown)}")
    runners: dict[str, object] = {}

    # --- INT8 static 양자화: 백엔드별 변형 (ORT CPU는 U8S8, TRT explicit는 대칭 S8S8)
    stem = str(Path(onnx_path).with_suffix(""))

    # --- FP16 ONNX (TRT 11은 빌더 플래그 대신 모델 자료형으로 정밀도를 지정)
    half_onnx = None
    if "trt_fp16" in selected:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("FP16 ONNX export는 CUDA 디바이스가 필요하다")
            p = model.export(format="onnx", imgsz=imgsz, batch=1, dynamic=False,
                             half=True, device=0)
            half_onnx = f"{stem}_half.onnx"
            Path(half_onnx).unlink(missing_ok=True)
            Path(p).rename(half_onnx)
            # half export가 FP32 ONNX를 같은 이름으로 덮어썼으므로 복원
            onnx_path = model.export(format="onnx", imgsz=imgsz, batch=1, dynamic=False,
                                     device="cpu")
            print(f"fp16 onnx: {half_onnx}", flush=True)
        except Exception as exc:
            print(f"FP16 ONNX export 실패: {type(exc).__name__}: {exc}", flush=True)
    int8_variants = {}
    int8_targets = [("ort", "u8s8"), ("trt", "s8s8_sym")]
    for key, variant in int8_targets:
        if f"{key}_int8" not in selected and f"{key}_cpu_int8" not in selected:
            continue
        path = f"{stem}_int8_{key}.onnx"
        patterns = ("/model.23/",) if key == "ort" else ("/model.23/", "/model.10/")
        try:
            quantize_onnx_int8(onnx_path, path, calib_imgs, imgsz, variant, patterns)
            int8_variants[key] = path
            print(f"int8 onnx[{variant}]: {path}", flush=True)
        except Exception as exc:
            print(f"INT8 양자화[{variant}] 실패: {type(exc).__name__}: {exc}", flush=True)

    # --- TensorRT
    for prec, src, fp16 in (("fp32", onnx_path, False), ("fp16", half_onnx, True),
                            ("int8", int8_variants.get("trt"), False)):
        if f"trt_{prec}" not in selected or src is None:
            continue
        try:
            eng = f"{stem}_{prec}.engine"
            method = build_trt_engine(src, eng, fp16=fp16)
            print(f"engine[{prec}]: {eng} ({method})", flush=True)
            runners[f"trt_{prec}"] = TrtRunner(eng, imgsz, conf)
        except Exception as exc:
            print(f"TensorRT[{prec}] 실패: {type(exc).__name__}: {exc}", flush=True)

    # --- ORT CPU
    if "ort_cpu_fp32" in selected:
        runners["ort_cpu_fp32"] = OrtRunner(onnx_path, threads, imgsz, conf)
    if "ort_cpu_int8" in selected and "ort" in int8_variants:
        runners["ort_cpu_int8"] = OrtRunner(int8_variants["ort"], threads, imgsz, conf)

    # --- sanity (감정밀도 vs FP32)
    sanity: dict[str, dict] = {}
    sanity_pairs = {
        "trt": ("trt_fp32", "trt_int8"),
        "ort_cpu": ("ort_cpu_fp32", "ort_cpu_int8"),
        "trt_fp16": ("trt_fp32", "trt_fp16"),
    }
    for name, (ka, kb) in sanity_pairs.items():
        a, b = runners.get(ka), runners.get(kb)
        if a is not None and b is not None:
            sanity[name] = sanity_iou(a, b, bench_imgs)
            print(f"sanity[{name}]: {sanity[name]}", flush=True)

    # --- 벤치마크
    sanity_key = {"trt_int8": "trt", "ort_cpu_int8": "ort_cpu", "trt_fp16": "trt_fp16"}
    for key, runner in runners.items():
        samples = bench(runner, bench_imgs, warmup, n_iter)
        meta = bench_meta(cfg, args.config, {
            "backend": key, "calib": calib_source, "seed": args.seed,
            "int8_variant": {"trt_int8": "s8s8_sym", "ort_cpu_int8": "u8s8"}.get(key),
            "sanity_vs_fp32": sanity.get(sanity_key.get(key, "")),
        })
        save_tau_json(files[key], samples, meta)

    make_hist_fig(files, Path(args.fig))


if __name__ == "__main__":
    main()
