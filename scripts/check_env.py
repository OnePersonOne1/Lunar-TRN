"""환경 점검 CLI: torch·CUDA·GPU 정보와 패키지 import 가능 여부를 results/env.json에 기록."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PACKAGES: list[str] = [
    "numpy",
    "scipy",
    "matplotlib",
    "yaml",
    "cv2",
    "rasterio",
    "torch",
    "torchvision",
    "ultralytics",
    "tensorrt",
    "onnxruntime",
    "openvino",
    "pytest",
]


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _check_imports() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            out[name] = {"ok": True, "version": getattr(mod, "__version__", None)}
        except Exception as exc:
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


def _check_gpu(torch_ok: bool) -> dict:
    gpu: dict = {"available": False}
    if not torch_ok:
        return gpu
    import torch

    gpu["torch_version"] = torch.__version__
    gpu["cuda_version"] = torch.version.cuda
    gpu["available"] = torch.cuda.is_available()
    if gpu["available"]:
        gpu["name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        gpu["compute_capability"] = f"sm_{major}{minor}"
    return gpu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=0)  # CLI 규약 통일용, 여기서는 미사용
    parser.add_argument("--out", default="results/env.json")
    args = parser.parse_args()

    imports = _check_imports()
    gpu = _check_gpu(imports["torch"]["ok"])

    report = {
        "meta": {
            "git_hash": _git_hash(),
            "config_hash": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": platform.python_version(),
                "gpu": gpu.get("name"),
            },
        },
        "gpu": gpu,
        "imports": imports,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"gpu": gpu, "out": str(out_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
