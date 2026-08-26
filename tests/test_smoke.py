"""P0 스모크 테스트: config 로드, 전 모듈 import, K_cam 검증."""
from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    "sim.dynamics",
    "sim.guidance",
    "sim.ekf",
    "sim.measurement",
    "sim.loop",
    "sim.mc",
    "perception.camera",
    "perception.labeler",
    "perception.detect",
    "perception.associate",
    "perception.pnp",
    "perception.bench_tau",
    "data.crop",
    "data.catalog",
    "unity.client",
]


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_config_loads(cfg: dict) -> None:
    for key in (
        "seed", "moon", "site", "camera", "catalog", "trn_band", "imu",
        "scenario", "ekf", "measurement", "tau", "mc", "detector",
        "dataset", "bench", "unity",
    ):
        assert key in cfg, f"config.yaml에 {key} 섹션이 없다"
    assert cfg["moon"]["g"] == [0.0, 0.0, -1.62]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_K_cam(cfg: dict) -> None:
    from perception.camera import K_cam

    K_c = K_cam(cfg)
    W = cfg["camera"]["W"]
    H = cfg["camera"]["H"]
    f = H / (2.0 * math.tan(math.radians(cfg["camera"]["fov_v_deg"]) / 2.0))
    expected = np.array(
        [
            [f, 0.0, W / 2.0],
            [0.0, f, H / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert K_c.shape == (3, 3)
    np.testing.assert_allclose(K_c, expected, rtol=1e-12)
