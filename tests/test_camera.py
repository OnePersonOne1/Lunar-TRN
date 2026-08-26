"""P3 카메라 테스트: 투영↔역투영 round-trip, 방향 규약, z_C ≤ 0 무효 (계약 §2.2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from perception.camera import K_cam, backproject_ray, project

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_nadir_point_at_principal_point(cfg: dict) -> None:
    r = np.array([100.0, -200.0, 20000.0])
    p = np.array([[100.0, -200.0, 5000.0]])  # 카메라 바로 아래
    uv, z_C, valid = project(p, r, cfg)
    K_c = K_cam(cfg)
    assert valid[0]
    assert z_C[0] == pytest.approx(15000.0)
    np.testing.assert_allclose(uv[0], [K_c[0, 2], K_c[1, 2]])


def test_east_increases_u_north_decreases_v(cfg: dict) -> None:
    r = np.array([0.0, 0.0, 10000.0])
    below = np.array([0.0, 0.0, 0.0])
    east = np.array([500.0, 0.0, 0.0])
    north = np.array([0.0, 500.0, 0.0])
    uv, _, valid = project(np.vstack([below, east, north]), r, cfg)
    assert valid.all()
    assert uv[1, 0] > uv[0, 0] and uv[1, 1] == pytest.approx(uv[0, 1])  # 동쪽 → u 증가
    assert uv[2, 1] < uv[0, 1] and uv[2, 0] == pytest.approx(uv[0, 0])  # 북쪽 → v 감소


def test_point_above_camera_invalid(cfg: dict) -> None:
    r = np.array([0.0, 0.0, 10000.0])
    p = np.array([[0.0, 0.0, 15000.0]])  # 카메라 위 (z_C < 0)
    _, z_C, valid = project(p, r, cfg)
    assert not valid[0]
    assert z_C[0] < 0


def test_project_backproject_roundtrip(cfg: dict) -> None:
    rng = np.random.default_rng(0)
    r = np.array([1000.0, -2000.0, 25000.0])
    pts = r + rng.uniform(-1, 1, size=(20, 3)) * np.array([5000.0, 5000.0, 0.0]) \
        - np.array([0.0, 0.0, 20000.0])
    uv, z_C, valid = project(pts, r, cfg)
    assert valid.all()
    for k in range(len(pts)):
        d = backproject_ray(uv[k, 0], uv[k, 1], r, cfg)
        # 광선 r + s·d 가 원래 점을 지나야 한다
        s = np.dot(pts[k] - r, d)
        np.testing.assert_allclose(r + s * d, pts[k], atol=1e-6)
