"""P3 카탈로그 변환 테스트: 등장방형 평면 근사 (계약 §2.1)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from data.catalog import latlon_to_xy

R_M = 1737400.0


def test_origin_maps_to_zero() -> None:
    x, y = latlon_to_xy(10.0, 20.0, 10.0, 20.0, R_M)
    assert x == pytest.approx(0.0) and y == pytest.approx(0.0)


def test_one_degree_north() -> None:
    _, y = latlon_to_xy(11.0, 20.0, 10.0, 20.0, R_M)
    assert y == pytest.approx(R_M * math.radians(1.0))


def test_one_degree_east_scaled_by_cos_lat0() -> None:
    lat0 = 45.0
    x, _ = latlon_to_xy(45.0, 21.0, lat0, 20.0, R_M)
    assert x == pytest.approx(R_M * math.radians(1.0) * math.cos(math.radians(lat0)))


def test_vectorized() -> None:
    lat = np.array([10.0, 10.5, 9.5])
    lon = np.array([20.0, 20.0, 20.5])
    x, y = latlon_to_xy(lat, lon, 10.0, 20.0, R_M)
    assert x.shape == (3,) and y.shape == (3,)
    assert y[1] > 0 > y[2]
    assert x[2] > 0
