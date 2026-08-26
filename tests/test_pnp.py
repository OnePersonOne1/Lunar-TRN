"""P5 PnP·연관 테스트: 무잡음 대응쌍 복원 < 1e-3 m, 이상치 30%에서 RANSAC 복원, 연관 게이트."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from perception.associate import associate
from perception.camera import project
from perception.pnp import solve_pnp

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _synthetic_scene(cfg: dict, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """r_true 아래 지면 크레이터 n개와 정확한 투영 uv를 만든다."""
    rng = np.random.default_rng(seed)
    r_true = np.array([500.0, -300.0, 25000.0])
    half = r_true[2] * 0.5  # 시야 안(반화각 tan 0.577)에 안전히 들어오는 범위
    pts = np.column_stack([
        r_true[0] + rng.uniform(-half, half, n),
        r_true[1] + rng.uniform(-half, half, n),
        rng.uniform(-200.0, 200.0, n),
    ])
    uv, _, valid = project(pts, r_true, cfg)
    assert valid.all()
    return r_true, pts, uv


def test_pnp_noiseless(cfg: dict) -> None:
    r_true, pts, uv = _synthetic_scene(cfg, 12, seed=1)
    res = solve_pnp(pts, uv, cfg)
    assert res["valid"] and res["n_inliers"] == 12
    assert np.linalg.norm(res["r_PnP"] - r_true) < 1e-3
    assert res["reproj_err_px"] < 1e-6


def test_pnp_with_30pct_outliers(cfg: dict) -> None:
    r_true, pts, uv = _synthetic_scene(cfg, 20, seed=2)
    rng = np.random.default_rng(3)
    uv_bad = uv.copy()
    out_idx = rng.choice(20, size=6, replace=False)
    uv_bad[out_idx] += rng.uniform(80.0, 300.0, size=(6, 2)) * rng.choice([-1, 1], size=(6, 2))
    res = solve_pnp(pts, uv_bad, cfg)
    assert res["valid"]
    assert np.linalg.norm(res["r_PnP"] - r_true) < 1.0  # RANSAC이 참값 복원
    assert set(res["inlier_idx"]).isdisjoint(set(out_idx.tolist()))


def test_pnp_too_few_pairs_invalid(cfg: dict) -> None:
    r_true, pts, uv = _synthetic_scene(cfg, 3, seed=4)
    res = solve_pnp(pts, uv, cfg)
    assert not res["valid"] and res["r_PnP"] is None


def test_associate_nearest_and_ambiguity(cfg: dict) -> None:
    r = np.array([0.0, 0.0, 20000.0])
    gate = cfg["measurement"]["assoc_gate_px"]
    # 카탈로그: 원점 크레이터 1개 + 화면 안 떨어진 크레이터 1개 + 서로 가까운 쌍(모호)
    m_per_px = r[2] / 886.8  # f≈886.8 (1024, 60°) — 대략 스케일
    cat = np.array([
        [0.0, 0.0, 0.0, 2000.0],
        [4000.0, 3000.0, 0.0, 2000.0],
        [-4000.0, -3000.0, 0.0, 2000.0],
        [-4000.0 - gate * 0.5 * m_per_px, -3000.0, 0.0, 2000.0],  # 위와 게이트 안 겹침
    ])
    uv, _, _ = project(cat[:, :3], r, cfg)
    det = np.array([
        uv[0] + [3.0, -2.0],   # 크레이터 0 근처
        uv[1] + [-5.0, 4.0],   # 크레이터 1 근처
        uv[2] + [1.0, 1.0],    # 크레이터 2/3 모두 게이트 안 → 모호, 기각돼야 함
    ])
    pairs = associate(det, r, cat, cfg)
    assert (0, 0) in pairs and (1, 1) in pairs
    assert all(d != 2 for d, _ in pairs)
