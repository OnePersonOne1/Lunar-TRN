"""P3 라벨러 테스트: 화면 밖 제외, p_min 필터, 경계 클리핑(50% 규칙) (계약 §2.3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from perception.camera import K_cam, backproject_ray
from perception.labeler import label_frame

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _crater(x: float, y: float, D: float) -> list[float]:
    return [x, y, 0.0, D]


def test_center_crater_labeled(cfg: dict) -> None:
    r = np.array([0.0, 0.0, 20000.0])
    cat = np.array([_crater(0.0, 0.0, 2000.0)])
    labels, ids = label_frame(cat, r, cfg)
    assert len(labels) == 1
    cx, cy, w, h = labels[0]
    assert cx == pytest.approx(0.5) and cy == pytest.approx(0.5)
    f = K_cam(cfg)[0, 0]
    d_px = f * 2000.0 / 20000.0
    assert w == pytest.approx(d_px / cfg["camera"]["W"])
    assert h == pytest.approx(d_px / cfg["camera"]["H"])


def test_offscreen_crater_excluded(cfg: dict) -> None:
    r = np.array([0.0, 0.0, 20000.0])
    # 화면 밖 동쪽으로 멀리 (수직 FOV 60°, 고도 20 km → 반화각 접선 0.577 → x > 11.5 km는 밖)
    cat = np.array([_crater(50000.0, 0.0, 2000.0)])
    labels, _ = label_frame(cat, r, cfg)
    assert len(labels) == 0


def test_small_crater_below_p_min_excluded(cfg: dict) -> None:
    r = np.array([0.0, 0.0, 20000.0])
    f = K_cam(cfg)[0, 0]
    p_min = cfg["catalog"]["p_min_px"]
    D_small = (p_min - 1) * 20000.0 / f  # 투영 직경 p_min−1 px
    cat = np.array([_crater(0.0, 0.0, D_small)])
    labels, _ = label_frame(cat, r, cfg)
    assert len(labels) == 0


def test_boundary_clipping_50_percent(cfg: dict) -> None:
    """중심이 화면 안이지만 경계에 걸친 크레이터: 클리핑 후 50% 미만이면 제외."""
    W, H = cfg["camera"]["W"], cfg["camera"]["H"]
    r = np.array([0.0, 0.0, 20000.0])
    f = K_cam(cfg)[0, 0]
    # 화면 오른쪽 끝(u ≈ W-1)에 중심이 오는 점을 역투영으로 구성
    d = backproject_ray(W - 1.0, H / 2.0, r, cfg)
    s = (0.0 - r[2]) / d[2]  # z=0 평면과 교차
    p = r + s * d
    D = 3000.0
    cat = np.array([[p[0], p[1], 0.0, D]])
    labels, _ = label_frame(cat, r, cfg)
    # bbox의 오른쪽 절반이 거의 전부 잘림 → 가시 비율 ≈ 50% 부근. 규칙 판정과 일치하는지 확인
    d_px = f * D / (r[2] - 0.0)
    visible = (W - ((W - 1.0) - d_px / 2.0)) * d_px  # 클리핑 후 면적
    expect_kept = visible >= 0.5 * d_px * d_px
    assert (len(labels) == 1) == expect_kept
    if labels.size:
        # 클리핑된 bbox는 화면 안에 있어야 한다
        cx, cy, w, h = labels[0]
        assert cx + w / 2.0 <= 1.0 + 1e-9


def test_clipped_bbox_shrinks(cfg: dict) -> None:
    """왼쪽 경계 살짝 걸침: 라벨은 유지되고 bbox 폭이 줄어든다."""
    W, H = cfg["camera"]["W"], cfg["camera"]["H"]
    r = np.array([0.0, 0.0, 20000.0])
    f = K_cam(cfg)[0, 0]
    D = 3000.0
    d_px = f * D / r[2]
    # 중심 u = d_px/4 → 왼쪽으로 bbox의 1/4이 잘림 (가시 75% ≥ 50%)
    d = backproject_ray(d_px / 4.0, H / 2.0, r, cfg)
    s = (0.0 - r[2]) / d[2]
    p = r + s * d
    cat = np.array([[p[0], p[1], 0.0, D]])
    labels, _ = label_frame(cat, r, cfg)
    assert len(labels) == 1
    _, _, w, _ = labels[0]
    assert w * W == pytest.approx(0.75 * d_px, rel=1e-3)
