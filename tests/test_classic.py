"""P7c: 고전 베이스라인(PCA 템플릿 탐지)·AP 지표 — 구현 전 작성한 테스트."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from perception.classic import PCADetector, train_pca_detector
from perception.metrics import average_precision, map_50_95, match_counts


@pytest.fixture()
def cfg() -> dict:
    with open("config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _box(cx: float, cy: float, d: float) -> list[float]:
    return [cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2]


# ---------------------------------------------------------------- metrics

def test_ap_perfect_detections() -> None:
    gts = [np.array([_box(50, 50, 20), _box(120, 80, 30)]), np.array([_box(30, 30, 10)])]
    dets = [np.array([_box(50, 50, 20) + [0.9], _box(120, 80, 30) + [0.8]]),
            np.array([_box(30, 30, 10) + [0.7]])]
    assert average_precision(dets, gts, 0.5) == pytest.approx(1.0)
    assert map_50_95(dets, gts) == pytest.approx(1.0)


def test_ap_false_positive_ranked_first_halves_precision() -> None:
    gts = [np.array([_box(50, 50, 20)])]
    dets = [np.array([_box(200, 200, 20) + [0.9], _box(50, 50, 20) + [0.5]])]
    # 순위 1 FP, 순위 2 TP → recall 1에서 precision 0.5 → AP 0.5
    assert average_precision(dets, gts, 0.5) == pytest.approx(0.5)


def test_ap_empty_detections_is_zero() -> None:
    gts = [np.array([_box(50, 50, 20)])]
    assert average_precision([np.empty((0, 5))], gts, 0.5) == 0.0


def test_match_counts() -> None:
    gts = [np.array([_box(50, 50, 20), _box(120, 80, 30)])]
    dets = [np.array([_box(50, 50, 20) + [0.9], _box(300, 300, 20) + [0.9]])]
    tp, fp, fn = match_counts(dets, gts, 0.5)
    assert (tp, fp, fn) == (1, 1, 1)


# ---------------------------------------------------------------- PCA 탐지기

def _synthetic_frame(rng: np.random.Generator, size: int, craters: list[tuple]) -> np.ndarray:
    """회색 배경 + 잡음 + 원형 크레이터(한쪽 밝고 반대쪽 어두운 그늘 패턴)."""
    import cv2

    img = np.full((size, size), 120, dtype=np.float32)
    img += rng.normal(0.0, 4.0, img.shape).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for cx, cy, d in craters:
        r = d / 2.0
        dist = np.hypot(xx - cx, yy - cy)
        inside = dist <= r
        shade = np.where(xx - cx < 0, -50.0, 40.0)
        img[inside] += shade[inside] * (1.0 - dist[inside] / r) ** 0.5
        cv2.circle(img, (int(cx), int(cy)), int(r), 160.0, 1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.dstack([img, img, img])


def _make_set(rng: np.random.Generator, n: int, size: int = 256) -> tuple[list, list]:
    imgs, gts = [], []
    for _ in range(n):
        craters = []
        for _ in range(3):
            d = float(rng.uniform(24, 60))
            cx, cy = rng.uniform(d, size - d, size=2)
            if all(np.hypot(cx - a, cy - b) > (d + c) for a, b, c in craters):
                craters.append((cx, cy, d))
        imgs.append(_synthetic_frame(rng, size, craters))
        gts.append(np.array([_box(cx, cy, d) for cx, cy, d in craters]))
    return imgs, gts


def test_pca_detector_train_save_load_detect(cfg: dict, tmp_path) -> None:
    rng = np.random.default_rng(0)
    imgs, gts = _make_set(rng, 12)
    c = dict(cfg["classic"])
    c.update({"d_min_px": 16, "d_max_px": 96, "n_tune_frames": 4, "n_neg_per_frame": 30})
    det, info = train_pca_detector(zip(imgs, gts), c, rng)
    assert det.basis.shape == (c["n_components"], c["patch_px"] ** 2)
    assert info["threshold"] >= c["score_floor"]

    path = tmp_path / "pca.npz"
    det.save(path)
    det2 = PCADetector.load(path)
    for img in imgs[:2]:
        a, b = det.detect(img), det2.detect(img)
        assert a.shape[1] == 6 and np.array_equal(a, b)

    # 학습 데이터와 같은 분포의 새 프레임에서 절반 이상은 맞춰야 한다(합성 = 쉬운 문제)
    vimgs, vgts = _make_set(np.random.default_rng(1), 6)
    dets = [det2.detect(im)[:, :5] for im in vimgs]
    tp, fp, fn = match_counts(dets, vgts, 0.5)
    assert tp / max(tp + fn, 1) >= 0.5


def test_pca_detector_deterministic(cfg: dict) -> None:
    imgs, gts = _make_set(np.random.default_rng(3), 6)
    c = dict(cfg["classic"])
    c.update({"d_min_px": 16, "d_max_px": 96, "n_tune_frames": 3, "n_neg_per_frame": 20,
              "n_components": 6})
    det_a, _ = train_pca_detector(zip(imgs, gts), c, np.random.default_rng(5))
    det_b, _ = train_pca_detector(zip(imgs, gts), c, np.random.default_rng(5))
    assert np.array_equal(det_a.basis, det_b.basis)
    assert det_a.threshold == det_b.threshold


def test_with_diameter_range_restricts_scales(cfg: dict, tmp_path) -> None:
    """고도 사전정보 변형: 미세 스케일을 빼면 후보 수가 줄고 저장·복원에도 유지된다."""
    imgs, gts = _make_set(np.random.default_rng(7), 6)
    c = dict(cfg["classic"])
    c.update({"d_min_px": 16, "d_max_px": 96, "n_tune_frames": 3, "n_neg_per_frame": 20,
              "n_components": 6})
    det, _ = train_pca_detector(zip(imgs, gts), c, np.random.default_rng(0))
    prior = det.with_diameter_range(32.0, c["d_max_px"])
    assert len(prior._ds) < len(det._ds)
    assert prior.threshold == det.threshold
    assert len(prior.candidates(imgs[0])) <= len(det.candidates(imgs[0]))

    path = tmp_path / "prior.npz"
    prior.save(path)
    assert PCADetector.load(path).params["d_min_px"] == 32.0


def test_detector_factory_dispatches_npz(cfg: dict, tmp_path) -> None:
    from perception.detect import Detector

    imgs, gts = _make_set(np.random.default_rng(4), 6)
    c = dict(cfg["classic"])
    c.update({"d_min_px": 16, "d_max_px": 96, "n_tune_frames": 3, "n_neg_per_frame": 20,
              "n_components": 6})
    det, _ = train_pca_detector(zip(imgs, gts), c, np.random.default_rng(0))
    path = tmp_path / "pca.npz"
    det.save(path)
    wrapped = Detector(str(path), cfg)
    assert wrapped.backend == "pca"
    assert wrapped.centers(imgs[0]).shape[1] == 2
