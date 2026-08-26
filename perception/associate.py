"""데이터 연관: EKF 예측 pose로 카탈로그 투영 후 최근접 매칭.  # TODO(oct): 기하 불변량 lost-in-space 매칭

게이트 반경(px) 안의 카탈로그 후보가 정확히 1개인 탐지만 채택한다(2개 이상이면 모호 → 기각).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.camera import project  # noqa: E402


def associate(
    det_uv: np.ndarray, r_pred: np.ndarray, catalog: np.ndarray, cfg: dict
) -> list[tuple[int, int]]:
    """탐지 중심 ↔ 카탈로그 대응을 만든다.

    det_uv: (N,2) 탐지 박스 중심 px. r_pred: EKF 예측 위치. catalog: (M,4) [x,y,z,D].
    반환: [(det_idx, catalog_idx), ...]. 게이트 안 후보 2개 이상인 탐지는 기각.
    같은 카탈로그 크레이터가 여러 탐지에 잡히면 가장 가까운 탐지만 남긴다.
    """
    gate = float(cfg["measurement"]["assoc_gate_px"])
    W, H = float(cfg["camera"]["W"]), float(cfg["camera"]["H"])
    uv, _, valid = project(catalog[:, :3], r_pred, cfg)
    usable = valid & (uv[:, 0] >= -gate) & (uv[:, 0] < W + gate) \
        & (uv[:, 1] >= -gate) & (uv[:, 1] < H + gate)
    idx_cat = np.flatnonzero(usable)
    if len(idx_cat) == 0 or len(det_uv) == 0:
        return []
    cat_uv = uv[idx_cat]

    pairs: dict[int, tuple[int, float]] = {}  # catalog_idx → (det_idx, dist)
    for d in range(len(det_uv)):
        dist = np.hypot(cat_uv[:, 0] - det_uv[d, 0], cat_uv[:, 1] - det_uv[d, 1])
        in_gate = np.flatnonzero(dist <= gate)
        if len(in_gate) != 1:  # 후보 0개(매칭 없음) 또는 ≥2개(모호) → 기각
            continue
        c = int(idx_cat[in_gate[0]])
        dd = float(dist[in_gate[0]])
        if c not in pairs or dd < pairs[c][1]:
            pairs[c] = (d, dd)
    return [(det, c) for c, (det, _) in sorted(pairs.items(), key=lambda kv: kv[1][0])]
