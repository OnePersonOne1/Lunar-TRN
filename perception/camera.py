"""핀홀 카메라 모델(계약 §2.2)의 단일 구현처. 라벨러·PnP·Unity 검증은 모두 이 모듈을 import한다."""
from __future__ import annotations

import math

import numpy as np


def K_cam(cfg: dict) -> np.ndarray:
    """config의 camera 섹션으로 내부 파라미터 행렬 K_c를 만든다.

    f = H / (2·tan(θ_v/2)), c_x = W/2, c_y = H/2.
    """
    cam = cfg["camera"]
    W = float(cam["W"])
    H = float(cam["H"])
    theta_v = math.radians(float(cam["fov_v_deg"]))
    f = H / (2.0 * math.tan(theta_v / 2.0))
    return np.array(
        [
            [f, 0.0, W / 2.0],
            [0.0, f, H / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
