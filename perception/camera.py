"""핀홀 카메라 모델(계약 §2.2)의 단일 구현처. 라벨러·PnP·Unity 검증은 모두 이 모듈을 import한다.

카메라 프레임 C: x_C = East, y_C = South, z_C = Down. R_{C←L} = diag(1, −1, −1).
카메라 원점 = 착륙선 위치 r, 자세는 nadir 고정(§1).
"""
from __future__ import annotations

import math

import numpy as np

# R_{C←L} = diag(1, -1, -1): 대각 행렬이므로 곱 대신 원소별 부호 반전으로 적용한다.
R_cam_from_L = np.diag([1.0, -1.0, -1.0])


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


def project(points_L: np.ndarray, r: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L 프레임 점들을 이미지로 투영한다.

    p_C = R_{C←L}·(p_L − r), u = f·x_C/z_C + c_x, v = f·y_C/z_C + c_y.
    반환: (uv (N,2), z_C (N,), valid (N,)). z_C ≤ 0인 점은 무효(uv는 NaN).
    """
    K_c = K_cam(cfg)
    f, c_x, c_y = K_c[0, 0], K_c[0, 2], K_c[1, 2]
    p = np.atleast_2d(np.asarray(points_L, dtype=float)) - np.asarray(r, dtype=float)[:3]
    x_C, y_C, z_C = p[:, 0], -p[:, 1], -p[:, 2]
    valid = z_C > 0.0
    uv = np.full((len(p), 2), np.nan)
    uv[valid, 0] = f * x_C[valid] / z_C[valid] + c_x
    uv[valid, 1] = f * y_C[valid] / z_C[valid] + c_y
    return uv, z_C, valid


def backproject_ray(u: float, v: float, r: np.ndarray, cfg: dict) -> np.ndarray:
    """픽셀 (u, v)의 시선 방향 단위벡터를 L 프레임으로 돌려준다.

    광선 원점은 착륙선 위치 r (방향 벡터 자체는 r에 무관, nadir 자세 고정).
    """
    K_c = K_cam(cfg)
    f, c_x, c_y = K_c[0, 0], K_c[0, 2], K_c[1, 2]
    d_C = np.array([(u - c_x) / f, (v - c_y) / f, 1.0])
    d_L = np.array([d_C[0], -d_C[1], -d_C[2]])  # R_{C←L}ᵀ = diag(1,-1,-1)
    return d_L / np.linalg.norm(d_L)
