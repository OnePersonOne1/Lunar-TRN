"""카탈로그 매칭 대응쌍으로 PnP(RANSAC) L-프레임 위치 추정 (계약 §2.4 측정).

cv2.solvePnPRansac(3D 크레이터 중심 ↔ 2D 탐지 중심, K_c, 왜곡 없음) → r_PnP = −Rᵀ·t.
valid: 대응쌍 ≥ 4, RANSAC 인라이어 ≥ measurement.n_min_inliers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.camera import K_cam  # noqa: E402

RANSAC_REPROJ_PX = 8.0
RANSAC_ITERS = 200


def solve_pnp(points_L: np.ndarray, points_uv: np.ndarray, cfg: dict) -> dict:
    """대응쌍으로 카메라(착륙선) 위치를 푼다.

    반환 dict: r_PnP (3,) | None, valid, n_pairs, n_inliers, inlier_idx, reproj_err_px.
    """
    n_min = int(cfg["measurement"]["n_min_inliers"])
    n_pairs = len(points_L)
    out = {"r_PnP": None, "valid": False, "n_pairs": n_pairs,
           "n_inliers": 0, "inlier_idx": [], "reproj_err_px": None}
    if n_pairs < 4:  # 계약 §2.4: 대응쌍 ≥ 4
        return out

    obj = np.ascontiguousarray(points_L, dtype=np.float64).reshape(-1, 1, 3)
    img = np.ascontiguousarray(points_uv, dtype=np.float64).reshape(-1, 1, 2)
    K_c = K_cam(cfg)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, K_c, None,
        iterationsCount=RANSAC_ITERS,
        reprojectionError=RANSAC_REPROJ_PX,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None:
        return out
    inl = inliers.ravel().tolist()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K_c, None, rvec, tvec)  # 정밀화
    R, _ = cv2.Rodrigues(rvec)
    r_pnp = (-R.T @ tvec).ravel()

    proj, _ = cv2.projectPoints(obj[inl], rvec, tvec, K_c, None)
    err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img[inl].reshape(-1, 2), axis=1)))

    out.update({
        "r_PnP": r_pnp,
        "valid": len(inl) >= n_min,
        "n_inliers": len(inl),
        "inlier_idx": inl,
        "reproj_err_px": err,
    })
    return out
