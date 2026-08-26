"""ZEM/ZEV 유도 법칙, t_go = T_f − t 고정 (계약 §2.5).  # TODO(oct): 최적 t_go 탐색"""
from __future__ import annotations

import numpy as np


def zem_zev(r_hat: np.ndarray, v_hat: np.ndarray, t_go: float, cfg: dict) -> np.ndarray:
    """ZEM/ZEV 가속 명령. r_f = 0, v_f = 0 (L 원점 연착륙). |a_T| > a_max이면 방향 유지 포화.

    t_go < t_go_min 처리(직전 명령 유지)는 호출자(sim/loop.py)가 담당한다.
    """
    g = np.asarray(cfg["moon"]["g"], dtype=float)
    a_max = float(cfg["scenario"]["a_max"])
    zem = -(r_hat + v_hat * t_go + 0.5 * g * t_go**2)
    zev = -(v_hat + g * t_go)
    a_T = (6.0 / t_go**2) * zem - (2.0 / t_go) * zev
    norm = float(np.linalg.norm(a_T))
    if norm > a_max:
        a_T = a_T * (a_max / norm)
    return a_T
