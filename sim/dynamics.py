"""3-DOF 참값 동역학: ṙ = v, v̇ = g + a_T, RK4 전파 (계약 §2.4)."""
from __future__ import annotations

import numpy as np


def rk4_step(x: np.ndarray, a_T: np.ndarray, dt: float, g: np.ndarray) -> np.ndarray:
    """상태 x = [r; v] ∈ ℝ⁶를 한 스텝 전파한다. a_T는 스텝 동안 상수(ZOH)."""
    a = g + a_T

    def f(s: np.ndarray) -> np.ndarray:
        return np.concatenate([s[3:], a])

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
