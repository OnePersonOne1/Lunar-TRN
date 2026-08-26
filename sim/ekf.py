"""6-state EKF: 지연 보상(링버퍼 재전파) + χ² 게이팅 (계약 §2.4).

상태 x̂ = [r̂; v̂], 공분산 P_cov. 예측은 IMU 주기, 보정은 측정 도착 시.
F = [[I₃, Δt·I₃], [0₃, I₃]], G = [[½Δt²·I₃], [Δt·I₃]], Q = G·σ_a²·Gᵀ, H = [I₃ 0₃].
"""
from __future__ import annotations

import numpy as np

_I6 = np.eye(6)


class EKF:
    """지연 보상 링버퍼를 가진 선형 칼만 필터 (3-DOF TRN 측정용).

    predict()마다 (t, x̂, P, a_imu, dt)를 버퍼에 남긴다. delayed_update()는
    t_c 시점 스냅샷으로 되돌려 보정한 뒤 저장된 IMU 입력으로 현재 시각까지
    재전파하고, 버퍼의 후속 스냅샷도 보정 후 값으로 갱신한다(중첩 지연 대응).
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        sigma_a: float,
        gate_chi2: float,
        g: np.ndarray,
        t0: float = 0.0,
        buffer_len: int = 4096,
    ) -> None:
        self.x_hat = np.asarray(x0, dtype=float).copy()
        self.P_cov = np.asarray(P0, dtype=float).copy()
        self.sigma_a = float(sigma_a)
        self.gate_chi2 = float(gate_chi2)
        self.g = np.asarray(g, dtype=float).copy()
        self.t = float(t0)
        self.buffer_len = int(buffer_len)
        self._buf: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []

    # ------------------------------------------------------------ 예측

    def _propagate(self, a_imu: np.ndarray, dt: float) -> None:
        """상태·공분산을 dt만큼 전파 (버퍼 기록 없음). v̇ = g + a_IMU."""
        a = self.g + a_imu
        r, v = self.x_hat[:3], self.x_hat[3:]
        self.x_hat = np.concatenate([r + v * dt + 0.5 * a * dt**2, v + a * dt])
        F = _I6.copy()
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        G = np.vstack([0.5 * dt**2 * np.eye(3), dt * np.eye(3)])
        self.P_cov = F @ self.P_cov @ F.T + (self.sigma_a**2) * (G @ G.T)
        self.t += dt

    def predict(self, a_imu: np.ndarray, dt: float) -> None:
        """IMU 한 샘플만큼 전파하고 링버퍼에 스냅샷을 남긴다."""
        a_imu = np.asarray(a_imu, dtype=float)
        self._propagate(a_imu, dt)
        self._buf.append((self.t, self.x_hat.copy(), self.P_cov.copy(), a_imu.copy(), dt))
        if len(self._buf) > self.buffer_len:
            del self._buf[: len(self._buf) - self.buffer_len]

    # ------------------------------------------------------------ 보정

    def gate(self, nu: np.ndarray, S_inn: np.ndarray) -> bool:
        """χ² 게이트: d² = νᵀ·S⁻¹·ν ≤ gate_chi2 이면 통과."""
        return float(nu @ np.linalg.solve(S_inn, nu)) <= self.gate_chi2

    def update(self, z: np.ndarray, R_meas: np.ndarray) -> tuple[bool, float]:
        """측정 z = r ∈ ℝ³ 보정. 게이트 기각 시 상태 불변. (accepted, d²) 반환."""
        nu = np.asarray(z, dtype=float) - self.x_hat[:3]
        S_inn = self.P_cov[:3, :3] + R_meas
        d2 = float(nu @ np.linalg.solve(S_inn, nu))
        if d2 > self.gate_chi2:
            return False, d2
        K_gain = self.P_cov[:, :3] @ np.linalg.inv(S_inn)
        self.x_hat = self.x_hat + K_gain @ nu
        self.P_cov = self.P_cov - K_gain @ self.P_cov[:3, :]
        self.P_cov = 0.5 * (self.P_cov + self.P_cov.T)
        return True, d2

    def delayed_update(self, z: np.ndarray, t_c: float, R_meas: np.ndarray) -> tuple[bool, float]:
        """촬영시각 t_c의 측정 z를 지연 보상 보정한다.

        t_c 스냅샷 복원 → update → 버퍼 IMU 입력으로 현재 시각까지 재전파.
        t_c가 버퍼 범위 밖(너무 오래됨)이면 보상 없이 plain update로 폴백한다.
        """
        if not self._buf:
            return self.update(z, R_meas)
        dts = self._buf[0][4]
        idx = len(self._buf) - 1 - int(round((self._buf[-1][0] - t_c) / dts))
        if idx < 0:
            return self.update(z, R_meas)  # 버퍼보다 오래된 측정: 폴백
        t_i, x_i, P_i, a_i, dt_i = self._buf[idx]
        if abs(t_i - t_c) > 0.5 * dts:
            return self.update(z, R_meas)  # 스냅샷 불일치: 폴백
        self.t, self.x_hat, self.P_cov = t_i, x_i.copy(), P_i.copy()
        accepted, d2 = self.update(z, R_meas)
        self._buf[idx] = (t_i, self.x_hat.copy(), self.P_cov.copy(), a_i, dt_i)
        for j in range(idx + 1, len(self._buf)):
            t_j, _, _, a_j, dt_j = self._buf[j]
            self._propagate(a_j, dt_j)
            self._buf[j] = (t_j, self.x_hat.copy(), self.P_cov.copy(), a_j, dt_j)
        return accepted, d2
