"""폐루프 시뮬레이션 실행기: 동역학 → 측정 → EKF(지연 보상) → ZEM/ZEV → 추력 명령."""
from __future__ import annotations

import heapq
import math

import numpy as np

from sim.dynamics import rk4_step
from sim.ekf import EKF
from sim.guidance import zem_zev
from sim.measurement import StatMeasurementModel, TauSampler, measurement_R


def run_closed_loop(
    cfg: dict,
    seed: int,
    tau: float | str | None = None,
    fp_rate: float | None = None,
    delay_comp: bool | None = None,
    measurement: str = "stat",
    detector_path: str | None = None,
    frames_dir: str | None = None,
) -> dict:
    """폐루프 1회 실행.

    measurement: "stat"(통계 측정모델+EKF) | "truth"(참값 상태로 유도, 측정·EKF 없음,
    실현가능성 테스트용) | "unity"(렌더→탐지→연관→PnP, Unity 서버 필요).
    tau: float(고정) | None(config 샘플러) | "wallclock"(unity 전용: 실측 지연 사용).
    tau/fp_rate/delay_comp가 None이면 config 값을 쓴다.
    반환 dict: landing_xy, landing_v, t_land, traj_t, traj_true, traj_est, est_error,
    nees, a_cmd, gate_log, tau_log, meas_log(unity), meas_assumed.
    """
    if measurement not in ("stat", "truth", "unity"):
        raise ValueError(f"알 수 없는 measurement: {measurement}")
    use_ekf = measurement in ("stat", "unity")
    if tau == "wallclock" and measurement != "unity":
        raise ValueError("tau='wallclock'은 measurement='unity'에서만 쓴다")

    rng = np.random.default_rng(seed)
    g = np.asarray(cfg["moon"]["g"], dtype=float)
    sc = cfg["scenario"]
    T_f = float(sc["T_f"])
    t_go_min = float(sc["t_go_min"])
    dt = 1.0 / float(cfg["imu"]["rate_hz"])
    sigma_a = float(cfg["imu"]["sigma_a"])
    n_steps = int(round(T_f / dt))
    spf = int(round(cfg["imu"]["rate_hz"] / cfg["camera"]["rate_hz"]))  # 측정 주기(IMU 스텝 수)
    h_min = float(cfg["trn_band"]["h_min_m"])
    h_max = float(cfg["trn_band"]["h_max_m"])
    if delay_comp is None:
        delay_comp = bool(cfg["ekf"]["delay_compensation"])

    x = np.concatenate([sc["r0"], sc["v0"]]).astype(float)  # 참값 [r; v]

    meas_assumed = True
    ekf: EKF | None = None
    unity_model = None
    if use_ekf:
        if measurement == "unity":
            from sim.measurement import UnityMeasurementModel

            if detector_path is None:
                raise ValueError("measurement='unity'는 detector_path가 필요하다")
            unity_model = UnityMeasurementModel(cfg, rng, detector_path, frames_dir=frames_dir)
            meas_assumed = False
        else:
            meas_model = StatMeasurementModel(cfg, rng, fp_rate=fp_rate)
            meas_assumed = meas_model.assumed
        R_meas, _ = measurement_R(cfg)
        tau_is_number = tau is not None and tau != "wallclock"
        tau_sampler = TauSampler(cfg, rng) if (tau is None) else None
        if tau_is_number:
            tau_max = float(tau)
        elif tau == "wallclock":
            tau_max = float(cfg["tau"]["constant_s"]) * 10.0  # 버퍼 여유 (실측은 보통 이보다 짧다)
        else:
            tau_max = tau_sampler.max_tau
        buffer_len = int(math.ceil((tau_max + spf * dt) / dt)) + 2
        x0_err = rng.normal(0.0, np.asarray(cfg["ekf"]["x0_error"], dtype=float))
        ekf = EKF(
            x + x0_err,
            np.diag(np.asarray(cfg["ekf"]["P0_diag"], dtype=float)),
            sigma_a=sigma_a,
            gate_chi2=float(cfg["ekf"]["gate_chi2"]),
            g=g,
            t0=0.0,
            buffer_len=buffer_len,
        )

    traj_t = np.empty(n_steps + 1)
    traj_true = np.empty((n_steps + 1, 6))
    traj_est = np.empty((n_steps + 1, 6)) if use_ekf else None
    nees = np.empty(n_steps + 1) if use_ekf else None
    a_cmd = np.zeros((n_steps + 1, 3))

    def log_state(k: int, t: float) -> None:
        traj_t[k] = t
        traj_true[k] = x
        if use_ekf:
            traj_est[k] = ekf.x_hat
            e = ekf.x_hat - x
            nees[k] = float(e @ np.linalg.solve(ekf.P_cov, e))

    log_state(0, 0.0)
    pending: list[tuple[float, int, float, np.ndarray, float]] = []  # (t_arr, seq, t_c, z, τ)
    seq = 0
    gate_log: list[dict] = []
    meas_log: list[dict] = []
    a_T = np.zeros(3)
    t_land = T_f
    k_land = n_steps

    for k in range(n_steps):
        t = k * dt
        t_go = T_f - t
        if t_go >= t_go_min:
            state_g = x if measurement == "truth" else ekf.x_hat
            a_T = zem_zev(state_g[:3], state_g[3:], t_go, cfg)
        a_cmd[k] = a_T  # t_go < t_go_min이면 직전 명령 유지

        x = rk4_step(x, a_T, dt, g)
        t_next = (k + 1) * dt

        if use_ekf:
            a_imu = a_T + rng.normal(0.0, sigma_a, 3)
            ekf.predict(a_imu, dt)
            if (k + 1) % spf == 0 and h_min <= x[2] <= h_max:  # TRN 고도 구간에서만 촬영
                if unity_model is not None:
                    info = unity_model.sample_frame(x[:3], ekf.x_hat[:3], seq, t_next)
                    info["t_c"] = t_next
                    meas_log.append(info)
                    z, valid = info["z"], info["valid"]
                    if tau == "wallclock":
                        tau_k = info["tau_wallclock_s"]
                    elif tau is not None:
                        tau_k = float(tau)
                    else:
                        tau_k = tau_sampler.sample()
                else:
                    z, valid = meas_model.sample(x[:3])
                    tau_k = float(tau) if tau is not None else tau_sampler.sample()
                if valid:
                    heapq.heappush(pending, (t_next + tau_k, seq, t_next, z, tau_k))
                seq += 1
            while pending and pending[0][0] <= t_next + 1e-9:
                t_arr, _, t_c, z, tau_k = heapq.heappop(pending)
                if delay_comp:
                    accepted, d2 = ekf.delayed_update(z, t_c, R_meas)
                else:
                    accepted, d2 = ekf.update(z, R_meas)
                gate_log.append(
                    {"t_arr": t_arr, "t_c": t_c, "tau": tau_k, "d2": d2, "accepted": accepted}
                )

        log_state(k + 1, t_next)
        if x[2] <= 0.0:  # 착륙 판정 r_z ≤ 0 (평면 착륙)
            t_land = t_next
            k_land = k + 1
            break

    if unity_model is not None:
        unity_model.close()
    n = k_land + 1
    return {
        "landing_xy": traj_true[k_land, :2].copy(),
        "landing_v": float(np.linalg.norm(traj_true[k_land, 3:])),
        "t_land": t_land,
        "traj_t": traj_t[:n],
        "traj_true": traj_true[:n],
        "traj_est": traj_est[:n] if use_ekf else None,
        "est_error": (traj_est[:n] - traj_true[:n]) if use_ekf else None,
        "nees": nees[:n] if use_ekf else None,
        "a_cmd": a_cmd[:n],
        "gate_log": gate_log,
        "tau_log": [e["tau"] for e in gate_log],
        "meas_log": meas_log,
        "meas_assumed": meas_assumed,
    }
