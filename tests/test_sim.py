"""P1 시뮬 코어 테스트: RK4 해석해, 참값 ZEM/ZEV 착륙, EKF NEES 일관성, 지연 보상, χ² 게이트."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.dynamics import rk4_step
from sim.ekf import EKF
from sim.guidance import zem_zev
from sim.loop import run_closed_loop
from sim.measurement import StatMeasurementModel, TauSampler
import sim.mc as mc

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------- dynamics

def test_rk4_free_fall_vs_analytic(cfg: dict) -> None:
    """자유낙하(a_T=0) 100 s: RK4 vs 해석해 위치 오차 < 1e-6 m."""
    g = np.array(cfg["moon"]["g"], float)
    dt = 1.0 / cfg["imu"]["rate_hz"]
    r0 = np.array([-150000.0, 200.0, 30000.0])
    v0 = np.array([900.0, -10.0, -40.0])
    x = np.concatenate([r0, v0])
    a_T = np.zeros(3)
    T = 100.0
    n = int(round(T / dt))
    for _ in range(n):
        x = rk4_step(x, a_T, dt, g)
    r_ana = r0 + v0 * T + 0.5 * g * T**2
    v_ana = v0 + g * T
    assert np.linalg.norm(x[:3] - r_ana) < 1e-6
    assert np.linalg.norm(x[3:] - v_ana) < 1e-9


# ---------------------------------------------------------------- guidance

def test_zem_zev_formula(cfg: dict) -> None:
    """포화 없는 구간에서 ZEM/ZEV 정의식과 일치."""
    g = np.array(cfg["moon"]["g"], float)
    r = np.array([-1000.0, 50.0, 2000.0])
    v = np.array([30.0, -1.0, -20.0])
    t_go = 100.0
    zem = -(r + v * t_go + 0.5 * g * t_go**2)
    zev = -(v + g * t_go)
    expected = (6.0 / t_go**2) * zem - (2.0 / t_go) * zev
    assert np.linalg.norm(expected) <= cfg["scenario"]["a_max"], "테스트 전제: 비포화"
    np.testing.assert_allclose(zem_zev(r, v, t_go, cfg), expected, rtol=1e-12)


def test_zem_zev_saturation_keeps_direction(cfg: dict) -> None:
    r = np.array([-150000.0, 0.0, 30000.0])
    v = np.array([900.0, 0.0, -40.0])
    t_go = 20.0  # 비현실적으로 짧은 t_go → 강한 포화
    a = zem_zev(r, v, t_go, cfg)
    a_max = cfg["scenario"]["a_max"]
    assert np.linalg.norm(a) == pytest.approx(a_max, rel=1e-9)


# ---------------------------------------------------------------- 참값 폐루프 실현가능성

def test_truth_state_landing(cfg: dict) -> None:
    """참값 상태 + ZEM/ZEV: 착륙 오차 < 1 m, 착륙 속도 < 0.1 m/s, 전 구간 |a_T| ≤ a_max."""
    res = run_closed_loop(cfg, seed=0, measurement="truth")
    err = float(np.linalg.norm(res["landing_xy"]))
    a_norm = np.linalg.norm(res["a_cmd"], axis=1)
    assert err < 1.0, f"착륙 오차 {err:.3f} m — scenario 재검토 필요"
    assert res["landing_v"] < 0.1, f"착륙 속도 {res['landing_v']:.3f} m/s"
    assert a_norm.max() <= cfg["scenario"]["a_max"] * (1 + 1e-9)


# ---------------------------------------------------------------- EKF NEES 일관성

@pytest.mark.slow
def test_ekf_nees_consistency(cfg: dict) -> None:
    """통계 측정모델 50 seed: 시간 평균 NEES가 χ²₆ 95% 구간 안.

    일관성 검정용 in-memory 설정: 짧은 horizon(T_f=30 s, TRN 구간 안에 머무름),
    초기 오차 1σ = sqrt(P0) (config의 보수적 P0와 실제 산포를 일치시켜야 필터가 정합).
    config.yaml 자체는 바꾸지 않는다.
    """
    from scipy import stats

    cfg2 = copy.deepcopy(cfg)
    cfg2["scenario"]["T_f"] = 30.0
    # PyYAML은 "1.0e6"을 문자열로 읽으므로(YAML 1.1 float 형식은 1.0e+6) float 변환
    cfg2["ekf"]["x0_error"] = [math.sqrt(float(p)) for p in cfg2["ekf"]["P0_diag"]]
    n_seeds = 50
    avg_nees = []
    for seed in range(n_seeds):
        res = run_closed_loop(cfg2, seed=seed, tau=0.0, fp_rate=0.0)
        nees = res["nees"]
        assert nees is not None and np.isfinite(nees).all()
        avg_nees.append(float(nees.mean()))
    mean_nees = float(np.mean(avg_nees))
    k = 6 * n_seeds
    lo = stats.chi2.ppf(0.025, k) / n_seeds
    hi = stats.chi2.ppf(0.975, k) / n_seeds
    assert lo < mean_nees < hi, f"NEES {mean_nees:.2f} ∉ [{lo:.2f}, {hi:.2f}]"


# ---------------------------------------------------------------- 지연 보상

def _make_ekf(cfg: dict, x0: np.ndarray, p_pos: float, p_vel: float) -> EKF:
    g = np.array(cfg["moon"]["g"], float)
    P0 = np.diag([p_pos] * 3 + [p_vel] * 3)
    return EKF(x0, P0, sigma_a=cfg["imu"]["sigma_a"],
               gate_chi2=cfg["ekf"]["gate_chi2"], g=g)


def test_delay_compensation_matches_no_delay(cfg: dict) -> None:
    """무잡음, τ=1 s: delayed_update == (t_c 시점 update 후 전파) 1e-6 이내.
    미보상(도착 시점 plain update)은 |v|·τ 규모의 위치 편향."""
    g = np.array(cfg["moon"]["g"], float)
    dt = 1.0 / cfg["imu"]["rate_hz"]
    r0 = np.array([0.0, 0.0, 20000.0])
    v0 = np.array([900.0, 0.0, -40.0])
    x_true = np.concatenate([r0, v0])
    tau = 1.0
    n_c = int(round(0.5 / dt))          # t_c = 0.5 s
    n_end = int(round((0.5 + tau) / dt))  # 도착 = 1.5 s
    R_meas = np.diag([1e-6] * 3)

    # 참값 궤적 (자유낙하, a_T = 0 → IMU 입력 0)
    truth = [x_true.copy()]
    for _ in range(n_end):
        truth.append(rk4_step(truth[-1], np.zeros(3), dt, g))
    z = truth[n_c][:3].copy()
    t_c = n_c * dt

    ekf_a = _make_ekf(cfg, x_true, 1e6, 1e4)   # 기준: t_c에서 즉시 보정
    ekf_b = _make_ekf(cfg, x_true, 1e6, 1e4)   # 지연 보상
    ekf_c = _make_ekf(cfg, x_true, 1e6, 1e4)   # 미보상
    for k in range(n_end):
        if k == n_c:
            acc, _ = ekf_a.update(z, R_meas)
            assert acc
        ekf_a.predict(np.zeros(3), dt)
        ekf_b.predict(np.zeros(3), dt)
        ekf_c.predict(np.zeros(3), dt)

    acc_b, _ = ekf_b.delayed_update(z, t_c, R_meas)
    assert acc_b
    assert np.linalg.norm(ekf_a.x_hat - ekf_b.x_hat) < 1e-6
    assert np.abs(ekf_a.P_cov - ekf_b.P_cov).max() < 1e-6

    acc_c, _ = ekf_c.update(z, R_meas)  # z(t_c)를 현재값처럼 취급
    assert acc_c
    bias = np.linalg.norm(ekf_c.x_hat[:3] - ekf_a.x_hat[:3])
    v_tau = np.linalg.norm(truth[n_c][3:]) * tau
    assert 0.5 * v_tau < bias < 1.5 * v_tau, f"미보상 편향 {bias:.1f} vs |v|τ {v_tau:.1f}"


# ---------------------------------------------------------------- 게이트

def test_gate_rejects_outlier_accepts_inlier(cfg: dict) -> None:
    x0 = np.array([0.0, 0.0, 20000.0, 900.0, 0.0, -40.0])
    ekf = _make_ekf(cfg, x0, 100.0, 1.0)  # 위치 분산 100 m² → σ_S ≈ sqrt(100+R)
    R_meas = np.diag([25.0] * 3)
    sigma_s = math.sqrt(100.0 + 25.0)
    z_in = x0[:3] + np.array([1.0, -1.0, 1.0]) / math.sqrt(3) * sigma_s
    acc, d2 = ekf.update(z_in, R_meas)
    assert acc and d2 < cfg["ekf"]["gate_chi2"]

    ekf2 = _make_ekf(cfg, x0, 100.0, 1.0)
    z_out = x0[:3] + np.array([10.0, 10.0, 10.0]) / math.sqrt(3) * sigma_s
    acc2, d2_2 = ekf2.update(z_out, R_meas)
    assert (not acc2) and d2_2 > cfg["ekf"]["gate_chi2"]
    np.testing.assert_array_equal(ekf2.x_hat, x0)  # 기각 시 상태 불변


def test_gate_reject_streak_recovers(cfg: dict) -> None:
    """연속 기각 N회 후 P 팽창으로 재획득: 편향 측정이 지속돼도 영구 기각되지 않는다."""
    x0 = np.array([0.0, 0.0, 20000.0, 900.0, 0.0, -40.0])
    ekf = _make_ekf(cfg, x0, 100.0, 1.0)
    ekf.reject_streak_n = 3
    ekf.reject_inflate = 4.0
    R_meas = np.diag([25.0] * 3)
    z_biased = x0[:3] + np.array([80.0, 0.0, 0.0])  # σ_S≈11.2의 ~7배 편향 → 기각 대상
    results = [ekf.update(z_biased, R_meas)[0] for _ in range(12)]
    assert not any(results[:3])          # 처음엔 기각
    assert any(results), "P 팽창 후에도 재획득 실패"
    # 팽창 없는 필터는 영구 기각
    ekf2 = _make_ekf(cfg, x0, 100.0, 1.0)
    assert not any(ekf2.update(z_biased, R_meas)[0] for _ in range(12))


# ---------------------------------------------------------------- 측정 모델·τ 샘플러

def test_stat_measurement_model(cfg: dict) -> None:
    cfg = copy.deepcopy(cfg)
    cfg["measurement"]["mode"] = "assumed"  # config 가정값 경로를 검증 (보정 오프셋은 test_p7b)
    rng = np.random.default_rng(0)
    r = np.array([100.0, -50.0, 20000.0])
    m0 = StatMeasurementModel(cfg, rng, fp_rate=0.0)
    errs = np.array([np.linalg.norm(m0.sample(r)[0] - r) for _ in range(200)])
    sig = np.linalg.norm(cfg["measurement"]["sigma_xyz_m"])
    assert errs.max() < 6.0 * sig

    m1 = StatMeasurementModel(cfg, rng, fp_rate=1.0)
    z, valid = m1.sample(r)
    assert valid
    assert np.linalg.norm(z - r) == pytest.approx(cfg["measurement"]["fp_offset_m"], rel=1e-9)


def test_tau_sampler_constant(cfg: dict) -> None:
    ts = TauSampler(cfg, np.random.default_rng(0))
    assert ts.sample() == cfg["tau"]["constant_s"]
    assert ts.max_tau == cfg["tau"]["constant_s"]


# ---------------------------------------------------------------- MC 통계

def test_mc_statistics() -> None:
    rng = np.random.default_rng(0)
    xy = rng.normal(0.0, [30.0, 10.0], size=(4000, 2))
    r50 = mc.cep(xy)
    # 이론 CEP(중앙값 반경)와 비교: 수치 적분 없이 몬테카를로 자체 일관성만 확인
    assert 20.0 < r50 < 40.0
    ell = mc.error_ellipse_95(xy)
    a, b = ell["semi_axes_m"]
    assert a > b
    assert a == pytest.approx(30.0 * math.sqrt(5.991), rel=0.1)
    assert b == pytest.approx(10.0 * math.sqrt(5.991), rel=0.1)
    lo, hi = mc.bootstrap_cep_ci(xy, n_boot=200, rng=rng)
    assert lo < r50 < hi
