"""P7b: 직렬 처리 모델(tau.serial)·오검출 오프셋 보정·ΔV — 구현 전 작성한 테스트."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import yaml

from sim.loop import run_closed_loop
from sim.measurement import StatMeasurementModel

# 수정 전 코드(커밋 1dda5e4, measurement.mode=assumed)에서 캡처한 골든 값.
# serial=false 경로는 이 값을 비트 단위로 재현해야 한다.
GOLDEN = {
    (0.5, 0): ([-60.64876553785154, -34.81380674233422], 0.2986238181503756),
    (2.5, 7): ([-21.085927982960886, -3.8638617816719965], 0.21179267436258847),
}


@pytest.fixture()
def cfg() -> dict:
    with open("config.yaml", encoding="utf-8") as fh:
        c = yaml.safe_load(fh)
    c["measurement"]["mode"] = "assumed"  # 골든 캡처 조건과 일치시킴
    return c


def test_serial_false_bit_exact(cfg: dict) -> None:
    """serial 기능 추가 후에도 serial=false는 기존 결과를 비트 단위로 재현한다."""
    cfg["tau"]["serial"] = False
    for (tau, seed), (xy, v) in GOLDEN.items():
        r = run_closed_loop(cfg, seed, tau=tau)
        assert r["landing_xy"].tolist() == xy
        assert r["landing_v"] == v


def test_serial_key_absent_equals_false(cfg: dict) -> None:
    """config에 tau.serial이 없으면 false와 동일하게 동작한다."""
    cfg_no = copy.deepcopy(cfg)
    cfg_no["tau"].pop("serial", None)
    cfg_off = copy.deepcopy(cfg)
    cfg_off["tau"]["serial"] = False
    r_no = run_closed_loop(cfg_no, 3, tau=1.0)
    r_off = run_closed_loop(cfg_off, 3, tau=1.0)
    assert r_no["landing_xy"].tolist() == r_off["landing_xy"].tolist()


def test_serial_true_drops_frames(cfg: dict) -> None:
    """카메라 1 Hz·τ=2.5 s 상수·serial=true: 밴드 내 측정 수가 병렬 대비 약 1/3."""
    assert float(cfg["camera"]["rate_hz"]) == 1.0
    cfg_par = copy.deepcopy(cfg)
    cfg_par["tau"]["serial"] = False
    cfg_ser = copy.deepcopy(cfg)
    cfg_ser["tau"]["serial"] = True
    r_par = run_closed_loop(cfg_par, 7, tau=2.5)
    r_ser = run_closed_loop(cfg_ser, 7, tau=2.5)
    assert r_par["n_dropped"] == 0
    assert r_par["n_meas"] > 0
    assert r_ser["n_dropped"] > 0
    ratio = r_ser["n_meas"] / r_par["n_meas"]
    assert 0.25 <= ratio <= 0.45, f"측정 수 비율 {ratio:.3f} (기대 ~1/3)"
    # 촬영 기회는 건너뛰거나 촬영하거나 둘 중 하나 (궤적 차이로 ±약간 허용)
    assert abs((r_ser["n_meas"] + r_ser["n_dropped"]) - r_par["n_meas"]) <= 5


def test_delta_v_consistent(cfg: dict) -> None:
    """delta_v_mps = Σ|a_T|·dt 가 반환된 a_cmd 적분과 일치, truth 모드에서도 존재."""
    dt = 1.0 / float(cfg["imu"]["rate_hz"])
    for measurement in ("stat", "truth"):
        r = run_closed_loop(cfg, 1, tau=0.5, measurement=measurement)
        a = np.asarray(r["a_cmd"])
        expected = float(np.sum(np.linalg.norm(a[:-1], axis=1)) * dt)
        assert r["delta_v_mps"] > 0.0
        assert abs(r["delta_v_mps"] - expected) < 1e-9 * max(expected, 1.0)


def test_fp_offset_calibrated(cfg: dict, tmp_path) -> None:
    """calibrated 모드 + fp_offset_med_m 존재 → 보정 오프셋 사용, 사용값 보고."""
    model = {"sigma_xyz_m": [50.0, 50.0, 30.0], "fp_offset_med_m": 381.43}
    path = tmp_path / "measurement_model.json"
    path.write_text(json.dumps(model), encoding="utf-8")

    cfg_cal = copy.deepcopy(cfg)
    cfg_cal["measurement"]["mode"] = "calibrated"
    cfg_cal["measurement"]["file"] = str(path)
    m = StatMeasurementModel(cfg_cal, np.random.default_rng(0))
    assert m.fp_offset == 381.43
    r = run_closed_loop(cfg_cal, 0, tau=0.5)
    assert r["fp_offset_used_m"] == 381.43

    # assumed 모드는 config 값(2000 m) 유지
    m2 = StatMeasurementModel(cfg, np.random.default_rng(0))
    assert m2.fp_offset == float(cfg["measurement"]["fp_offset_m"])
    r2 = run_closed_loop(cfg, 0, tau=0.5)
    assert r2["fp_offset_used_m"] == float(cfg["measurement"]["fp_offset_m"])

    # calibrated이지만 파일에 fp_offset_med_m이 없으면 config 값으로 폴백
    model_no = {"sigma_xyz_m": [50.0, 50.0, 30.0]}
    path_no = tmp_path / "no_offset.json"
    path_no.write_text(json.dumps(model_no), encoding="utf-8")
    cfg_no = copy.deepcopy(cfg_cal)
    cfg_no["measurement"]["file"] = str(path_no)
    m3 = StatMeasurementModel(cfg_no, np.random.default_rng(0))
    assert m3.fp_offset == float(cfg["measurement"]["fp_offset_m"])


def test_t_c_jitter(cfg: dict) -> None:
    """타임스탬프 지터: 0이면 비트 동일(rng 미소비), 보상 on에서만 결과를 바꾼다."""
    base = run_closed_loop(cfg, 5, tau=0.5)
    explicit0 = run_closed_loop(cfg, 5, tau=0.5, t_c_jitter=0.0)
    assert base["landing_xy"].tolist() == explicit0["landing_xy"].tolist()

    jit_on = run_closed_loop(cfg, 5, tau=0.5, t_c_jitter=0.3)
    assert jit_on["landing_xy"].tolist() != base["landing_xy"].tolist()

    # 미보상은 t_c를 쓰지 않으므로 지터가 rng 스트림조차 건드리지 않아야 한다
    off0 = run_closed_loop(cfg, 5, tau=0.5, delay_comp=False)
    off_j = run_closed_loop(cfg, 5, tau=0.5, delay_comp=False, t_c_jitter=0.3)
    assert off0["landing_xy"].tolist() == off_j["landing_xy"].tolist()


def test_camera_rate_override(cfg: dict) -> None:
    """카메라 레이트를 메모리에서 올리면 밴드 내 측정 수가 비례해서 는다."""
    cfg2 = copy.deepcopy(cfg)
    cfg2["camera"]["rate_hz"] = 2.0
    r1 = run_closed_loop(cfg, 11, tau=0.05)
    r2 = run_closed_loop(cfg2, 11, tau=0.05)
    assert r1["n_meas"] > 0
    ratio = r2["n_meas"] / r1["n_meas"]
    # 궤적·밴드 체류가 레이트에 따라 달라져 정확히 2는 아님 (11시드 실측 2.37)
    assert 1.6 <= ratio <= 2.6, f"측정 수 비율 {ratio:.2f} (기대 ~2)"


def test_bench_cpu_parsers() -> None:
    """bench_cpu의 CoreMark/Dhrystone 출력 파서."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "bench_cpu", Path("scripts/bench_cpu.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cm = mod.parse_coremark(
        "2K performance run parameters for coremark.\n"
        "CoreMark Size    : 666\n"
        "Total ticks      : 12000\n"
        "Total time (secs): 12.000000\n"
        "Iterations/Sec   : 33221.591000\n"
        "Iterations       : 400000\n"
        "Correct operation validated. See README.md for run and reporting rules.\n"
    )
    assert cm["iterations_per_sec"] == pytest.approx(33221.591)
    assert cm["iterations"] == 400000
    assert cm["validated"] is True
    assert mod.parse_coremark("garbage")["validated"] is False

    dh = mod.parse_dhrystone("Dhrystones per Second:      17570000.0\n")
    assert dh["dmips"] == pytest.approx(10000.0)
    assert mod.parse_dhrystone("nothing here") == {}


def test_tau_scaling_helpers() -> None:
    """make_tau_scaling의 CI 분리 판정·환산 τ 판정."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "make_tau_scaling", Path("scripts/make_tau_scaling.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.ci_separated([310.0, 337.0], [225.0, 255.0]) is True
    assert mod.ci_separated([144.0, 171.0], [152.0, 179.0]) is False
    assert mod.verdict(0.5, 1.0, 5.0) == "평탄 구간 — 보상 시 흡수"
    assert mod.verdict(2.0, 1.0, 5.0) == "드롭 영역 — 측정 손실로 악화"
    assert mod.verdict(20.0, 1.0, 5.0) == "격자 밖 — 성립 불가(외삽 금지)"
