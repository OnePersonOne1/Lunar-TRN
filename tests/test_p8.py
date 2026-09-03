"""P8 Unity 실런 MC 러너의 순수 로직 테스트 (Unity 불필요: 체크포인트·집계만)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_mc_unity import aggregate, dedupe_meas, load_done_seeds  # noqa: E402
from sim.measurement import load_registration_correction, registration_bias  # noqa: E402


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    return {"mc": {"bootstrap_n": 20, "crash_error_m": 5000.0}, "bench": {"cpu_threads": 1}}


def _run_rec(seed: int, x: float, y: float) -> dict:
    return {
        "seed": seed, "landing_xy_m": [x, y],
        "landing_error_m": float(np.hypot(x, y)), "landing_v_mps": 1.0,
        "n_meas": 99, "n_dropped": 0, "gate_accept": 0.9,
        "tau_wallclock_median_s": 0.2, "tau_wallclock_p95_s": 0.21,
        "tau_wallclock_max_s": 0.22, "wall_s": 30.0,
    }


def test_load_done_seeds(tmp_path: Path) -> None:
    ckpt = tmp_path / "runs.jsonl"
    assert load_done_seeds(ckpt) == set()  # 파일 없음 = 빈 집합
    with ckpt.open("w", encoding="utf-8") as fh:
        for s in (0, 2):
            fh.write(json.dumps(_run_rec(s, 1.0, 0.0)) + "\n")
        fh.write("\n")  # 빈 줄은 무시
    assert load_done_seeds(ckpt) == {0, 2}


def test_dedupe_meas_keeps_last() -> None:
    rows = [
        {"seed": 0, "t_c": 1.0, "tau_wallclock_s": 0.1},
        {"seed": 0, "t_c": 2.0, "tau_wallclock_s": 0.2},
        {"seed": 0, "t_c": 1.0, "tau_wallclock_s": 0.3},  # 재실행 중복 — 마지막 우선
        {"seed": 1, "t_c": 1.0, "tau_wallclock_s": 0.4},
    ]
    out = dedupe_meas(rows)
    assert len(out) == 3
    assert [r["tau_wallclock_s"] for r in out if r["seed"] == 0 and r["t_c"] == 1.0] == [0.3]


def test_registration_correction(tmp_path: Path) -> None:
    corr = {
        "kind": "east_bins",
        "table": [
            {"east_lo_m": -100e3, "east_hi_m": -90e3, "bias_xyz_m": [10.0, -5.0, 2.0]},
            {"east_lo_m": -90e3, "east_hi_m": -80e3, "bias_xyz_m": [30.0, 5.0, -2.0]},
        ],
    }
    p = tmp_path / "corr.json"
    p.write_text(json.dumps(corr), encoding="utf-8")
    c = load_registration_correction(p)
    assert np.allclose(registration_bias(-95e3, c), [10.0, -5.0, 2.0])   # bin 내부
    assert np.allclose(registration_bias(-85e3, c), [30.0, 5.0, -2.0])
    assert np.allclose(registration_bias(-200e3, c), [10.0, -5.0, 2.0])  # 왼쪽 클램프
    assert np.allclose(registration_bias(0.0, c), [30.0, 5.0, -2.0])     # 오른쪽 클램프
    assert load_registration_correction(tmp_path / "none.json") is None


def test_aggregate(cfg: dict, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mc:\n  bootstrap_n: 20\n", encoding="utf-8")
    # 반경 10..80의 8런 → CEP = 반경 중앙값 45
    runs = [_run_rec(s, 10.0 * (s + 1), 0.0) for s in range(8)]
    meas = [
        {"seed": s, "t_c": float(t), "tau_wallclock_s": 0.1 + 0.01 * s}
        for s in range(8) for t in range(3)
    ]
    # 발산 런 2개 추가: NaN 착륙점 + 유한하지만 crash_error_m 초과 추락 — 둘 다 분리돼야 한다
    div = _run_rec(8, float("nan"), float("nan"))
    div["landing_error_m"] = float("nan")
    div["landing_v_mps"] = float("nan")
    runs.append(div)
    runs.append(_run_rec(9, -27000.0, 6000.0))  # 연관 실패 추락(유한값)
    out = aggregate(cfg, runs, meas, params={"tau": "wallclock"},
                    failed_seeds=[99], config_path=config_path)
    assert out["cep_m"] == pytest.approx(45.0)  # 착륙 8런만으로 계산
    assert out["params"]["n_runs"] == 10
    assert out["n_landed"] == 8
    assert out["n_diverged"] == 2 and out["diverged_seeds"] == [8, 9]
    assert np.isfinite(out["landing_v_mean_mps"])
    assert out["failed_seeds"] == [99]
    assert out["r95_m"] == pytest.approx(np.percentile(np.arange(10.0, 90.0, 10.0), 95))
    assert out["r95_over_cep"] == pytest.approx(out["r95_m"] / 45.0)
    assert out["cep_ci95_m"][0] <= 45.0 <= out["cep_ci95_m"][1]
    assert out["tau_wallclock_s"]["n_frames"] == 24
    assert out["mean_gate_accept"] == pytest.approx(0.9)
    assert len(out["landing_xy_m"]) == 8
    assert "ellipse95" in out and "meta" in out
