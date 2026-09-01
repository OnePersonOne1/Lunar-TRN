"""몬테카를로 배치 실행: CEP(반경 오차 중앙값), 95% 오차 타원, 부트스트랩 CI, results meta 헬퍼."""
from __future__ import annotations

import hashlib
import math
import platform
import subprocess
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

from sim.loop import run_closed_loop


def _one_run(args: tuple) -> tuple[list[float], float, dict]:
    cfg, seed, loop_kwargs = args
    res = run_closed_loop(cfg, seed, **loop_kwargs)
    gl = res["gate_log"]
    n_accept = sum(e["accepted"] for e in gl)
    extras = {
        "n_meas": res["n_meas"],
        "n_dropped": res["n_dropped"],
        "n_accept": n_accept,
        "gate_accept": (n_accept / len(gl)) if gl else float("nan"),
        "delta_v_mps": res["delta_v_mps"],
        "fp_offset_used_m": res["fp_offset_used_m"],
    }
    return res["landing_xy"].tolist(), res["landing_v"], extras


def run_mc(
    cfg: dict, n_runs: int, workers: int, seed0: int = 0, **loop_kwargs
) -> tuple[np.ndarray, np.ndarray, dict]:
    """seed = seed0..seed0+n_runs-1 병렬 실행.

    반환: (landing_xy (n,2), landing_v (n,), extras) — extras는 run별 배열
    {n_meas, n_dropped, gate_accept, delta_v_mps}와 fp_offset_used_m(스칼라).
    """
    tasks = [(cfg, seed0 + i, loop_kwargs) for i in range(n_runs)]
    if workers <= 1:
        out = [_one_run(t) for t in tasks]
    else:
        with Pool(processes=workers) as pool:
            out = pool.map(_one_run, tasks)
    xy = np.asarray([o[0] for o in out], dtype=float)
    v = np.asarray([o[1] for o in out], dtype=float)
    extras = {
        key: np.asarray([o[2][key] for o in out], dtype=float)
        for key in ("n_meas", "n_dropped", "n_accept", "gate_accept", "delta_v_mps")
    }
    extras["fp_offset_used_m"] = out[0][2]["fp_offset_used_m"] if out else None
    return xy, v, extras


def extras_summary(extras: dict) -> dict:
    """run_mc extras → 조건별 요약(평균 측정 수·드롭 수·게이트 수락률·ΔV 통계)."""
    dv = extras["delta_v_mps"]
    return {
        "mean_n_meas": float(np.mean(extras["n_meas"])),
        "mean_n_dropped": float(np.mean(extras["n_dropped"])),
        "mean_n_accept": float(np.mean(extras["n_accept"])),
        "mean_gate_accept": float(np.nanmean(extras["gate_accept"])),
        "delta_v_mps": {
            "mean": float(np.mean(dv)),
            "p50": float(np.percentile(dv, 50)),
            "p95": float(np.percentile(dv, 95)),
        },
        "fp_offset_used_m": extras["fp_offset_used_m"],
    }


def cep(landing_xy: np.ndarray) -> float:
    """CEP: 목표점(원점) 기준 반경 오차의 중앙값."""
    return float(np.median(np.linalg.norm(landing_xy, axis=1)))


def error_ellipse_95(landing_xy: np.ndarray) -> dict:
    """표본 공분산 고유분해 기반 95% 오차 타원 (중심, 반축 [장, 단], 장축 각도 rad)."""
    center = landing_xy.mean(axis=0)
    cov = np.cov(landing_xy.T)
    eigval, eigvec = np.linalg.eigh(cov)  # 오름차순
    k = stats.chi2.ppf(0.95, 2)
    semi = np.sqrt(np.maximum(eigval, 0.0) * k)
    major = eigvec[:, 1]
    return {
        "center": center.tolist(),
        "semi_axes_m": [float(semi[1]), float(semi[0])],
        "angle_rad": float(math.atan2(major[1], major[0])),
    }


def bootstrap_cep_ci(
    landing_xy: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """CEP의 부트스트랩 95% CI."""
    n = len(landing_xy)
    ceps = np.array(
        [cep(landing_xy[rng.integers(0, n, n)]) for _ in range(n_boot)]
    )
    return float(np.percentile(ceps, 2.5)), float(np.percentile(ceps, 97.5))


def result_meta(config_path: str | Path) -> dict:
    """모든 results json에 넣는 meta: git_hash, config_hash, timestamp, hardware."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        git_hash = None
    return {
        "git_hash": git_hash,
        "config_hash": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
    }
