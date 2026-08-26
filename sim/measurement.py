"""TRN 측정 모델: PnP 위치 측정 z = r + 잡음·오검출, 지연 τ 샘플러 (계약 §2.4)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def measurement_sigma(cfg: dict) -> tuple[np.ndarray, bool]:
    """축별 측정 σ와 assumed 여부를 반환.

    measurement.mode == "calibrated"이고 파일이 있으면 보정 통계를 로드,
    아니면 config 가정값(assumed=True → 그림 제목에 "assumed measurement stats").
    """
    m = cfg["measurement"]
    if m["mode"] == "calibrated":
        path = Path(m["file"])
        if path.exists():
            model = json.loads(path.read_text(encoding="utf-8"))
            return np.asarray(model["sigma_xyz_m"], dtype=float), False
    return np.asarray(m["sigma_xyz_m"], dtype=float), True


def measurement_R(cfg: dict) -> tuple[np.ndarray, bool]:
    """측정 잡음 공분산 R_meas = diag(σ_x², σ_y², σ_z²)와 assumed 여부."""
    sigma, assumed = measurement_sigma(cfg)
    return np.diag(sigma**2), assumed


class StatMeasurementModel:
    """참값 위치에서 통계 측정 z를 생성 (Tier 1/2용).

    z = r + N(0, diag σ²); 확률 fp_rate로 무작위 방향 fp_offset_m 이상치(오검출 모사).
    """

    def __init__(self, cfg: dict, rng: np.random.Generator, fp_rate: float | None = None) -> None:
        m = cfg["measurement"]
        self.rng = rng
        self.fp_rate = float(m["fp_rate"] if fp_rate is None else fp_rate)
        self.fp_offset = float(m["fp_offset_m"])
        self.sigma, self.assumed = measurement_sigma(cfg)

    def sample(self, r_true: np.ndarray) -> tuple[np.ndarray, bool]:
        """(z, valid) 반환. 통계 모델은 항상 valid."""
        if self.rng.random() < self.fp_rate:
            u = self.rng.normal(size=3)
            u /= np.linalg.norm(u)
            return r_true + self.fp_offset * u, True
        return r_true + self.rng.normal(0.0, self.sigma), True


class TauSampler:
    """추론 지연 τ 샘플러: constant | empirical(results/tau_*.json의 samples_s 리샘플)."""

    def __init__(self, cfg: dict, rng: np.random.Generator) -> None:
        t = cfg["tau"]
        self.rng = rng
        self.mode = t["mode"]
        if self.mode == "constant":
            self._const = float(t["constant_s"])
        elif self.mode == "empirical":
            data = json.loads(Path(t["empirical_file"]).read_text(encoding="utf-8"))
            self._samples = np.asarray(data["samples_s"], dtype=float)
            if self._samples.size == 0:
                raise ValueError(f"empirical τ 파일에 샘플이 없다: {t['empirical_file']}")
        else:
            raise ValueError(f"알 수 없는 tau.mode: {self.mode}")

    def sample(self) -> float:
        if self.mode == "constant":
            return self._const
        return float(self.rng.choice(self._samples))

    @property
    def max_tau(self) -> float:
        return self._const if self.mode == "constant" else float(self._samples.max())
