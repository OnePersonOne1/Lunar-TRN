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
    model = _calibrated_model(cfg)
    if model is not None:
        return np.asarray(model["sigma_xyz_m"], dtype=float), False
    return np.asarray(cfg["measurement"]["sigma_xyz_m"], dtype=float), True


def _calibrated_model(cfg: dict) -> dict | None:
    """measurement.mode == "calibrated"이고 파일이 있으면 보정 모델 dict, 아니면 None."""
    m = cfg["measurement"]
    if m["mode"] == "calibrated":
        path = Path(m["file"])
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def measurement_R(cfg: dict) -> tuple[np.ndarray, bool]:
    """측정 잡음 공분산 R_meas = diag(σ_x², σ_y², σ_z²)와 assumed 여부."""
    sigma, assumed = measurement_sigma(cfg)
    return np.diag(sigma**2), assumed


class StatMeasurementModel:
    """참값 위치에서 통계 측정 z를 생성 (Tier 1/2용).

    z = r + N(0, diag σ²); 확률 fp_rate로 무작위 방향 fp_offset_m 이상치(오검출 모사).
    """

    def __init__(
        self,
        cfg: dict,
        rng: np.random.Generator,
        fp_rate: float | None = None,
        fp_offset: float | None = None,
    ) -> None:
        m = cfg["measurement"]
        self.rng = rng
        self.fp_rate = float(m["fp_rate"] if fp_rate is None else fp_rate)
        # 오프셋 우선순위: 명시 인자 > calibrated 파일(fp_offset_med_m) > config 가정값
        model = _calibrated_model(cfg)
        if fp_offset is not None:
            self.fp_offset = float(fp_offset)
        elif model is not None and model.get("fp_offset_med_m"):
            self.fp_offset = float(model["fp_offset_med_m"])
        else:
            self.fp_offset = float(m["fp_offset_m"])
        self.sigma, self.assumed = measurement_sigma(cfg)

    def sample(self, r_true: np.ndarray) -> tuple[np.ndarray, bool]:
        """(z, valid) 반환. 통계 모델은 항상 valid."""
        if self.rng.random() < self.fp_rate:
            u = self.rng.normal(size=3)
            u /= np.linalg.norm(u)
            return r_true + self.fp_offset * u, True
        return r_true + self.rng.normal(0.0, self.sigma), True


class UnityMeasurementModel:
    """Unity-in-the-loop 측정 (P6): 렌더 → 탐지 → 연관(EKF 예측 pose) → PnP → z.

    τ_wallclock(탐지+연관+PnP 실측 시간)을 함께 반환한다. frames_dir가 있으면
    렌더+탐지 박스+투영 카탈로그 overlay PNG를 저장한다(영상용).
    """

    def __init__(
        self,
        cfg: dict,
        rng: np.random.Generator,
        detector_path: str,
        catalog_path: str = "data/processed/catalog_L.csv",
        frames_dir: str | None = None,
    ) -> None:
        from perception.detect import Detector
        from unity.client import RenderClient

        self.cfg = cfg
        self.client = RenderClient(cfg)
        self.detector = Detector(detector_path, cfg)
        cat = np.genfromtxt(catalog_path, delimiter=",", names=True)
        self.catalog = np.column_stack([cat["x"], cat["y"], cat["z"], cat["D"]])
        ds = cfg["dataset"]
        self.sun_az = float(rng.uniform(*[float(v) for v in ds["sun_az_deg"]]))
        self.sun_el = float(rng.uniform(*[float(v) for v in ds["sun_el_deg"]]))
        self.frames_dir = Path(frames_dir) if frames_dir else None
        if self.frames_dir:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.assumed = False  # 실측정 경로

    def sample_frame(
        self, r_true: np.ndarray, r_pred: np.ndarray, frame_id: int, t: float
    ) -> dict:
        """한 프레임 측정. 반환: z, valid, tau_wallclock_s, n_det, n_match, n_inliers,
        pnp_err_m(참값 대비), reproj_err_px."""
        import time as _time

        from perception.associate import associate
        from perception.pnp import solve_pnp

        img = self.client.render(r_true, self.sun_az, self.sun_el, frame_id=frame_id, t=t)
        t0 = _time.perf_counter()
        boxes = self.detector.detect(img)  # (n, 6) [x0, y0, x1, y1, conf, cls]
        centers = (
            np.empty((0, 2))
            if len(boxes) == 0
            else np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0,
                                  (boxes[:, 1] + boxes[:, 3]) / 2.0])
        )
        pairs = associate(centers, r_pred, self.catalog, self.cfg)
        res = {"r_PnP": None, "valid": False, "n_inliers": 0, "reproj_err_px": None}
        if len(pairs) >= 4:
            pts_L = self.catalog[[c for _, c in pairs], :3]
            uv = centers[[d for d, _ in pairs]]
            res = solve_pnp(pts_L, uv, self.cfg)
        tau_wall = _time.perf_counter() - t0

        if self.frames_dir is not None:
            self._save_overlay(img, boxes, centers, r_pred, frame_id, pairs)
        z = res["r_PnP"]
        return {
            "z": None if z is None else np.asarray(z, dtype=float),
            "valid": bool(res["valid"]),
            "tau_wallclock_s": tau_wall,
            "n_det": int(len(centers)),
            "n_match": int(len(pairs)),
            "n_inliers": int(res["n_inliers"]),
            "pnp_err_m": None if z is None else float(np.linalg.norm(z - r_true)),
            "reproj_err_px": res["reproj_err_px"],
        }

    def _save_overlay(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
        centers: np.ndarray,
        r_pred: np.ndarray,
        frame_id: int,
        pairs: list,
    ) -> None:
        """합본 한 장: 연관(associate) 결과로 색 구분.

        초록 원 = 카탈로그·탐지 매칭 / 보라 원 = 카탈로그 미탐지(missed) /
        주황 십자+conf = 매칭 탐지 / 보라 십자+conf = 비매칭 탐지(FP 또는 D_min 미만).
        """
        import cv2

        from perception.camera import K_cam, project

        green, orange, purple = (0, 255, 0), (0, 160, 255), (230, 60, 200)
        matched_det = {d for d, _ in pairs}
        matched_cat = {c for _, c in pairs}

        canvas = img.copy()
        h_img, w_img = img.shape[:2]
        f = K_cam(self.cfg)[0, 0]
        uv, z_C, valid = project(self.catalog[:, :3], r_pred, self.cfg)
        n_miss = 0
        for i in np.flatnonzero(valid):
            on_screen = 0 <= uv[i, 0] < w_img and 0 <= uv[i, 1] < h_img
            if not on_screen and i not in matched_cat:
                continue  # 화면 밖 미매칭은 매칭 후보가 아님 — 잡음 줄이기
            rad = f * self.catalog[i, 3] / z_C[i] / 2.0
            if i not in matched_cat:
                n_miss += 1
            cv2.circle(canvas, (int(uv[i, 0]), int(uv[i, 1])), int(rad),
                       green if i in matched_cat else purple, 2)
        for j, (u, v) in enumerate(centers):
            c = orange if j in matched_det else purple
            cv2.drawMarker(canvas, (int(u), int(v)), c, cv2.MARKER_CROSS, 16, 2)
            cv2.putText(canvas, f"{boxes[j, 4]:.2f}", (int(u) + 6, int(v) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(canvas, f"{boxes[j, 4]:.2f}", (int(u) + 6, int(v) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

        def label(text: str, y: int, color: tuple) -> None:
            cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        n_fp = len(centers) - len(matched_det)
        label(f"MATCHED {len(pairs)}", 34, green)
        label(f"MISSED {n_miss}", 68, purple)
        label(f"UNMATCHED DET {n_fp}", 102, purple)
        cv2.imwrite(str(self.frames_dir / f"{frame_id:05d}.png"), canvas)

    def close(self) -> None:
        self.client.close()


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
