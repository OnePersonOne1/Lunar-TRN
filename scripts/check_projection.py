"""투영·정합 검증 CLI: Unity 렌더 위에 camera.py 투영 카탈로그 원 overlay + 중심 픽셀 오차 측정.

알려진 pose에서 렌더 → 카탈로그 원 overlay(figs/p4_projection_check.png) → 가장 큰 크레이터
5개의 예측 중심 vs 렌더 상 중심(Hough 원 검출) 픽셀 오차 → results/p4_projection_check.json.
5 px 이내면 합격.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.camera import K_cam, project  # noqa: E402
from sim.mc import result_meta  # noqa: E402
from unity.client import RenderClient  # noqa: E402

PASS_PX = 5.0
N_CRATERS = 5


def detect_circle_center(
    img_gray: np.ndarray, u: float, v: float, rad_px: float
) -> tuple[float, float] | None:
    """예측 위치 주변 창에서 Hough 원 검출 → 가장 가까운 원의 중심."""
    half = int(rad_px * 1.6)
    x0, y0 = int(u) - half, int(v) - half
    x1, y1 = int(u) + half, int(v) + half
    h, w = img_gray.shape
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    win = img_gray[y0:y1, x0:x1]
    if win.size == 0:
        return None
    win = cv2.GaussianBlur(win, (5, 5), 0)
    circles = cv2.HoughCircles(
        win, cv2.HOUGH_GRADIENT, dp=1, minDist=rad_px,
        param1=80, param2=25,
        minRadius=int(rad_px * 0.6), maxRadius=int(rad_px * 1.4),
    )
    if circles is None:
        return None
    cand = circles[0]
    d = np.hypot(cand[:, 0] + x0 - u, cand[:, 1] + y0 - v)
    k = int(d.argmin())
    return float(cand[k, 0] + x0), float(cand[k, 1] + y0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # CLI 규약 통일용
    ap.add_argument("--catalog", default="data/processed/catalog_L.csv")
    ap.add_argument("--out", default="results/p4_projection_check.json")
    ap.add_argument("--fig", default="figs/p4_projection_check.png")
    ap.add_argument("--pose", type=float, nargs=3, default=None,
                    help="카메라 위치 x y z [m] (기본: 원점 상공, trn_band 중간 고도)")
    ap.add_argument("--sun", type=float, nargs=2, default=[135.0, 30.0], help="sun_az sun_el [deg]")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cat = np.genfromtxt(args.catalog, delimiter=",", names=True)
    catalog = np.column_stack([cat["x"], cat["y"], cat["z"], cat["D"]])
    if args.pose is None:
        h_mid = 0.5 * (float(cfg["trn_band"]["h_min_m"]) + float(cfg["trn_band"]["h_max_m"]))
        r = np.array([0.0, 0.0, h_mid])
    else:
        r = np.array(args.pose)

    client = RenderClient(cfg)
    img = client.render(r, args.sun[0], args.sun[1], frame_id=0)
    client.close()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    W, H = float(cfg["camera"]["W"]), float(cfg["camera"]["H"])
    f = K_cam(cfg)[0, 0]
    uv, z_C, valid = project(catalog[:, :3], r, cfg)
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

    overlay = img.copy()
    for i in np.flatnonzero(inside):
        rad = f * catalog[i, 3] / z_C[i] / 2.0
        cv2.circle(overlay, (int(uv[i, 0]), int(uv[i, 1])), int(rad), (0, 255, 0), 2)

    order = np.argsort(-catalog[:, 3])
    checks = []
    for i in order:
        if not inside[i] or len(checks) >= N_CRATERS:
            continue
        rad = f * catalog[i, 3] / z_C[i] / 2.0
        found = detect_circle_center(gray, uv[i, 0], uv[i, 1], rad)
        err = None if found is None else float(np.hypot(found[0] - uv[i, 0], found[1] - uv[i, 1]))
        checks.append({
            "catalog_id": int(cat["id"][i]) if "id" in (cat.dtype.names or ()) else int(i),
            "D_m": float(catalog[i, 3]),
            "predicted_uv": [float(uv[i, 0]), float(uv[i, 1])],
            "detected_uv": None if found is None else [found[0], found[1]],
            "error_px": err,
        })
        if found is not None:
            cv2.drawMarker(overlay, (int(found[0]), int(found[1])), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)

    errs = [c["error_px"] for c in checks if c["error_px"] is not None]
    passed = bool(errs) and all(e <= PASS_PX for e in errs)
    out = {
        "meta": result_meta(args.config),
        "pose": r.tolist(),
        "sun_az_el_deg": args.sun,
        "checks": checks,
        "mean_error_px": float(np.mean(errs)) if errs else None,
        "passed_5px": passed,
        "note": "5 px 초과 시 의심 순서: 텍스처 상하/좌우 반전, 고도 스케일, 지형 오프셋, RAW 바이트 오더",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(fig_path), overlay)
    print(json.dumps({"mean_error_px": out["mean_error_px"], "passed_5px": passed}, indent=2))
    print(f"fig: {fig_path}\nsaved: {out_path}")


if __name__ == "__main__":
    main()
