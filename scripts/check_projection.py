"""투영·정합 검증 CLI: Unity 렌더 위에 camera.py 투영 카탈로그 원 overlay + 중심 픽셀 오차 측정.

알려진 pose에서 렌더 → 카탈로그 원 overlay(figs/p4_projection_check.png) → 가장 큰 크레이터
5개의 예측 중심 픽셀 오차 → results/p4_projection_check.json. 5 px 이내면 합격.

오차 측정: WAC 텍스처를 같은 pose의 평면 근사 카메라 뷰로 리샘플한 기준 이미지에서
크레이터 주변 패치를 떼어 렌더에 템플릿 매칭(NCC, 고역 통과로 조명 성분 제거).
렌더의 알베도가 그 텍스처에서 왔으므로, 이 매칭 오차는 순수하게 기하 정합
(지형 배치·카메라 모델·이미지 방향) 오차를 측정한다. (Hough 원 검출은 저대비
크레이터에서 불안정해 교체함.)
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
SEARCH_PX = 40      # 예측 위치 주변 탐색 반경
NCC_MIN = 0.3       # 이 미만이면 매칭 실패로 처리
HIGHPASS_SIGMA = 25.0


def highpass(a: np.ndarray) -> np.ndarray:
    """조명(저주파) 성분 제거 + 정규화 — 텍스처 알베도 무늬만 남긴다."""
    hp = a.astype(np.float32) - cv2.GaussianBlur(a.astype(np.float32), (0, 0), HIGHPASS_SIGMA)
    return (hp - hp.mean()) / (hp.std() + 1e-9)


def ref_patch(
    cfg: dict, r: np.ndarray, tex: np.ndarray, x_min: float, y_max: float, dx_m: float,
    ui: int, vi: int, half: int, z_plane: float,
) -> np.ndarray:
    """예측 픽셀 (ui, vi) 주변 half 창을, 고도 z_plane 평면 가정으로 텍스처에서 리샘플.

    z_plane에 크레이터 자신의 카탈로그 고도를 넣으면 그 크레이터의 텍스처 무늬가
    3-D 투영 위치와 일치한다(단일 전역 평면을 쓰면 가장자리에서 시차 오차 발생).
    """
    f = K_cam(cfg)[0, 0]
    W, H = float(cfg["camera"]["W"]), float(cfg["camera"]["H"])
    h_eff = float(r[2]) - z_plane
    uu, vv = np.meshgrid(np.arange(ui - half, ui + half, dtype=np.float32),
                         np.arange(vi - half, vi + half, dtype=np.float32))
    x_g = r[0] + (uu - W / 2.0) * h_eff / f
    y_g = r[1] - (vv - H / 2.0) * h_eff / f
    return cv2.remap(tex, ((x_g - x_min) / dx_m).astype(np.float32),
                     ((y_max - y_g) / dx_m).astype(np.float32), cv2.INTER_LINEAR)


def detect_by_template(
    render_hp: np.ndarray, tpl_patch: np.ndarray, u: float, v: float
) -> tuple[float, float, float] | None:
    """크레이터 텍스처 패치를 렌더의 예측 위치 주변에 NCC 매칭 → (u, v, ncc)."""
    h, w = render_hp.shape
    t_half = tpl_patch.shape[0] // 2
    s_half = t_half + SEARCH_PX
    ui, vi = int(round(u)), int(round(v))
    if not (s_half <= ui < w - s_half and s_half <= vi < h - s_half):
        return None
    win = render_hp[vi - s_half:vi + s_half, ui - s_half:ui + s_half]
    res = cv2.matchTemplate(win, tpl_patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < NCC_MIN:
        return None
    du = loc[0] - SEARCH_PX
    dv = loc[1] - SEARCH_PX
    return float(u + du), float(v + dv), float(score)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # CLI 규약 통일용
    ap.add_argument("--catalog", default="data/processed/catalog_L.csv")
    ap.add_argument("--texture", default="data/processed/texture_L.png")
    ap.add_argument("--dem", default="data/processed/dem_L.npz")
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
    render_hp = highpass(gray)
    tex = cv2.imread(args.texture, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    dem = np.load(args.dem)
    dx_m = float(dem["x"][1] - dem["x"][0])
    x_min, y_max = float(dem["x"][0]), float(dem["y"].max())

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
        t_half = int(max(rad * 1.5, 24))
        s_half = t_half + SEARCH_PX
        ui, vi = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if not (s_half <= ui < W - s_half and s_half <= vi < H - s_half):
            continue  # 탐색 창이 화면 밖 — 다음으로 큰 크레이터로
        pad = 32  # 고역 통과 컨텍스트
        patch = ref_patch(cfg, r, tex, x_min, y_max, dx_m, ui, vi, t_half + pad,
                          z_plane=float(catalog[i, 2]))
        tpl = highpass(patch)[pad:-pad, pad:-pad]
        found = detect_by_template(render_hp, tpl, uv[i, 0], uv[i, 1])
        err = None if found is None else float(np.hypot(found[0] - uv[i, 0], found[1] - uv[i, 1]))
        checks.append({
            "catalog_id": int(cat["id"][i]) if "id" in (cat.dtype.names or ()) else int(i),
            "D_m": float(catalog[i, 3]),
            "predicted_uv": [float(uv[i, 0]), float(uv[i, 1])],
            "detected_uv": None if found is None else [found[0], found[1]],
            "ncc": None if found is None else found[2],
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
