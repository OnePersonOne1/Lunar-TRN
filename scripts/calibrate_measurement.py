"""측정 통계 보정 CLI: val 프레임에서 연관→PnP 오차 통계 → results/measurement_model.json.

참값 pose + 예측 오차 모사(ekf.x0_error 규모 무작위 섭동)로 연관을 돌려 실제 운용 조건을 흉내낸다.
sim/measurement.py의 calibrated 모드가 읽는 스키마: {"sigma_xyz_m": [...], ...}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.associate import associate  # noqa: E402
from perception.detect import Detector  # noqa: E402
from perception.pnp import solve_pnp  # noqa: E402
from sim.mc import result_meta  # noqa: E402


def calibrate(model_path: str, cfg: dict, dataset: Path, seed: int) -> dict:
    import cv2

    rng = np.random.default_rng(seed)
    det = Detector(model_path, cfg)
    cat = np.genfromtxt("data/processed/catalog_L.csv", delimiter=",", names=True)
    catalog = np.column_stack([cat["x"], cat["y"], cat["z"], cat["D"]])
    poses = np.genfromtxt(dataset / "poses.csv", delimiter=",", names=True, dtype=None,
                          encoding="utf-8")
    x0_err = np.asarray([float(v) for v in cfg["ekf"]["x0_error"]][:3])

    h_min = float(cfg["trn_band"]["h_min_m"])
    h_max = float(cfg["trn_band"]["h_max_m"])
    errs, n_valid, n_total = [], 0, 0
    for row in poses:
        if str(row["split"]) != "val":
            continue
        if not (h_min <= float(row["z"]) <= h_max):
            continue  # 루프가 측정을 만드는 고도 구간(trn_band)만 보정에 사용
        stem = f"traj{int(row['traj_id'])}_{int(row['frame_id']):05d}"
        img_path = dataset / "images" / "val" / f"{stem}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        n_total += 1
        r_true = np.array([row["x"], row["y"], row["z"]], dtype=float)
        r_pred = r_true + rng.normal(0.0, x0_err)  # 예측 오차 모사
        centers = det.centers(img)
        pairs = associate(centers, r_pred, catalog, cfg)
        if len(pairs) < 4:
            continue
        pts_L = catalog[[c for _, c in pairs], :3]
        uv = centers[[d for d, _ in pairs]]
        res = solve_pnp(pts_L, uv, cfg)
        if not res["valid"]:
            continue
        n_valid += 1
        errs.append(res["r_PnP"] - r_true)

    errs = np.asarray(errs)
    if len(errs) < 10:
        return {"error": f"유효 PnP 측정이 {len(errs)}개뿐 — 보정 불가", "n_frames": n_total}
    # EKF의 R은 정상 측정의 잡음이어야 한다. 소수의 대형 실패(오연관 PnP 합의)는
    # 루프의 χ² 게이트가 기각하는 대상이므로, σ는 로버스트(MAD 기반 3σ 클리핑)로 추정하고
    # 대형 실패는 fp_rate(발생률)·fp_offset(크기)으로 따로 보고한다.
    # σ는 0 기준으로 추정한다: EKF에 bias 상태가 없어(TODO(oct)) 비행 구간·태양각에 따라
    # 이동하는 계통 편향이 혁신에 그대로 남으므로, R가 편향까지 덮지 않으면
    # χ² 게이트가 정상 측정을 연쇄 기각한다(P6 진단에서 관측).
    sigma_mad = 1.4826 * np.median(np.abs(errs), axis=0)          # 0 기준 MAD
    inlier = np.all(np.abs(errs) <= 3.0 * np.maximum(sigma_mad, 1e-9), axis=1)
    sigma = np.sqrt(np.mean(errs[inlier] ** 2, axis=0))           # 0 기준 RMS (클리핑)
    bias = errs[inlier].mean(axis=0)
    out_norm = np.linalg.norm(errs[~inlier], axis=1)
    return {
        "sigma_xyz_m": sigma.tolist(),          # sim이 읽는 값 (클리핑 std)
        "bias_xyz_m": bias.tolist(),
        "sigma_mad_xyz_m": sigma_mad.tolist(),
        "sigma_raw_std_xyz_m": errs.std(axis=0).tolist(),
        "fp_rate_est": float((~inlier).mean()),     # 게이트 대상 대형 실패 발생률
        "fp_offset_med_m": float(np.median(out_norm)) if len(out_norm) else None,
        "valid_ratio": n_valid / max(n_total, 1),
        "n_frames": n_total,
        "n_valid": n_valid,
        "n_inlier": int(inlier.sum()),
        "errors_xyz_m": errs.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp32", default="runs/train/crater/weights/best.pt")
    ap.add_argument("--int8", default="runs/export/crater_int8_ort.onnx")
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--out", default="results/measurement_model.json")
    ap.add_argument("--fig", default="figs/p5_pnp_error_hist.png")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    dataset = Path(args.dataset)

    results = {}
    for key, path in (("fp32", args.fp32), ("int8", args.int8)):
        r = calibrate(path, cfg, dataset, args.seed)
        r["model"] = path
        results[key] = r
        print(f"{key}: {json.dumps({k: v for k, v in r.items() if k != 'errors_xyz_m'}, ensure_ascii=False)}")

    # measurement_model.json은 INT8(실제 온보드 조건) 기준. fp32는 비교용으로 같은 파일에 포함.
    primary = results["int8"] if "sigma_xyz_m" in results["int8"] else results["fp32"]
    out = {"meta": result_meta(args.config), **primary, "by_precision": {
        k: {kk: vv for kk, vv in v.items() if kk != "errors_xyz_m"} for k, v in results.items()
    }}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for key, r in results.items():
        if "errors_xyz_m" not in r:
            continue
        e = np.asarray(r["errors_xyz_m"])
        for i, ax in enumerate(axes):
            ax.hist(e[:, i], bins=30, alpha=0.6, label=key)
    for i, (ax, lab) in enumerate(zip(axes, ["E", "N", "U"])):
        ax.set_xlabel(f"PnP error {lab} [m]")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("PnP position error (val frames, simulated prediction error)")
    fig.tight_layout()
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"fig: {fig_path}")


if __name__ == "__main__":
    main()
