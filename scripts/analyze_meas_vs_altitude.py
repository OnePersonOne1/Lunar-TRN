"""고도별 측정 오차 분석 CLI: 데이터셋 전 프레임에서 연관→PnP 오차의 고도 의존성.

trn_band 하한(h_min) 결정 근거 산출: 고도 bin별 로버스트 σ(0 기준 MAD), 유효율,
매칭 크레이터 수 → results/p5_sigma_vs_altitude.json + figs/p5_sigma_vs_altitude.png.
탐지기는 INT8(ORT CPU) — 폐루프에서 쓰는 것과 동일 산출물.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="runs/export/crater_int8_ort.onnx")
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--catalog", default="data/processed/catalog_L.csv")
    ap.add_argument("--bin-km", type=float, default=1.0)
    ap.add_argument("--out", default="results/p5_sigma_vs_altitude.json")
    ap.add_argument("--fig", default="figs/p5_sigma_vs_altitude.png")
    args = ap.parse_args()

    import cv2

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    rng = np.random.default_rng(args.seed)
    det = Detector(args.model, cfg)
    cat = np.genfromtxt(args.catalog, delimiter=",", names=True)
    catalog = np.column_stack([cat["x"], cat["y"], cat["z"], cat["D"]])
    dataset = Path(args.dataset)
    poses = np.genfromtxt(dataset / "poses.csv", delimiter=",", names=True, dtype=None,
                          encoding="utf-8")
    x0_err = np.asarray([float(v) for v in cfg["ekf"]["x0_error"]][:3])

    rows = []  # [h, ex, ey, ez, n_match] (유효 PnP만)
    n_total = n_invalid = 0
    for row in poses:
        stem = f"traj{int(row['traj_id'])}_{int(row['frame_id']):05d}"
        img = cv2.imread(str(dataset / "images" / str(row["split"]) / f"{stem}.png"))
        if img is None:
            continue
        n_total += 1
        r_true = np.array([row["x"], row["y"], row["z"]], dtype=float)
        r_pred = r_true + rng.normal(0.0, x0_err)
        centers = det.centers(img)
        pairs = associate(centers, r_pred, catalog, cfg)
        if len(pairs) < 4:
            n_invalid += 1
            continue
        res = solve_pnp(catalog[[c for _, c in pairs], :3], centers[[d for d, _ in pairs]], cfg)
        if not res["valid"]:
            n_invalid += 1
            continue
        e = res["r_PnP"] - r_true
        rows.append([r_true[2], e[0], e[1], e[2], len(pairs)])
        if n_total % 100 == 0:
            print(f"{n_total} frames...", flush=True)

    a = np.asarray(rows)
    h_km = a[:, 0] / 1e3
    lo = np.floor(h_km.min() / args.bin_km) * args.bin_km
    hi = np.ceil(h_km.max() / args.bin_km) * args.bin_km
    edges = np.arange(lo, hi + 1e-9, args.bin_km)
    bins = []
    for i in range(len(edges) - 1):
        m = (h_km >= edges[i]) & (h_km < edges[i + 1])
        if m.sum() < 5:
            continue
        e = a[m, 1:4]
        sigma = 1.4826 * np.median(np.abs(e), axis=0)  # 0 기준 로버스트 σ
        bins.append({
            "h_lo_km": float(edges[i]), "h_hi_km": float(edges[i + 1]),
            "n_frames": int(m.sum()),
            "sigma_xyz_m": [round(float(v), 1) for v in sigma],
            "sigma_horiz_m": round(float(np.hypot(sigma[0], sigma[1])), 1),
            "p90_norm_m": round(float(np.percentile(np.linalg.norm(e, axis=1), 90)), 1),
            "n_match_med": float(np.median(a[m, 4])),
        })

    out = {
        "meta": result_meta(args.config),
        "model": args.model,
        "n_frames_total": n_total,
        "n_valid": len(rows),
        "valid_ratio": round(len(rows) / max(n_total, 1), 4),
        "note": "고도 bin별 연관→PnP 오차 (0 기준 로버스트 σ). 태양각은 프레임별 무작위(데이터셋).",
        "bins": bins,
        "frames": [[round(float(v), 2) for v in r] for r in rows],  # [h_m, ex, ey, ez, n_match]
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hc = [0.5 * (b["h_lo_km"] + b["h_hi_km"]) for b in bins]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (lab, c) in enumerate([("sigma E", "tab:blue"), ("sigma N", "tab:orange"),
                                  ("sigma U", "tab:green")]):
        ax.plot(hc, [b["sigma_xyz_m"][i] for b in bins], "o-", color=c, label=lab)
    ax.plot(hc, [b["p90_norm_m"] for b in bins], "s--", color="tab:red", alpha=0.6,
            label="p90 |error|")
    for h_cand, ls in ((19.0, ":"), (22.0, "--")):
        ax.axvline(h_cand, color="gray", linestyle=ls, alpha=0.8)
        ax.text(h_cand, ax.get_ylim()[1], f" h_min {h_cand:g} km", rotation=90,
                va="top", fontsize=8, color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("altitude [km]")
    ax.set_ylabel("PnP measurement error [m]")
    ax2 = ax.twinx()
    ax2.bar(hc, [b["n_match_med"] for b in bins], width=0.6, alpha=0.15, color="tab:purple")
    ax2.set_ylabel("median matched craters", color="tab:purple")
    ax.set_title("Measurement error vs altitude (INT8 detector, per-frame random sun)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=150)
    print(f"fig: {args.fig}")


if __name__ == "__main__":
    main()
