"""씬-카탈로그 정합 진단/보정 CLI (P8b): 회랑을 따라 측정 오차 벡터 b(r)을 실측.

P8 실런 MC에서 착륙이 공통 방향으로 밀리는 계통 바이어스가 발견됐다(개루프 평균
바이어스는 ~30 m로 작음). 가설: 정합 오차 b(r)이 트랙을 따라 서서히 변해 EKF가
변화율을 속도로 오인. 본 스크립트는 참값 pose에서 렌더→탐지→연관(참값 pose)→PnP를
돌려 오차 벡터 e(t) = z − r_true 를 East 축을 따라 기록하고, --fit을 주면 East 구간별
보정 테이블(results/registration_correction.json)을 만든다. UnityMeasurementModel이
이 파일이 있으면 z에서 b(r̂)을 빼도록 한다(측정 모델 보정 — 계약 §2.4의 z 정의 불변).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mc import result_meta  # noqa: E402

BLUE, ORANGE, INK = "#1452C7", "#D16608", "#3B4148"


def run_diag(cfg: dict, traj: np.ndarray, detector: str, suns: list[tuple[float, float]],
             seed: int) -> list[dict]:
    """밴드 내 참값 pose마다 (태양각별) 오차 벡터 실측."""
    from sim.measurement import UnityMeasurementModel

    h_min, h_max = float(cfg["trn_band"]["h_min_m"]), float(cfg["trn_band"]["h_max_m"])
    rows: list[dict] = []
    model = UnityMeasurementModel(cfg, np.random.default_rng(seed), detector)
    try:
        for sun_az, sun_el in suns:
            model.sun_az, model.sun_el = float(sun_az), float(sun_el)
            for i, (t, x, y, z) in enumerate(traj):
                if not (h_min <= z <= h_max):
                    continue
                r = np.array([x, y, z])
                res = model.sample_frame(r, r, i, t)  # 연관은 참값 pose
                e = None if res["z"] is None else (np.asarray(res["z"]) - r).tolist()
                rows.append({
                    "sun_az": sun_az, "sun_el": sun_el, "t": float(t),
                    "east_m": float(x), "h_m": float(z), "e_xyz_m": e,
                    "valid": res["valid"], "n_match": res["n_match"],
                    "n_inliers": res["n_inliers"],
                })
                if len(rows) % 25 == 0:
                    print(f"  {len(rows)} frames...", flush=True)
    finally:
        model.close()
    return rows


def fit_bins(rows: list[dict], bin_km: float) -> dict:
    """East 구간별(bin) 오차 벡터 중앙값 → 보정 테이블. 태양각 전체를 합친다."""
    ok = [r for r in rows if r["valid"] and r["e_xyz_m"] is not None]
    east = np.array([r["east_m"] for r in ok])
    e = np.array([r["e_xyz_m"] for r in ok])
    lo = np.floor(east.min() / (bin_km * 1e3)) * bin_km * 1e3
    edges = np.arange(lo, east.max() + bin_km * 1e3, bin_km * 1e3)
    table = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (east >= a) & (east < b)
        if sel.sum() >= 4:
            table.append({"east_lo_m": float(a), "east_hi_m": float(b),
                          "n": int(sel.sum()),
                          "bias_xyz_m": np.median(e[sel], axis=0).tolist()})
    return {"kind": "east_bins", "bin_km": bin_km, "table": table,
            "global_bias_xyz_m": np.median(e, axis=0).tolist(), "n_frames": len(ok)}


def make_fig(rows: list[dict], fit: dict | None, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    comps = ["East err [m]", "North err [m]", "Up err [m]"]
    suns = sorted({(r["sun_az"], r["sun_el"]) for r in rows})
    cols = [BLUE, ORANGE, "#087A29", "#8A5EC7"]
    for (az, el), col in zip(suns, cols):
        sel = [r for r in rows if r["valid"] and r["e_xyz_m"] is not None
               and r["sun_az"] == az and r["sun_el"] == el]
        east = np.array([r["east_m"] for r in sel]) / 1e3
        e = np.array([r["e_xyz_m"] for r in sel])
        for k, ax in enumerate(axes):
            ax.plot(east, e[:, k], ".", ms=4, alpha=0.6, color=col,
                    label=f"sun az {az:g} el {el:g}" if k == 0 else None)
    if fit is not None:
        for k, ax in enumerate(axes):
            xs = [(b["east_lo_m"] + b["east_hi_m"]) / 2e3 for b in fit["table"]]
            ys = [b["bias_xyz_m"][k] for b in fit["table"]]
            ax.plot(xs, ys, "-", lw=2.5, color=INK,
                    label="bin median (correction)" if k == 0 else None)
    for k, ax in enumerate(axes):
        ax.axhline(0, color=INK, lw=0.8, alpha=0.5)
        ax.set_ylabel(comps[k])
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    axes[-1].set_xlabel("East [km] (along-track)")
    fig.suptitle("Measurement error vector along corridor (truth-pose association)",
                 fontsize=12, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"fig: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traj", default="frames/traj_demo.csv")
    ap.add_argument("--detector", default="runs/export/crater_int8_ort.onnx")
    ap.add_argument("--sun", default="135,30;315,30",
                    help='"az,el;az,el" 목록 — 조명 의존성 확인용')
    ap.add_argument("--fit", choices=["none", "east_bins"], default="none")
    ap.add_argument("--bin-km", type=float, default=10.0)
    ap.add_argument("--out", default="results/p8_reg_diag.json")
    ap.add_argument("--correction-out", default="results/registration_correction.json")
    ap.add_argument("--fig", default="figs/p8_reg_diag.png")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    d = np.genfromtxt(args.traj, delimiter=",", names=True)
    traj = np.column_stack([d["t"], d["x"], d["y"], d["z"]])
    traj = traj[:: max(1, int(round(1.0 / (traj[1, 0] - traj[0, 0]))))]  # 1 s 간격
    suns = [tuple(float(v) for v in s.split(",")) for s in args.sun.split(";")]

    rows = run_diag(cfg, traj, args.detector, suns, args.seed)
    ok = [r for r in rows if r["valid"] and r["e_xyz_m"] is not None]
    e = np.array([r["e_xyz_m"] for r in ok])
    out = {
        "meta": result_meta(args.config),
        "params": {"detector": args.detector, "suns": suns, "seed": args.seed},
        "n_frames": len(rows), "n_valid": len(ok),
        "bias_mean_xyz_m": e.mean(axis=0).tolist(),
        "bias_median_xyz_m": np.median(e, axis=0).tolist(),
        "sigma_mad_xyz_m": (1.4826 * np.median(np.abs(e - np.median(e, axis=0)), axis=0)).tolist(),
        "rows": rows,
    }
    fit = None
    if args.fit != "none":
        fit = fit_bins(rows, args.bin_km)
        fit["meta"] = out["meta"]
        Path(args.correction_out).write_text(json.dumps(fit, indent=2), encoding="utf-8")
        print(f"correction: {args.correction_out} ({len(fit['table'])} bins)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    make_fig(rows, fit, Path(args.fig))
    print(f"diag: {args.out}  valid {len(ok)}/{len(rows)}")
    print("bias median [E,N,U] =", [round(v, 1) for v in out["bias_median_xyz_m"]], "m")


if __name__ == "__main__":
    main()
