"""TRN 고도 구간 분석 CLI: 공칭 궤적 고도별 시야 내 크레이터 수·최소 투영 직경 → trn_band 제안.

config trn_band는 바꾸지 않는다. 제안값만 results/p3_altitude_band.json에 기록한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.camera import K_cam, project  # noqa: E402
from sim.loop import run_closed_loop  # noqa: E402
from sim.mc import result_meta  # noqa: E402


def in_view_stats(catalog: np.ndarray, r: np.ndarray, cfg: dict) -> tuple[int, float]:
    """(중심이 화면 안 & 투영 직경 ≥ p_min 인 크레이터 수, 그중 최소 투영 직경 px)."""
    W, H = float(cfg["camera"]["W"]), float(cfg["camera"]["H"])
    p_min = float(cfg["catalog"]["p_min_px"])
    f = K_cam(cfg)[0, 0]
    uv, z_C, valid = project(catalog[:, :3], r, cfg)
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    d_px = np.where(inside, f * catalog[:, 3] / np.where(z_C > 0, z_C, np.inf), 0.0)
    ok = d_px >= p_min
    return int(ok.sum()), (float(d_px[ok].min()) if ok.any() else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--catalog", default="data/processed/catalog_L.csv")
    ap.add_argument("--out", default="results/p3_altitude_band.json")
    ap.add_argument("--fig", default="figs/p3_altitude_band.png")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cat = np.genfromtxt(args.catalog, delimiter=",", names=True)
    catalog = np.column_stack([cat["x"], cat["y"], cat["z"], cat["D"]])
    n_min = int(cfg["trn_band"]["n_min_in_view"])

    # 공칭 궤적: 참값 상태 + ZEM/ZEV (P1 검증 완료)
    res = run_closed_loop(cfg, args.seed, measurement="truth")
    traj = res["traj_true"]
    # 카메라 주기마다 평가
    spf = int(round(cfg["imu"]["rate_hz"] / cfg["camera"]["rate_hz"]))
    idx = np.arange(0, len(traj), spf)
    h = traj[idx, 2]
    counts, dmins = [], []
    for k in idx:
        n, dmin = in_view_stats(catalog, traj[k, :3], cfg)
        counts.append(n)
        dmins.append(dmin)
    counts = np.array(counts)
    dmins = np.array(dmins)

    ok = counts >= n_min  # d_px ≥ p_min은 in_view_stats에서 이미 반영
    h_ok = h[ok]
    proposal = (
        {"h_min_m": float(h_ok.min()), "h_max_m": float(h_ok.max())} if ok.any() else None
    )
    out = {
        "meta": result_meta(args.config),
        "n_min_in_view": n_min,
        "p_min_px": cfg["catalog"]["p_min_px"],
        "altitude_m": h.tolist(),
        "n_in_view": counts.tolist(),
        "min_proj_diam_px": dmins.tolist(),
        "proposed_band": proposal,
        "config_band": {"h_min_m": cfg["trn_band"]["h_min_m"], "h_max_m": cfg["trn_band"]["h_max_m"]},
        "note": "config trn_band는 사람이 확정한다. 이 파일은 제안일 뿐이다.",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"proposed_band": proposal}, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(h / 1000.0, counts, "b-", label="craters in view (d≥p_min)")
    ax1.axhline(n_min, color="b", linestyle=":", label=f"n_min={n_min}")
    ax1.set_xlabel("altitude [km]")
    ax1.set_ylabel("count", color="b")
    ax2 = ax1.twinx()
    ax2.plot(h / 1000.0, dmins, "g--", label="min projected diameter [px]")
    ax2.set_ylabel("min diameter [px]", color="g")
    if proposal:
        ax1.axvspan(proposal["h_min_m"] / 1000.0, proposal["h_max_m"] / 1000.0,
                    alpha=0.15, color="orange", label="proposed band")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("TRN altitude band feasibility")
    fig.tight_layout()
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"fig: {fig_path}")


if __name__ == "__main__":
    main()
