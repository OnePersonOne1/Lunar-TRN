"""P8 집계 CLI: Unity 실런 MC vs 통계 MC 비교 그림 3장 + results/p8_summary.json.

a) 착륙 산포 겹침(실런 vs 통계, 95% 타원·CEP)  b) 고도 bin별 PnP 오차 로버스트 σ
(폐루프 내 vs 데이터셋 iid)  c) 폐루프 내 τ vs 벤치 τ 히스토그램.
숫자는 전부 results/*.json·jsonl에서 읽는다.
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

BLUE, ORANGE, GREEN, GRAY, INK = "#1452C7", "#D16608", "#087A29", "#8A93A0", "#3B4148"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def robust_sigma(x: np.ndarray) -> float:
    """1.4826 x MAD (정규 일치 로버스트 σ)."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _style(ax) -> None:
    ax.grid(True, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_dispersion(mc: dict, stat: dict, out: Path, mc_corr: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    groups = [(stat, ORANGE, "stat MC"), (mc, BLUE, "Unity MC")]
    if mc_corr is not None:
        groups.append((mc_corr, GREEN, "Unity MC + reg. corr."))
    for d, col, name in groups:
        xy = np.asarray(d["landing_xy_m"])
        ax.scatter(xy[:, 0], xy[:, 1], s=14, color=col, alpha=0.45, lw=0, label=name)
        e = d["ellipse95"]
        ax.add_patch(Ellipse(e["center"], 2 * e["semi_axes_m"][0], 2 * e["semi_axes_m"][1],
                             angle=np.degrees(e["angle_rad"]), fill=False, color=col, lw=2))
    ax.plot(0, 0, "+", color=INK, ms=14, mew=2)
    ax.annotate("target", (0, 0), xytext=(8, 8), textcoords="offset points",
                color=INK, fontsize=9)
    # 축은 p99 반경까지 — 발산성 추락(수 km급) 1런이 축을 망치지 않게 하고 주석으로 표기
    all_xy = np.vstack([np.asarray(d["landing_xy_m"]) for d, _, _ in groups])
    lim = max(450.0, 1.15 * float(np.percentile(np.linalg.norm(all_xy, axis=1), 99)))
    n_out = int((np.abs(all_xy) > lim).any(axis=1).sum())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if n_out:
        ax.annotate(f"{n_out} run(s) outside view (association-failure crash)",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8.5, color=INK)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal")
    _style(ax)
    ax.legend(loc="upper right", frameon=False)
    title = (
        f"Unity MC CEP {mc['cep_m']:.1f} m [{mc['cep_ci95_m'][0]:.1f}, {mc['cep_ci95_m'][1]:.1f}]"
        f"  ·  stat MC CEP {stat['cep_m']:.1f} m "
        f"[{stat['cep_ci95_m'][0]:.1f}, {stat['cep_ci95_m'][1]:.1f}]")
    if mc_corr is not None:
        title += (f"\n+ registration corr. CEP {mc_corr['cep_m']:.1f} m "
                  f"[{mc_corr['cep_ci95_m'][0]:.1f}, {mc_corr['cep_ci95_m'][1]:.1f}]")
    ax.set_title(f"Landing dispersion (n=200/group)\n{title}", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"fig: {out}")


def fig_err_vs_altitude(meas: list[dict], p5: dict, out: Path) -> tuple[list, list]:
    """반환: (폐루프 bin 표, 데이터셋 bin 표) — 보고용."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_v = [m for m in meas if m["valid"] and m["pnp_err_m"] is not None
              and np.isfinite(m["pnp_err_m"]) and np.isfinite(m["h_true_m"])]  # 발산 런 잔여 NaN 제외
    h = np.asarray([m["h_true_m"] for m in rows_v])
    err = np.asarray([m["pnp_err_m"] for m in rows_v])
    edges = np.arange(22.0, 31.0, 1.0)  # TRN 밴드 22–30 km, 1 km bin
    loop_rows, ds_rows = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (h >= lo * 1e3) & (h < hi * 1e3)
        if sel.sum() >= 5:
            loop_rows.append({"h_lo_km": lo, "h_hi_km": hi, "n": int(sel.sum()),
                              "sigma_m": robust_sigma(err[sel])})
    for b in p5["bins"]:
        if 22.0 <= b["h_lo_km"] < 30.0:
            ds_rows.append({"h_lo_km": b["h_lo_km"], "h_hi_km": b["h_hi_km"],
                            "n": b["n_frames"], "sigma_m": b["sigma_horiz_m"]})

    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, height_ratios=[3, 1])
    for rows, col, mark, name in ((ds_rows, ORANGE, "s", "dataset iid (sigma_horiz)"),
                                  (loop_rows, BLUE, "o", "in-loop (robust sigma |PnP err|)")):
        hc = [0.5 * (r["h_lo_km"] + r["h_hi_km"]) for r in rows]
        ax.plot(hc, [r["sigma_m"] for r in rows], marker=mark, ms=6, lw=2,
                color=col, label=name)
    ax.set_ylabel("sigma [m]")
    _style(ax)
    ax.legend(frameon=False)
    ax.set_title("PnP error vs altitude: dataset (iid, val) vs closed-loop (P8 Unity MC)\n"
                 "note: dataset = horizontal sigma, in-loop = robust sigma of 3D error norm",
                 fontsize=11, color=INK)
    hc = [0.5 * (r["h_lo_km"] + r["h_hi_km"]) for r in loop_rows]
    axn.bar(hc, [r["n"] for r in loop_rows], width=0.85, color=GRAY, alpha=0.6)
    axn.set_ylabel("frames")
    axn.set_xlabel("altitude [km]")
    _style(axn)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"fig: {out}")
    return loop_rows, ds_rows


def fig_tau_hist(meas: list[dict], bench: dict, out: Path, clip_ms: float = 300.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tl = np.asarray([m["tau_wallclock_s"] for m in meas]) * 1e3
    tb = np.asarray(bench["samples_s"]) * 1e3
    n_out = int((tl > clip_ms).sum())
    bins = np.linspace(min(tb.min(), tl.min()), clip_ms, 60)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(tb, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
            color=ORANGE, label=f"bench (n={len(tb)})")
    ax.hist(np.clip(tl, None, clip_ms), bins=bins, density=True, histtype="step",
            lw=2, color=BLUE, label=f"in-loop (n={len(tl)})")
    ax.axvline(float(np.median(tb)), color=ORANGE, ls="--", lw=1)
    ax.axvline(float(np.median(tl)), color=BLUE, ls="--", lw=1)
    if n_out:
        ax.annotate(f"> {clip_ms:.0f} ms: {n_out} frames (warmup, max {tl.max():.0f} ms)",
                    xy=(0.98, 0.85), xycoords="axes fraction", ha="right",
                    fontsize=9, color=INK)
    ax.set_xlabel("tau [ms]")
    ax.set_ylabel("density")
    _style(ax)
    ax.legend(frameon=False)
    ax.set_title("Inference-chain latency: closed-loop wallclock vs standalone bench (n INT8 CPU)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"fig: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # 결정론 집계 — 시드 미사용
    ap.add_argument("--out", default="figs", help="그림 출력 디렉터리")
    ap.add_argument("--mc", default="results/p8_unity_mc.json")
    ap.add_argument("--mc-corr", default="results/p8_unity_mc_corr.json",
                    help="정합 보정 실런 MC (없으면 생략)")
    ap.add_argument("--meas", default="results/p8_unity_mc_meas.jsonl")
    ap.add_argument("--baseline", default="results/p7b_baseline.json")
    ap.add_argument("--p6", default="results/p6_closed_loop.json")
    ap.add_argument("--p5-alt", default="results/p5_sigma_vs_altitude.json")
    ap.add_argument("--bench", default="results/tau_ort_cpu_int8.json")
    ap.add_argument("--summary-out", default="results/p8_summary.json")
    args = ap.parse_args()

    def jload(p: str) -> dict:
        return json.loads(Path(p).read_text(encoding="utf-8"))

    mc, stat, p6, p5, bench = (jload(args.mc), jload(args.baseline), jload(args.p6),
                               jload(args.p5_alt), jload(args.bench))
    meas = load_jsonl(Path(args.meas))
    mc_corr = (json.loads(Path(args.mc_corr).read_text(encoding="utf-8"))
               if Path(args.mc_corr).exists() else None)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_dispersion(mc, stat, out_dir / "p8_landing_dispersion.png", mc_corr)
    loop_rows, ds_rows = fig_err_vs_altitude(meas, p5, out_dir / "p8_err_vs_altitude_inloop.png")
    fig_tau_hist(meas, bench, out_dir / "p8_tau_inloop_hist.png")

    stat_xy = np.asarray(stat["landing_xy_m"])
    stat_radii = np.linalg.norm(stat_xy, axis=1)
    unity_xy = np.asarray(mc["landing_xy_m"])
    tb = np.asarray(bench["samples_s"]) * 1e3
    summary = {
        "meta": result_meta(args.config),
        "cep_unity_m": mc["cep_m"],
        "cep_unity_ci95_m": mc["cep_ci95_m"],
        "cep_stat_m": stat["cep_m"],
        "cep_stat_ci95_m": stat["cep_ci95_m"],
        "ratio_unity_over_stat": mc["cep_m"] / stat["cep_m"],
        "p6_single_err_m": p6["landing_error_m"],
        "r95_over_cep_unity": mc["r95_over_cep"],
        "r95_over_cep_stat": float(np.percentile(stat_radii, 95) / stat["cep_m"]),
        "n_diverged_unity": mc.get("n_diverged", 0),
        "unity_mean_offset_m": unity_xy.mean(axis=0).tolist(),  # 계통 편향(바이어스) 벡터
        "unity_bias_norm_m": float(np.linalg.norm(unity_xy.mean(axis=0))),
        # 중앙값 기반(로버스트): 연관 실패 추락 런 1개가 평균을 오염시키는 것 방지
        "unity_median_offset_m": np.median(unity_xy, axis=0).tolist(),
        "unity_bias_med_norm_m": float(np.linalg.norm(np.median(unity_xy, axis=0))),
        "stat_mean_offset_m": stat_xy.mean(axis=0).tolist(),
        "stat_bias_norm_m": float(np.linalg.norm(stat_xy.mean(axis=0))),
        "tau_inloop_median_ms": mc["tau_wallclock_s"]["median"] * 1e3,
        "tau_inloop_p95_ms": mc["tau_wallclock_s"]["p95"] * 1e3,
        "tau_bench_median_ms": float(np.median(tb)),
        "tau_bench_p95_ms": float(np.percentile(tb, 95)),
        "sigma_vs_altitude": {"in_loop": loop_rows, "dataset_iid": ds_rows},
    }
    if mc_corr is not None:
        cxy = np.asarray(mc_corr["landing_xy_m"])
        summary.update({
            "cep_unity_corr_m": mc_corr["cep_m"],
            "cep_unity_corr_ci95_m": mc_corr["cep_ci95_m"],
            "unity_corr_bias_norm_m": float(np.linalg.norm(cxy.mean(axis=0))),
            "unity_corr_median_offset_m": np.median(cxy, axis=0).tolist(),
            "unity_corr_bias_med_norm_m": float(np.linalg.norm(np.median(cxy, axis=0))),
            "r95_over_cep_unity_corr": mc_corr["r95_over_cep"],
            "n_diverged_unity_corr": mc_corr.get("n_diverged", 0),
        })
    Path(args.summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary: {args.summary_out}")


if __name__ == "__main__":
    main()
