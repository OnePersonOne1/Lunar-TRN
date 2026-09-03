"""고전 vs 학습 탐지기의 착륙 CEP 비교 CLI(P7c): 측정 통계 × τ 2×2 → results/p7c_cep_compare.json.

조건(직렬 촬영 모델·지연 보상 on, 나머지 config 동일):
  1. yolo_int8   : YOLO INT8 측정 통계(σ·오검출률·오프셋) + YOLO INT8 실측 τ
  2. classic_pca : 고전 측정 통계 + 고전 실측 τ                      (탐지기 전체 교체)
  3. classic_stats_yolo_tau : 고전 측정 통계 + YOLO τ                 (측정 품질 효과만)
  4. yolo_stats_classic_tau : YOLO 측정 통계 + 고전 τ                 (지연 효과만)
3·4는 "mAP 격차가 CEP로 얼마나 번역되는가"를 품질 축과 지연 축으로 분해하기 위한 것이다.
오검출률은 각 모델의 보정 파일에서 측정된 fp_rate_est를 쓴다(config 가정값 아님).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mc import (  # noqa: E402
    bootstrap_cep_ci, cep, error_ellipse_95, extras_summary, result_meta, run_mc,
)


def _meas(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/p7c_cep_compare.json")
    ap.add_argument("--fig", default="figs/p7c_cep_compare.png")
    ap.add_argument("--yolo-meas", default="results/measurement_model.json")
    ap.add_argument("--classic-meas", default="results/measurement_model_classic.json")
    ap.add_argument("--det-compare", default="results/p7c_det_compare.json",
                    help="τ 실측 출처 (eval_classic.py 산출)")
    ap.add_argument("--classic-key", default="classic_pca_prior",
                    help="τ를 가져올 고전 조건 키 (폐루프는 고도 사전정보 변형을 쓴다)")
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["tau"]["serial"] = True
    n_runs = args.n_runs if args.n_runs is not None else int(cfg["mc"]["n_runs"])
    workers = args.workers if args.workers is not None else int(cfg["mc"]["workers"])
    rng = np.random.default_rng(args.seed)

    det = json.loads(Path(args.det_compare).read_text(encoding="utf-8"))["entries"]
    tau = {"yolo": float(det["yolo_int8"]["tau"]["median_s"]),
           "classic": float(det[args.classic_key]["tau"]["median_s"])}
    meas = {"yolo": args.yolo_meas, "classic": args.classic_meas}
    fp_rate = {k: float(_meas(v)["fp_rate_est"]) for k, v in meas.items()}
    sigma = {k: _meas(v)["sigma_xyz_m"] for k, v in meas.items()}

    conditions = [
        ("yolo_int8", "yolo", "yolo"),
        ("classic_pca", "classic", "classic"),
        ("classic_stats_yolo_tau", "classic", "yolo"),
        ("yolo_stats_classic_tau", "yolo", "classic"),
    ]
    out_conditions = []
    for label, m_key, t_key in conditions:
        c = json.loads(json.dumps(cfg))
        c["measurement"]["file"] = meas[m_key]
        xy, v, extras = run_mc(c, n_runs, workers, seed0=args.seed,
                               tau=tau[t_key], fp_rate=fp_rate[m_key], delay_comp=True)
        lo, hi = bootstrap_cep_ci(xy, int(cfg["mc"]["bootstrap_n"]), rng)
        summ = extras_summary(extras)
        drop = summ["mean_n_dropped"] / max(summ["mean_n_meas"] + summ["mean_n_dropped"], 1e-9)
        out_conditions.append({
            "label": label, "measurement_from": m_key, "tau_from": t_key,
            "tau_s": tau[t_key], "fp_rate": fp_rate[m_key], "sigma_xyz_m": sigma[m_key],
            "cep_m": cep(xy), "cep_ci95_m": [lo, hi], "ellipse95": error_ellipse_95(xy),
            "n_runs": n_runs, "drop_ratio": drop,
            "landing_v_mean_mps": float(v.mean()), **summ,
        })
        print(f"{label:24s} τ={tau[t_key] * 1e3:6.1f} ms  fp={fp_rate[m_key]:.3f}  "
              f"CEP={out_conditions[-1]['cep_m']:8.2f} m  CI=[{lo:.1f}, {hi:.1f}]  "
              f"gate={summ['mean_gate_accept']:.3f}", flush=True)

    out = {
        "meta": result_meta(args.config),
        "note": ("직렬 촬영 모델·지연 보상 on. 각 조건의 오검출률은 해당 모델 보정 파일의 "
                 "fp_rate_est(측정값). 3·4번은 품질/지연 효과 분해용 가상 조합."),
        "detector_metrics": {k: {kk: det[k][kk] for kk in
                                 ("mAP50", "mAP50_95", "precision", "recall", "fp_per_frame")}
                             for k in ("classic_pca", "classic_pca_prior", "yolo_int8", "yolo_fp32")
                             if k in det},
        "classic_key": args.classic_key,
        "conditions": out_conditions,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")

    _make_fig(out, Path(args.fig))


def _make_fig(out: dict, fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = out["conditions"]
    names = {"yolo_int8": "YOLO INT8\n(stats+τ)", "classic_pca": "Classic PCA\n(stats+τ)",
             "classic_stats_yolo_tau": "Classic stats\n+ YOLO τ",
             "yolo_stats_classic_tau": "YOLO stats\n+ Classic τ"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    dm = out["detector_metrics"]
    keys = [k for k in ("classic_pca", "classic_pca_prior", "yolo_int8") if k in dm]
    names_d = {"classic_pca": "Classic PCA", "classic_pca_prior": "Classic PCA\n(alt. prior)",
               "yolo_int8": "YOLO INT8"}
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [dm[k]["mAP50_95"] for k in keys], 0.4, label="mAP50-95")
    ax.bar(x + 0.2, [dm[k]["recall"] for k in keys], 0.4, label="recall")
    ax.set_xticks(x)
    ax.set_xticklabels([names_d[k] for k in keys], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("detection metrics (val)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    x = np.arange(len(conds))
    ceps = [c["cep_m"] for c in conds]
    err = np.array([[c["cep_m"] - c["cep_ci95_m"][0] for c in conds],
                    [c["cep_ci95_m"][1] - c["cep_m"] for c in conds]])
    ax.bar(x, ceps, yerr=err, capsize=4,
           color=["#1f77b4", "#888888", "#b0b0b0", "#7fb3d5"])
    for xi, c in zip(x, conds):
        ax.text(xi, c["cep_m"], f"{c['cep_m']:.0f}", ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([names.get(c["label"], c["label"]) for c in conds], fontsize=8)
    ax.set_ylabel("CEP [m]")
    ax.set_title(f"landing CEP, n={conds[0]['n_runs']} (95% CI)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Classic (PCA template) vs learned detector — detection metrics and landing CEP")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"fig: {fig_path}")


if __name__ == "__main__":
    main()
