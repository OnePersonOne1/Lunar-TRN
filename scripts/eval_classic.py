"""고전 베이스라인 vs YOLO 탐지 비교 CLI(P7c): 같은 val 프레임·같은 지표 구현으로
mAP50 / mAP50-95 / precision / recall / FP per frame / τ(1스레드 CPU) → results/p7c_det_compare.json.

지표는 perception/metrics.py 하나만 쓴다(계열별 평가 코드 차이 배제). 참고로 YOLO의
ultralytics val mAP(results/p5_det.json)도 함께 실어 교차 확인한다(보간 방식 차이로
소수점 셋째 자리 수준 차이는 정상).
τ는 perception/bench_tau.py와 같은 정의(전처리+추론+후처리), cpu_threads=1 고정.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.metrics import average_precision, map_50_95, match_counts  # noqa: E402
from scripts.train_classic import load_split  # noqa: E402
from sim.mc import result_meta  # noqa: E402

IOU_TP = 0.5


def _yolo_dets(model_path: str, cfg: dict, imgs: list[np.ndarray], conf: float) -> list[np.ndarray]:
    """(n, 5) [x0, y0, x1, y1, conf] — AP용 낮은 conf에서 뽑는다."""
    from perception.detect import Detector

    c = json.loads(json.dumps(cfg))
    c["detector"]["conf"] = conf
    det = Detector(model_path, c)
    return [det.detect(im)[:, :5].astype(float) for im in imgs]


def _classic_dets(model_path: str, cfg: dict, imgs: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    """(후보 전체(n,5), 운용 임계 z)."""
    from perception.classic import PCADetector

    det = PCADetector.load(model_path)
    return [det.candidates(im) for im in imgs], det.threshold


def _bench_tau(fn, imgs: list[np.ndarray], warmup: int, n_iter: int) -> dict:
    for i in range(warmup):
        fn(imgs[i % len(imgs)])
    ts = []
    for i in range(n_iter):
        t0 = time.perf_counter()
        fn(imgs[i % len(imgs)])
        ts.append(time.perf_counter() - t0)
    a = np.asarray(ts)
    return {"median_s": float(np.median(a)), "p95_s": float(np.percentile(a, 95)),
            "mean_s": float(a.mean()), "std_s": float(a.std()), "n_iter": len(ts),
            "samples_s": [float(v) for v in a]}


def _summary(dets_all: list[np.ndarray], dets_op: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    tp, fp, fn = match_counts(dets_op, gts, IOU_TP)
    n = max(len(gts), 1)
    return {
        "mAP50": average_precision(dets_all, gts, 0.5),
        "mAP50_95": map_50_95(dets_all, gts),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "fp_per_frame": fp / n, "det_per_frame": sum(len(d) for d in dets_op) / n,
        "n_gt": int(sum(len(g) for g in gts)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--classic", default=None, help="기본 config classic.model")
    ap.add_argument("--fp32", default="runs/detect/runs/train/crater/weights/best.pt")
    ap.add_argument("--int8", default="runs/export/crater_int8_ort.onnx")
    ap.add_argument("--n-tau", type=int, default=None, help="τ 벤치 반복 (기본 bench.n_iter)")
    ap.add_argument("--out", default="results/p7c_det_compare.json")
    ap.add_argument("--fig", default="figs/p7c_det_compare.png")
    args = ap.parse_args()

    import cv2

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cv2.setNumThreads(int(cfg["bench"]["cpu_threads"]))   # τ 비교 조건 통일(1스레드)
    classic_path = args.classic or cfg["classic"]["model"]
    map_conf = float(cfg["classic"]["map_conf"])
    warmup = int(cfg["bench"]["warmup"])
    n_iter = args.n_tau if args.n_tau is not None else int(cfg["bench"]["n_iter"])

    imgs, gts, _ = load_split(Path(args.dataset), "val", cfg)
    print(f"val frames: {len(imgs)}, GT: {sum(len(g) for g in gts)}", flush=True)

    entries: dict[str, dict] = {}

    cand, thr = _classic_dets(classic_path, cfg, imgs)
    op = [c[c[:, 4] >= thr] for c in cand]
    entries["classic_pca"] = {"model": classic_path, "threshold_z": thr, **_summary(cand, op, gts)}
    print(f"classic: {json.dumps({k: v for k, v in entries['classic_pca'].items()})}", flush=True)

    # 고도 사전정보 변형: 폐루프에서는 EKF 예측 고도와 카탈로그 D_min이 픽셀 직경 하한을
    # 정하므로(d = f·D/h), 그보다 미세한 스케일은 탐색할 이유가 없다. 스케일 전수 탐색이
    # 고전 방식 τ의 얼마를 차지하는지 분리해 보기 위한 조건.
    from perception.camera import K_cam
    from perception.classic import PCADetector as _PCA

    f_px = float(K_cam(cfg)[0, 0])
    d_min_prior = f_px * float(cfg["catalog"]["D_min_m"]) / float(cfg["trn_band"]["h_max_m"])
    prior = _PCA.load(classic_path).with_diameter_range(d_min_prior, cfg["classic"]["d_max_px"])
    cand_p = [prior.candidates(im) for im in imgs]
    op_p = [c[c[:, 4] >= prior.threshold] for c in cand_p]
    entries["classic_pca_prior"] = {
        "model": classic_path, "threshold_z": prior.threshold,
        "d_min_px": d_min_prior, "note": "altitude/catalog prior on scale search",
        **_summary(cand_p, op_p, gts)}
    print(f"classic(prior d_min={d_min_prior:.1f}px): mAP50={entries['classic_pca_prior']['mAP50']:.3f} "
          f"R={entries['classic_pca_prior']['recall']:.3f}", flush=True)

    for key, path in (("yolo_fp32", args.fp32), ("yolo_int8", args.int8)):
        dets_all = _yolo_dets(path, cfg, imgs, map_conf)
        op = [d[d[:, 4] >= float(cfg["detector"]["conf"])] for d in dets_all]
        entries[key] = {"model": path, "conf": float(cfg["detector"]["conf"]),
                        **_summary(dets_all, op, gts)}
        print(f"{key}: mAP50={entries[key]['mAP50']:.3f} mAP50-95={entries[key]['mAP50_95']:.3f} "
              f"P={entries[key]['precision']:.3f} R={entries[key]['recall']:.3f}", flush=True)

    # --- τ (같은 1스레드 CPU 조건)
    from perception.classic import PCADetector
    from perception.detect import Detector

    bench_imgs = [imgs[i] for i in np.random.default_rng(args.seed).choice(len(imgs), 20, replace=False)]
    pca = PCADetector.load(classic_path)
    entries["classic_pca"]["tau"] = _bench_tau(pca.detect, bench_imgs, warmup, n_iter)
    print(f"classic τ median={entries['classic_pca']['tau']['median_s'] * 1e3:.1f} ms", flush=True)
    entries["classic_pca_prior"]["tau"] = _bench_tau(prior.detect, bench_imgs, warmup, n_iter)
    print(f"classic(prior) τ median={entries['classic_pca_prior']['tau']['median_s'] * 1e3:.1f} ms",
          flush=True)
    ort = Detector(args.int8, cfg)
    entries["yolo_int8"]["tau"] = _bench_tau(ort.detect, bench_imgs, warmup, n_iter)
    print(f"yolo int8 τ median={entries['yolo_int8']['tau']['median_s'] * 1e3:.1f} ms", flush=True)

    ref = {}
    p5 = Path("results/p5_det.json")
    if p5.exists():
        d = json.loads(p5.read_text(encoding="utf-8"))
        ref = {k: {"mAP50": d[k]["mAP50"], "mAP50_95": d[k]["mAP50_95"]}
               for k in ("fp32", "int8") if k in d}

    out = {
        "meta": result_meta(args.config),
        "note": ("모든 지표는 perception/metrics.py 단일 구현(VOC2010+ 적분). "
                 "ultralytics val(101점 보간) 값은 ultralytics_reference에 별도 기재. "
                 f"τ는 1스레드 CPU, warmup {warmup}, n_iter {n_iter}."),
        "n_val_frames": len(imgs),
        "entries": {k: {kk: vv for kk, vv in v.items()} for k, v in entries.items()},
        "ultralytics_reference": ref,
    }
    for v in out["entries"].values():          # samples_s는 파일 크기만 키운다
        v.get("tau", {}).pop("samples_s", None)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")

    _make_fig(out, Path(args.fig))


def _make_fig(out: dict, fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"classic_pca": "Classic PCA", "classic_pca_prior": "Classic PCA\n(alt. prior)",
              "yolo_fp32": "YOLO11n FP32", "yolo_int8": "YOLO11n INT8"}
    keys = [k for k in labels if k in out["entries"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(keys))
    colors = ["#888888", "#bbbbbb", "#1f77b4", "#d62728"]
    for ax, (field, title) in zip(axes, [("mAP50_95", "mAP50-95"), ("recall", "recall / precision"),
                                         ("tau", "τ (1-thread CPU) [ms]")]):
        if field == "tau":
            vals = [out["entries"][k].get("tau", {}).get("median_s", np.nan) * 1e3 for k in keys]
            ax.bar(x, vals, color=colors[: len(keys)])
            for xi, v in zip(x, vals):
                if np.isfinite(v):
                    ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom")
        elif field == "recall":
            ax.bar(x - 0.2, [out["entries"][k]["recall"] for k in keys], 0.4, label="recall")
            ax.bar(x + 0.2, [out["entries"][k]["precision"] for k in keys], 0.4, label="precision")
            ax.set_ylim(0, 1.05)
            ax.legend()
        else:
            vals = [out["entries"][k][field] for k in keys]
            ax.bar(x, vals, color=colors[: len(keys)])
            for xi, v in zip(x, vals):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom")
            ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[k] for k in keys], fontsize=9)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Classic (PCA template) vs learned detector — val {out['n_val_frames']} frames")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"fig: {fig_path}")


if __name__ == "__main__":
    main()
