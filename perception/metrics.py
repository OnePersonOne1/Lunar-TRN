"""탐지 지표(AP·mAP50-95·TP/FP/FN) — 탐지기 계열과 무관한 공통 구현.

입력 규약: 이미지별 탐지 (n, ≥5) [x0, y0, x1, y1, score, ...], 이미지별 GT (m, 4) px.
AP는 VOC2010+ 방식(전 구간 precision 포락선 적분). ultralytics val은 101점 보간이라
소수점 셋째 자리에서 다를 수 있다 — scripts/eval_classic.py가 같은 모델로 교차 확인한다.
"""
from __future__ import annotations

import numpy as np

IOU_THRESHOLDS_50_95 = np.arange(0.5, 0.96, 0.05)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, m) IoU. a, b는 [x0, y0, x1, y1] 행렬."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def greedy_match(dets: np.ndarray, gts: np.ndarray, iou_thr: float) -> np.ndarray:
    """score 내림차순으로 각 탐지를 미매칭 GT 중 IoU 최대(≥ thr)에 배정. 탐지별 TP 불리언."""
    tp = np.zeros(len(dets), dtype=bool)
    if len(dets) == 0 or len(gts) == 0:
        return tp
    order = np.argsort(-dets[:, 4], kind="stable")
    m = iou_matrix(dets[order, :4], gts)
    used = np.zeros(len(gts), dtype=bool)
    for k, i in enumerate(order):
        row = np.where(used, -1.0, m[k])
        j = int(row.argmax())
        if row[j] >= iou_thr:
            tp[i] = True
            used[j] = True
    return tp


def average_precision(dets: list[np.ndarray], gts: list[np.ndarray], iou_thr: float) -> float:
    """단일 클래스 AP@iou_thr."""
    scores, flags, n_gt = [], [], 0
    for d, g in zip(dets, gts):
        d = np.asarray(d, dtype=float).reshape(-1, d.shape[1] if len(d) else 5)
        g = np.asarray(g, dtype=float).reshape(-1, 4)
        n_gt += len(g)
        if len(d) == 0:
            continue
        scores.append(d[:, 4])
        flags.append(greedy_match(d, g, iou_thr))
    if n_gt == 0 or not scores:
        return 0.0
    s = np.concatenate(scores)
    f = np.concatenate(flags)
    order = np.argsort(-s, kind="stable")
    tp_cum = np.cumsum(f[order])
    fp_cum = np.cumsum(~f[order])
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    # precision 포락선(단조 감소) 후 recall 증분 적분
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def map_50_95(dets: list[np.ndarray], gts: list[np.ndarray]) -> float:
    return float(np.mean([average_precision(dets, gts, float(t)) for t in IOU_THRESHOLDS_50_95]))


def match_counts(dets: list[np.ndarray], gts: list[np.ndarray], iou_thr: float) -> tuple[int, int, int]:
    """운용 임계에서의 (TP, FP, FN) 합계."""
    tp = fp = fn = 0
    for d, g in zip(dets, gts):
        d = np.asarray(d, dtype=float)
        g = np.asarray(g, dtype=float).reshape(-1, 4)
        if len(d) == 0:
            fn += len(g)
            continue
        flags = greedy_match(d, g, iou_thr)
        tp += int(flags.sum())
        fp += int((~flags).sum())
        fn += len(g) - int(flags.sum())
    return tp, fp, fn
