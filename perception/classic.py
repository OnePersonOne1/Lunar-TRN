"""고전 베이스라인 크레이터 탐지기(P7c): PCA 외형 부분공간 템플릿 + Fisher 선형 판별.

SLIM 계열 비딥러닝 방식(PCA 외형 기저 탐지)의 근사. 학습 데이터·라벨·평가 프레임은
YOLO와 동일하게 두고, 탐지기 계열만 바꿔 mAP·측정 통계·τ·CEP를 나란히 잰다.

알고리즘
- 학습: GT 원(직경 D) 주변 crop_margin·D 창을 patch_px²로 정규화(평균 0·노름 1) →
  양성 패치 PCA 기저 U(k개). 특징 f = [Uᵀ(x−μ), 잔차 에너지(DFFS)] (k+1). 양성 vs 무작위
  음성 창으로 Fisher LDA 방향 w → 점수를 음성 기준 z-score로 표준화. 운용 임계는
  train 프레임 F1 최대로 결정(YOLO의 conf에 대응).
- 탐지: 직경 격자(d_min…d_max, scale_step)마다 이미지를 patch_px/(crop_margin·d)로 축소해
  모든 창의 정규화 투영을 상관 필터(k+1개)+박스 필터 2개로 한 번에 계산(고전 템플릿 매칭
  구현 그대로) → z ≥ score_floor 국소 최대 → 스케일 간 NMS.
반환 형식은 perception.detect.Detector와 같은 (n, 6) [x0, y0, x1, y1, conf, cls].
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from perception.metrics import greedy_match, iou_matrix

_EPS = 1e-6


def _to_gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return g.astype(np.float32)


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """평균 0·노름 1 벡터 (patch_px²,)."""
    x = patch.astype(np.float64).ravel()
    x = x - x.mean()
    return x / max(float(np.linalg.norm(x)), _EPS)


def _crop_window(gray: np.ndarray, cx: float, cy: float, win: float, patch_px: int) -> np.ndarray | None:
    """중심(cx, cy)·한 변 win 창을 patch_px²로. 화면 밖이면 None."""
    h, w = gray.shape
    x0, y0 = int(round(cx - win / 2)), int(round(cy - win / 2))
    x1, y1 = int(round(cx + win / 2)), int(round(cy + win / 2))
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return cv2.resize(gray[y0:y1, x0:x1], (patch_px, patch_px), interpolation=cv2.INTER_AREA)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """greedy NMS → 유지 인덱스(score 내림차순)."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.asarray(keep, dtype=int)


def diameter_grid(d_min: float, d_max: float, step: float) -> np.ndarray:
    ds = [float(d_min)]
    while ds[-1] * step <= d_max:
        ds.append(ds[-1] * step)
    return np.asarray(ds)


class PCADetector:
    """학습 결과(기저·판별 방향·임계)를 담고 (n, 6) 탐지를 돌려주는 고전 탐지기."""

    def __init__(self, params: dict, mean: np.ndarray, basis: np.ndarray, w_lda: np.ndarray,
                 z_mu: float, z_sd: float, threshold: float) -> None:
        self.params = {k: float(v) if isinstance(v, (int, float)) else v for k, v in params.items()}
        self.patch_px = int(params["patch_px"])
        self.mean = np.asarray(mean, dtype=np.float64)
        self.basis = np.asarray(basis, dtype=np.float64)        # (k, patch²), 각 행 평균 0
        self.w_lda = np.asarray(w_lda, dtype=np.float64)        # (k+1,)
        self.z_mu, self.z_sd = float(z_mu), float(z_sd)
        self.threshold = float(threshold)
        p = self.patch_px
        self._kernels = [b.reshape(p, p).astype(np.float32) for b in self.basis]
        self._kernel_mu = self.mean.reshape(p, p).astype(np.float32)
        self._proj_mu = self.basis @ self.mean                  # Uμ
        self._mu_sq = float(self.mean @ self.mean)
        self._ds = diameter_grid(float(params["d_min_px"]), float(params["d_max_px"]),
                                 float(params["scale_step"]))

    # ------------------------------------------------------------ 특징·점수
    def features(self, patches: np.ndarray) -> np.ndarray:
        """정규화 패치 행렬 (N, patch²) → (N, 2k+1) [c, c², dffs].

        c는 부호(조명 방향에 따라 뒤집힘)를, c²는 부호와 무관한 에너지(DIFS)를,
        dffs는 부분공간 밖 잔차를 담는다 — 고전 eigen-appearance 탐지의 표준 특징 조합.
        """
        xc = patches - self.mean
        c = xc @ self.basis.T
        dffs = np.sum(xc**2, axis=1) - np.sum(c**2, axis=1)
        return np.column_stack([c, c**2, dffs])

    def zscore(self, feats: np.ndarray) -> np.ndarray:
        return (feats @ self.w_lda - self.z_mu) / self.z_sd

    def _score_map(self, gray_s: np.ndarray) -> np.ndarray:
        """축소 영상의 모든 patch_px 창에 대한 z-score (h−p+1, w−p+1).

        상관은 cv2.matchTemplate(TM_CCORR, 유효 영역 출력)로 계산한다 — 고전 템플릿 매칭의
        표준 구현이며 filter2D보다 빠르다.
        """
        p = self.patch_px
        n = float(p * p)
        h, w = gray_s.shape
        if h < p or w < p:
            return np.empty((0, 0), dtype=np.float32)
        ones = np.ones((p, p), dtype=np.float32)
        s1 = cv2.matchTemplate(gray_s, ones, cv2.TM_CCORR)
        s2 = cv2.matchTemplate(gray_s * gray_s, ones, cv2.TM_CCORR)
        norm = np.sqrt(np.maximum(s2 - s1 * s1 / n, _EPS))
        # 기저·평균은 평균 0이므로 정규화 창과의 내적 = 원시 상관 / 노름
        corr_mu = cv2.matchTemplate(gray_s, self._kernel_mu, cv2.TM_CCORR) / norm
        x_mu_sq = 1.0 - 2.0 * corr_mu + self._mu_sq                 # ‖x−μ‖²
        score = np.zeros_like(norm)
        c_sq = np.zeros_like(norm)
        k = len(self._kernels)
        for i, ker in enumerate(self._kernels):
            c = cv2.matchTemplate(gray_s, ker, cv2.TM_CCORR) / norm - self._proj_mu[i]
            cc = c * c
            score += self.w_lda[i] * c + self.w_lda[k + i] * cc
            c_sq += cc
        score += self.w_lda[-1] * (x_mu_sq - c_sq)
        return (score - self.z_mu) / self.z_sd

    # ------------------------------------------------------------ 탐지
    def candidates(self, img: np.ndarray, score_floor: float | None = None) -> np.ndarray:
        """NMS까지 마친 후보 (n, 5) [x0, y0, x1, y1, z]. z ≥ score_floor."""
        floor = float(self.params["score_floor"]) if score_floor is None else float(score_floor)
        gray = _to_gray(img)
        h, w = gray.shape
        p = self.patch_px
        margin = float(self.params["crop_margin"])
        max_per_scale = int(self.params["max_cand_per_scale"])
        boxes, scores = [], []
        for d in self._ds:
            win = d * margin
            scale = p / win
            hs, ws = int(round(h * scale)), int(round(w * scale))
            if hs < p or ws < p:
                continue
            gray_s = cv2.resize(gray, (ws, hs), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            z = self._score_map(gray_s)
            if z.size == 0:
                continue
            local_max = cv2.dilate(z, np.ones((3, 3), np.uint8))
            ys, xs = np.nonzero((z >= floor) & (z >= local_max))
            if len(ys) == 0:
                continue
            zs = z[ys, xs].astype(np.float64)
            if len(zs) > max_per_scale:                 # 스케일별 상위 N만 (NMS 비용 상한)
                top = np.argpartition(-zs, max_per_scale)[:max_per_scale]
                ys, xs, zs = ys[top], xs[top], zs[top]
            cx = (xs + p / 2.0) / scale
            cy = (ys + p / 2.0) / scale
            boxes.append(np.column_stack([cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2]))
            scores.append(zs)
        if not boxes:
            return np.empty((0, 5))
        b = np.concatenate(boxes)
        s = np.concatenate(scores)
        keep = _nms(b, s, float(self.params["nms_iou"]))
        return np.column_stack([b[keep], s[keep]])

    def detect(self, img: np.ndarray) -> np.ndarray:
        """(n, 6) [x0, y0, x1, y1, conf, cls]. conf = sigmoid(z − 임계) → 임계에서 0.5."""
        cand = self.candidates(img)
        cand = cand[cand[:, 4] >= self.threshold]
        conf = 1.0 / (1.0 + np.exp(-(cand[:, 4] - self.threshold)))
        return np.column_stack([cand[:, :4], conf, np.zeros(len(cand))]).astype(np.float32)

    def with_diameter_range(self, d_min: float, d_max: float) -> "PCADetector":
        """직경 탐색 범위만 바꾼 사본 (고도·카탈로그 사전정보로 스케일 탐색을 줄일 때)."""
        params = dict(self.params, d_min_px=float(d_min), d_max_px=float(d_max))
        return PCADetector(params, self.mean, self.basis, self.w_lda,
                           self.z_mu, self.z_sd, self.threshold)

    # ------------------------------------------------------------ 저장
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = ["patch_px", "crop_margin", "d_min_px", "d_max_px", "scale_step", "nms_iou",
                "score_floor", "max_cand_per_scale"]
        np.savez(path, mean=self.mean, basis=self.basis, w_lda=self.w_lda,
                 z_mu=self.z_mu, z_sd=self.z_sd, threshold=self.threshold,
                 params=np.asarray([float(self.params[k]) for k in keys]),
                 param_keys=np.asarray(keys))

    @classmethod
    def load(cls, path: str | Path) -> "PCADetector":
        d = np.load(path, allow_pickle=False)
        params = {str(k): float(v) for k, v in zip(d["param_keys"], d["params"])}
        return cls(params, d["mean"], d["basis"], d["w_lda"],
                   float(d["z_mu"]), float(d["z_sd"]), float(d["threshold"]))


# ---------------------------------------------------------------- 학습

def _sample_patches(gray: np.ndarray, gts: np.ndarray, c: dict, rng: np.random.Generator
                    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """한 프레임의 양성(GT 창)·음성(무작위 창) 정규화 패치."""
    p, margin = int(c["patch_px"]), float(c["crop_margin"])
    d_min, d_max = float(c["d_min_px"]), float(c["d_max_px"])
    h, w = gray.shape
    pos, neg = [], []
    for x0, y0, x1, y1 in gts:
        d = max(x1 - x0, y1 - y0)
        if not (d_min <= d <= d_max):
            continue
        patch = _crop_window(gray, (x0 + x1) / 2, (y0 + y1) / 2, d * margin, p)
        if patch is not None:
            pos.append(normalize_patch(patch))
    n_neg = int(c["n_neg_per_frame"])
    d_hi = min(d_max, min(h, w) / margin)
    tries = 0
    while len(neg) < n_neg and tries < 20 * n_neg:
        tries += 1
        d = float(np.exp(rng.uniform(np.log(d_min), np.log(d_hi))))
        cx = rng.uniform(d * margin / 2, w - d * margin / 2)
        cy = rng.uniform(d * margin / 2, h - d * margin / 2)
        box = np.array([[cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2]])
        if len(gts) and iou_matrix(box, gts).max() > float(c["neg_iou_max"]):
            continue
        patch = _crop_window(gray, cx, cy, d * margin, p)
        if patch is not None:
            neg.append(normalize_patch(patch))
    return pos, neg


def _fisher_lda(fp: np.ndarray, fn: np.ndarray, ridge: float) -> np.ndarray:
    mu_p, mu_n = fp.mean(0), fn.mean(0)
    sw = np.cov(fp.T, bias=True) * len(fp) + np.cov(fn.T, bias=True) * len(fn)
    sw = sw / (len(fp) + len(fn))
    reg = ridge * np.trace(sw) / sw.shape[0] + _EPS
    w = np.linalg.solve(sw + reg * np.eye(sw.shape[0]), mu_p - mu_n)
    return w / max(float(np.linalg.norm(w)), _EPS)


def tune_threshold(det: PCADetector, imgs: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    """train 프레임 후보에서 F1(IoU 0.5) 최대 z 임계.

    greedy 매칭은 점수 내림차순 prefix로 결정되므로, 임계 t 이상 부분집합의 매칭은 전체
    매칭의 prefix와 같다. 매칭 한 번으로 모든 절단점의 TP/FP 곡선을 얻어 F1 최대를 고른다.
    """
    scores, flags, n_gt = [], [], 0
    for im, g in zip(imgs, gts):
        cand = det.candidates(im)
        n_gt += len(g)
        if len(cand):
            scores.append(cand[:, 4])
            flags.append(greedy_match(cand, np.asarray(g, dtype=float).reshape(-1, 4), 0.5))
    floor = float(det.params["score_floor"])
    if not scores or n_gt == 0:
        return {"threshold": floor, "f1": 0.0, "precision": 0.0, "recall": 0.0, "n_gt": n_gt}
    s = np.concatenate(scores)
    f = np.concatenate(flags)
    order = np.argsort(-s, kind="stable")
    s, f = s[order], f[order]
    tp = np.cumsum(f)
    n_keep = np.arange(1, len(s) + 1)
    prec, rec = tp / n_keep, tp / n_gt
    f1 = 2 * prec * rec / np.maximum(prec + rec, _EPS)
    i = int(np.argmax(f1))
    return {"threshold": float(s[i]), "f1": float(f1[i]), "precision": float(prec[i]),
            "recall": float(rec[i]), "n_gt": int(n_gt)}


def train_pca_detector(frames: Iterable[tuple[np.ndarray, np.ndarray]], c: dict,
                       rng: np.random.Generator) -> tuple[PCADetector, dict]:
    """(이미지, GT px 박스) 쌍 이터러블에서 PCA 기저·LDA·임계를 학습. 결정론(rng 인자).

    frames는 한 번만 순회한다(디스크에서 스트리밍 가능 — 전체 학습셋을 메모리에 올리지
    않는다). 임계 튜닝용 프레임은 순회 중 reservoir sampling으로 n_tune_frames개만 보관.
    """
    n_tune = int(c["n_tune_frames"])
    pos, neg, tune_frames, n_seen = [], [], [], 0
    for img, g in frames:
        g = np.asarray(g, dtype=float).reshape(-1, 4)
        p_i, n_i = _sample_patches(_to_gray(img), g, c, rng)
        pos += p_i
        neg += n_i
        if len(tune_frames) < n_tune:                    # reservoir sampling
            tune_frames.append((img, g))
        elif n_tune > 0:
            j = int(rng.integers(0, n_seen + 1))
            if j < n_tune:
                tune_frames[j] = (img, g)
        n_seen += 1
    if len(pos) < int(c["n_components"]) + 1 or len(neg) < 2:
        raise ValueError(f"학습 패치 부족: pos={len(pos)}, neg={len(neg)}")
    pos_arr = np.asarray(pos)
    n_pos_max = int(c["n_pos_max"])
    if len(pos_arr) > n_pos_max:
        pos_arr = pos_arr[rng.choice(len(pos_arr), n_pos_max, replace=False)]
    neg_arr = np.asarray(neg)

    mean = pos_arr.mean(0)
    _, sv, vt = np.linalg.svd(pos_arr - mean, full_matrices=False)
    k = int(c["n_components"])
    basis = vt[:k]
    evr = (sv[:k] ** 2 / np.sum(sv**2)).tolist()

    params = {key: c[key] for key in
              ("patch_px", "crop_margin", "d_min_px", "d_max_px", "scale_step", "nms_iou",
               "score_floor", "max_cand_per_scale")}
    det = PCADetector(params, mean, basis, np.zeros(2 * k + 1), 0.0, 1.0, 0.0)
    f_pos, f_neg = det.features(pos_arr), det.features(neg_arr)
    w = _fisher_lda(f_pos, f_neg, float(c["lda_ridge"]))
    s_neg = f_neg @ w
    det = PCADetector(params, mean, basis, w, float(s_neg.mean()), float(s_neg.std() + _EPS), 0.0)

    tune = tune_threshold(det, [im for im, _ in tune_frames], [g for _, g in tune_frames])
    det.threshold = tune["threshold"]
    info = {
        "n_frames": n_seen, "n_pos": int(len(pos_arr)), "n_neg": int(len(neg_arr)),
        "explained_variance_ratio": evr,
        "pos_z_mean": float(np.mean(det.zscore(f_pos))),
        "threshold": det.threshold, "tune_f1": tune["f1"],
        "tune_precision": tune["precision"], "tune_recall": tune["recall"],
        "n_tune_frames": len(tune_frames),
    }
    return det, info
