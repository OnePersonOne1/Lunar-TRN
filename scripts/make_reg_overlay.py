"""정합 오차 시각 증거 CLI: 원본 텍스처(WAC 크롭) 위 카탈로그 원 직접 오버레이.

Unity를 배제하고 data/processed/texture_L.png에 catalog_L.csv의 원을 그려,
Robbins 카탈로그와 WAC 모자이크의 국소 정합 어긋남(P8b 바이어스의 근본 원인)을
크레이터 단위로 보인다. 회랑(y≈0) 근처 큰 크레이터를 East 순으로 뽑아 몽타주 저장.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def to_px(x: float, y: float, scale: float, box_e: float, box_n: float) -> tuple[float, float]:
    """L 좌표[m] → texture_L.png 픽셀 (원점=박스 중심, row0=north)."""
    return (x + box_e / 2) / scale, (box_n / 2 - y) / scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # 결정론 — 시드 미사용
    ap.add_argument("--texture", default="data/processed/texture_L.png")
    ap.add_argument("--catalog", default="data/processed/catalog_L.csv")
    ap.add_argument("--out", default="figs/p8_texture_catalog_overlay.png")
    ap.add_argument("--n", type=int, default=6, help="몽타주 크레이터 수 (3의 배수 권장)")
    ap.add_argument("--y-max-km", type=float, default=12.0, help="회랑 중심 |y| 상한")
    ap.add_argument("--d-min-km", type=float, default=3.0, help="선택 크레이터 최소 직경")
    ap.add_argument("--east-range-km", type=float, nargs=2, default=(-155.0, -60.0),
                    help="TRN 밴드 회랑 East 구간")
    args = ap.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    box_e = float(cfg["site"]["box_km"][0]) * 1e3
    box_n = float(cfg["site"]["box_km"][1]) * 1e3

    tex = cv2.imread(args.texture)
    if tex is None:
        raise SystemExit(f"텍스처 없음: {args.texture} (data/crop.py 산출)")
    scale = box_e / tex.shape[1]  # m/px
    cat = np.genfromtxt(args.catalog, delimiter=",", names=True)

    e_lo, e_hi = (v * 1e3 for v in args.east_range_km)
    sel = ((np.abs(cat["y"]) < args.y_max_km * 1e3) & (cat["x"] > e_lo)
           & (cat["x"] < e_hi) & (cat["D"] > args.d_min_km * 1e3))
    idx = np.argsort(cat["D"][sel])[::-1][: 2 * args.n]
    rows = np.flatnonzero(sel)[idx]
    rows = rows[np.argsort(cat["x"][rows])][: args.n]

    panels = []
    for i in rows:
        x, y, d_m = cat["x"][i], cat["y"][i], cat["D"][i]
        u, v = to_px(x, y, scale, box_e, box_n)
        r = d_m / 2 / scale
        win = int(max(60, r * 2.2))
        u0, v0 = int(u - win), int(v - win)
        crop = tex[max(0, v0):v0 + 2 * win, max(0, u0):u0 + 2 * win].copy()
        if crop.size == 0:
            continue
        up = 4  # 확대해 픽셀(=100 m) 단위 어긋남이 보이게
        crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
        cv2.circle(crop, (int((u - u0) * up), int((v - v0) * up)), int(r * up), (0, 60, 255), 2)
        cv2.drawMarker(crop, (int((u - u0) * up), int((v - v0) * up)), (0, 60, 255),
                       cv2.MARKER_CROSS, 20, 2)
        crop = cv2.resize(crop, (480, 480), interpolation=cv2.INTER_AREA)
        for text, org in ((f"E={x / 1e3:.0f}km D={d_m / 1e3:.1f}km", (8, 24)),
                          (f"1px_orig={scale:.0f}m", (8, 470))):
            cv2.putText(crop, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(crop, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        panels.append(crop)

    ncol = 3
    while len(panels) % ncol:
        panels.append(np.zeros_like(panels[0]))
    grid = np.vstack([np.hstack(panels[k:k + ncol]) for k in range(0, len(panels), ncol)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"fig: {args.out} ({len(rows)} craters, {scale:.0f} m/px)")


if __name__ == "__main__":
    main()
