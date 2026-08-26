"""Robbins 크레이터 카탈로그(D ≥ D_min) → L 프레임 catalog_L.csv 변환 (계약 §2.1, §2.3).

등장방형 평면 근사: x = R_m·(lon − lon0)·cos(lat0), y = R_m·(lat − lat0),
z = DEM(lat, lon) − DEM(lat0, lon0).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def latlon_to_xy(
    lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float, lon0_deg: float, R_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """등장방형 평면 근사 (계약 §2.1). 입력 deg → 출력 m."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    x = R_m * (lon - lon0) * math.cos(lat0)
    y = R_m * (lat - lat0)
    return x, y


def _require_site(cfg: dict) -> tuple[float, float]:
    lat0, lon0 = cfg["site"]["lat0_deg"], cfg["site"]["lon0_deg"]
    if isinstance(lat0, str) or isinstance(lon0, str):
        raise SystemExit("config site.lat0_deg/lon0_deg가 TBD다. 사람이 P3에서 확정해야 한다.")
    return float(lat0), float(lon0)


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # CLI 규약 통일용, 여기서는 미사용
    ap.add_argument("--robbins", required=True, help="Robbins 카탈로그 CSV 경로 (data/raw/)")
    ap.add_argument("--dem", default="data/processed/dem_L.npz", help="crop.py 산출 DEM 격자")
    ap.add_argument("--out", default="data/processed/catalog_L.csv")
    ap.add_argument("--col-lat", default="LAT_CIRC_IMG")
    ap.add_argument("--col-lon", default="LON_CIRC_IMG")
    ap.add_argument("--col-diam", default="DIAM_CIRC_IMG", help="단위 km 가정")
    ap.add_argument("--registration-fig", default=None,
                    help="figs/p3_registration.png — texture_L.png 위 카탈로그 원 overlay")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    lat0, lon0 = _require_site(cfg)
    R_m = float(cfg["moon"]["R_m"])
    D_min = float(cfg["catalog"]["D_min_m"])
    box_e, box_n = (float(v) for v in cfg["site"]["box_km"])

    import csv

    with open(args.robbins, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        for c in (args.col_lat, args.col_lon, args.col_diam):
            if c not in cols:
                raise SystemExit(
                    f"열 '{c}'가 없다. 배포본 열 이름: {cols[:20]} … --col-* 인자로 매핑을 지정해라."
                )
        rows = [(float(r[args.col_lat]), float(r[args.col_lon]), float(r[args.col_diam]))
                for r in reader]
    lat = np.array([r[0] for r in rows])
    lon = np.array([r[1] for r in rows])
    D = np.array([r[2] for r in rows]) * 1000.0  # km → m

    x, y = latlon_to_xy(lat, lon, lat0, lon0, R_m)
    half_e, half_n = box_e * 1000.0 / 2.0, box_n * 1000.0 / 2.0
    keep = (D >= D_min) & (np.abs(x) <= half_e) & (np.abs(y) <= half_n)

    dem = np.load(args.dem)
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (dem["y"], dem["x"]), dem["z"], bounds_error=False, fill_value=0.0
    )
    z = interp(np.column_stack([y[keep], x[keep]]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "x", "y", "z", "D"])
        for i, (xi, yi, zi, di) in enumerate(zip(x[keep], y[keep], z, D[keep])):
            w.writerow([i, f"{xi:.2f}", f"{yi:.2f}", f"{zi:.2f}", f"{di:.1f}"])
    print(f"catalog_L: {keep.sum()}/{len(D)}개 (D ≥ {D_min} m, box {box_e}×{box_n} km) → {out_path}")

    if args.registration_fig:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tex_path = Path(args.dem).parent / "texture_L.png"
        fig, ax = plt.subplots(figsize=(12, 12 * box_n / box_e))
        if tex_path.exists():
            import cv2

            tex = cv2.imread(str(tex_path))[:, :, ::-1]
            ax.imshow(tex, extent=[-half_e, half_e, -half_n, half_n])  # 북이 위, 동이 오른쪽
        for xi, yi, di in zip(x[keep], y[keep], D[keep]):
            ax.add_patch(plt.Circle((xi, yi), di / 2.0, fill=False, color="lime", linewidth=0.8))
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_title("Robbins catalog vs texture registration")
        fig_path = Path(args.registration_fig)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"fig: {fig_path}")


if __name__ == "__main__":
    main()
