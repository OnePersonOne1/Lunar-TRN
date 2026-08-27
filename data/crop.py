"""SLDEM2015 DEM·LROC WAC 모자이크에서 site 박스 크롭 → L 격자 리샘플 (계약 §2.1).

출력: data/processed/dem_L.npz(x, y, z), texture_L.png(북이 위, 동이 오른쪽),
Unity용 16-bit RAW heightmap + heightmap_meta.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.catalog import _require_site  # noqa: E402


def _grid_latlon(cfg: dict, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """L 격자(x, y)와 대응 lat/lon (등장방형 역변환)."""
    lat0, lon0 = _require_site(cfg)
    R_m = float(cfg["moon"]["R_m"])
    box_e, box_n = (float(v) * 1000.0 for v in cfg["site"]["box_km"])
    x = np.linspace(-box_e / 2.0, box_e / 2.0, nx)
    y = np.linspace(-box_n / 2.0, box_n / 2.0, ny)
    lon = lon0 + np.degrees(x / (R_m * math.cos(math.radians(lat0))))
    lat = lat0 + np.degrees(y / R_m)
    return x, y, lat, lon


def _sample_raster(paths: list[str], lat: np.ndarray, lon: np.ndarray, scale: float) -> np.ndarray:
    """여러 타일을 병합해 (lat, lon) 격자에서 샘플링 (이중선형).

    SLDEM FLOAT_IMG(PDS)·WAC GeoTIFF는 등장방형 투영 미터 좌표라
    래스터 축을 CRS 기준으로 경위도 deg로 변환한 뒤 보간한다.
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import transform as crs_transform
    from scipy.interpolate import RegularGridInterpolator

    datasets = [rasterio.open(p) for p in paths]
    if len(datasets) == 1:
        # merge()는 PDS의 float32 범위 밖 nodata(-3.4e38)를 다루지 못해 0으로 채운다 → 단일 파일은 직접 읽기
        band = datasets[0].read(1).astype(np.float64) * scale
        transform = datasets[0].transform
    else:
        mosaic, transform = merge(datasets)
        band = mosaic[0].astype(np.float64) * scale
    ny_r, nx_r = band.shape
    cols = np.arange(nx_r)
    rows = np.arange(ny_r)
    xs = transform.c + transform.a * (cols + 0.5)
    ys = transform.f + transform.e * (rows + 0.5)  # e < 0: 위→아래 감소
    crs = datasets[0].crs
    if crs is not None and crs.is_projected:
        dst = rasterio.crs.CRS.from_proj4("+proj=longlat +R=1737400 +no_defs")
        lon_r = np.asarray(crs_transform(crs, dst, xs, np.zeros_like(xs))[0])
        lat_r = np.asarray(crs_transform(crs, dst, np.zeros_like(ys), ys)[1])
    else:
        lon_r, lat_r = xs, ys
    interp = RegularGridInterpolator(
        (lat_r[::-1], lon_r), band[::-1, :], bounds_error=False, fill_value=np.nan
    )
    LON, LAT = np.meshgrid(lon, lat)
    out = interp(np.column_stack([LAT.ravel(), LON.ravel()])).reshape(LAT.shape)
    if np.isnan(out).any():
        raise SystemExit("site 박스가 입력 타일 범위를 벗어난다. 타일을 더 넣거나 site를 조정해라.")
    for ds in datasets:
        ds.close()
    return out


def main() -> None:
    import cv2
    import yaml

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # CLI 규약 통일용, 여기서는 미사용
    ap.add_argument("--dem", nargs="+", required=True, help="SLDEM2015 타일 (m 단위 또는 --dem-scale)")
    ap.add_argument("--dem-scale", type=float, default=1.0, help="DEM 값 → m 배율 (SLDEM 0.5 m 단위면 0.5)")
    ap.add_argument("--texture", nargs="+", required=True, help="LROC WAC 모자이크 타일")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--grid-m", type=float, default=100.0, help="dem_L.npz 격자 간격 [m]")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    box_e, box_n = (float(v) * 1000.0 for v in cfg["site"]["box_km"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- DEM → L 격자
    nx = int(round(box_e / args.grid_m)) + 1
    ny = int(round(box_n / args.grid_m)) + 1
    x, y, lat, lon = _grid_latlon(cfg, nx, ny)
    z = _sample_raster(args.dem, lat, lon, args.dem_scale)
    z0 = float(z[ny // 2, nx // 2])  # 원점(중심) 고도
    z = z - z0  # z = DEM(lat,lon) − DEM(lat0,lon0)
    np.savez_compressed(out_dir / "dem_L.npz", x=x, y=y, z=z)
    print(f"dem_L.npz: {z.shape} (dz [{z.min():.1f}, {z.max():.1f}] m)")

    # --- 텍스처 (북이 위, 동이 오른쪽; 해상도는 원본 픽셀 밀도 유지 목적에서 grid와 동일 배율)
    tex = _sample_raster(args.texture, lat, lon, 1.0)
    tex_u8 = np.clip((tex - tex.min()) / max(tex.max() - tex.min(), 1e-9) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "texture_L.png"), tex_u8[::-1, :])  # 행 0이 북쪽이 되도록 뒤집기
    print(f"texture_L.png: {tex_u8.shape}")

    # --- Unity heightmap: 정사각 2^n+1, 부족한 쪽 0 패딩, 16-bit RAW little-endian
    res = int(cfg["unity"]["terrain_heightmap_res"])
    xh = np.linspace(-box_e / 2.0, box_e / 2.0, res)
    yh = np.linspace(-box_n / 2.0, box_n / 2.0, int(round(res * box_n / box_e)))
    _, _, lat_h, lon_h = _grid_latlon(cfg, len(xh), len(yh))
    zh = _sample_raster(args.dem, lat_h, lon_h, args.dem_scale) - z0
    z_min, z_max = float(zh.min()), float(zh.max())
    norm = (zh - z_min) / max(z_max - z_min, 1e-9)
    raw = np.zeros((res, res), dtype=np.uint16)
    raw[: len(yh), : len(xh)] = np.round(norm * 65535.0).astype(np.uint16)
    raw_path = out_dir / "heightmap.raw"
    raw[::-1, :].tofile(raw_path)  # Unity RAW: 첫 행이 북쪽(위)
    meta = {
        "resolution": res,
        "size_m": {"east": box_e, "north": box_n, "padded_square": box_e},
        "used_rows": len(yh), "used_cols": len(xh),
        "z_min_m": z_min, "z_max_m": z_max,
        "origin": "L 원점(착륙 목표점)이 박스 중심, heightmap 픽셀 (used_cols/2, used_rows/2)",
        "byte_order": "little-endian uint16", "row0": "north",
    }
    (out_dir / "heightmap_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"heightmap.raw: {res}x{res} uint16, z [{z_min:.1f}, {z_max:.1f}] m")


if __name__ == "__main__":
    main()
