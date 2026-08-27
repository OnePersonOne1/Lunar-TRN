#!/usr/bin/env bash
# P3 앞단: site 후보별 crop → catalog_L → altitude band 분석
set -eu
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python
ROBBINS=data/raw/robbins/lunar_crater_database_robbins_2018_bundle/data/lunar_crater_database_robbins_2018.csv
WAC=data/raw/WAC_GLOBAL_E300S0450_100M.TIF
declare -A DEM=(
  [slim]=data/raw/SLDEM2015_512_30S_00S_000_045_FLOAT.LBL
  [highlands]=data/raw/SLDEM2015_512_60S_30S_000_045_FLOAT.LBL
)
for c in slim highlands; do
  echo "=== [$c] crop ==="
  $PY data/crop.py --config "config_site_$c.yaml" --dem "${DEM[$c]}" --dem-scale 1000 \
      --texture "$WAC" --out "data/processed/$c"
  echo "=== [$c] catalog ==="
  $PY data/catalog.py --config "config_site_$c.yaml" --robbins "$ROBBINS" \
      --dem "data/processed/$c/dem_L.npz" --out "data/processed/$c/catalog_L.csv" \
      --registration-fig "figs/p3_registration_$c.png"
  echo "=== [$c] altitude band ==="
  $PY scripts/analyze_altitude_band.py --config "config_site_$c.yaml" \
      --catalog "data/processed/$c/catalog_L.csv" \
      --out "results/p3_altitude_band_$c.json" --fig "figs/p3_altitude_band_$c.png"
done
echo "P3 FRONTEND DONE"
