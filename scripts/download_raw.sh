#!/usr/bin/env bash
# P3 원본 데이터 다운로드 (data/raw/). curl -C - 로 중단 시 이어받기.
set -u
cd "$(dirname "$0")/.."
mkdir -p data/raw
BASE_SLDEM="https://imbrium.mit.edu/DATA/SLDEM2015/TILES/FLOAT_IMG"
BASE_WAC="https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/BROWSE/WAC_GLOBAL"
files=(
  "$BASE_SLDEM/SLDEM2015_512_30S_00S_000_045_FLOAT.LBL"
  "$BASE_SLDEM/SLDEM2015_512_30S_00S_000_045_FLOAT.IMG"
  "$BASE_SLDEM/SLDEM2015_512_60S_30S_000_045_FLOAT.LBL"
  "$BASE_SLDEM/SLDEM2015_512_60S_30S_000_045_FLOAT.IMG"
  "$BASE_WAC/WAC_GLOBAL_E300S0450_100M.TIF"
)
# Robbins 크레이터 카탈로그 (PDS4 번들 zip, 96 MB → data/raw/robbins/에 압축 해제)
# astropedia 호스트가 죽어 있을 때 쓰는 CKAN 미러:
ROBBINS_URL="https://astrogeology.usgs.gov/ckan/dataset/f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/resource/c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/lunar_crater_database_robbins_2018"
if [ ! -f data/raw/robbins/lunar_crater_database_robbins_2018_bundle/data/lunar_crater_database_robbins_2018.csv ]; then
  curl -fSL --retry 3 -o data/raw/lunar_crater_database_robbins_2018.zip "$ROBBINS_URL"
  python -c "import zipfile; zipfile.ZipFile('data/raw/lunar_crater_database_robbins_2018.zip').extractall('data/raw/robbins')"
fi
for url in "${files[@]}"; do
  name="data/raw/$(basename "$url")"
  echo "[$(date +%T)] GET $url"
  curl -fSL -C - --retry 5 --retry-delay 10 -o "$name" "$url" 2>&1 | tail -1
  echo "[$(date +%T)] done $name ($(stat -c%s "$name" 2>/dev/null || echo '?') bytes)"
done
echo "ALL DONE"
