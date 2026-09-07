# lunar-trn 합성 달 하강 크레이터 데이터셋 v1 (2026-09-07)

Synthetic lunar descent crater-detection dataset for TRN (Terrain Relative Navigation)
research. Auto-labeled renders of a lunar landing approach corridor.

이 문서는 배포 아카이브(`lunar-trn-dataset-v1.zip`)에 동봉된 README와 동일 내용 +
배포 정보다. 아카이브 본체는 용량(1.3 GB) 때문에 저장소 밖에서 배포한다.

## 배포

- 아카이브: `lunar-trn-dataset-v1.zip` (1,334.8 MB, 2,602 files)
- SHA256: `192E91051B654B3E2E8DBD1C16DD983102A64E52EF87A7CDB760F3742ACC86BD`
- 다운로드 링크: TBD (Google Drive 공개 링크 업로드 후 기입)
- 생성 시점 저장소 커밋: 7946504

## 구성

| 폴더 | 프레임 | 용도 |
|---|---|---|
| `dataset/` | 1000 (train 888 / val 112) | 학습·평가 본 세트 (SLIM 착륙지 인근) |
| `dataset_highlands/` | 297 | 남부 고지대 held-out 일반화 평가 세트 (학습 미사용) |

각 폴더:

- `images/{train,val}/*.png` — 1024×1024 렌더 (수직 FOV 60°, nadir 카메라)
- `labels/{train,val}/*.txt` — YOLO 형식 `0 cx cy w h` (정규화, 단일 클래스 crater)
- `poses.csv` — 프레임별 `traj_id, frame_id, t, x, y, z, sun_az_deg, sun_el_deg, split`
  (위치는 착륙 목표점 원점 ENU [m])
- `dataset.yaml` — ultralytics 학습용 (경로는 상대경로 `.`)

## 생성 방법

- 지형: SLDEM2015 DEM + LROC WAC 100 m/px 전역 모자이크를 Unity로 렌더.
- 궤적: TRN 측정 밴드(고도 22–30 km) 하강 접근 회랑을 1 Hz로 연속 촬영,
  같은 궤적을 태양각만 바꿔 반복(domain randomization: 방위 무작위, 고도 10–60°).
- 라벨: Robbins 크레이터 카탈로그(직경 D ≥ 1 km)를 핀홀 카메라 모델로 투영한
  원의 외접 사각형. 포함 조건: 중심이 화면 안, 투영 직경 ≥ 12 px, bbox의 50% 이상
  화면 안(경계 클리핑).
- 결정론 재생성: 저장소에서 `python scripts/make_dataset.py --config config.yaml --seed 0`
  (Unity 렌더 서버 필요 — `unity/README.md` 참조). 같은 시드면 같은 데이터셋이 나온다.

## 원천 데이터 출처 (public domain, NASA/PDS)

- SLDEM2015: Barker et al. (2016), LOLA(NASA)/Kaguya(JAXA) 병합 DEM.
- LROC WAC global mosaic: NASA/GSFC/Arizona State University.
- Lunar crater database: Robbins, S. J. (2019), JGR Planets.

렌더 영상과 라벨은 위 공개 데이터의 파생물이며, 연구·교육 목적으로 출처 표기와 함께
자유롭게 사용할 수 있습니다.

## 알려진 한계

- 텍스처 해상도 100 m/px를 카메라 GSD 19–34 m/px로 업샘플(3–5배) — 실사 대비 저주파.
- 라벨은 카탈로그 전수(열화 크레이터 포함) 기준 — 시각적으로 안 보이는 GT가 존재.
- 조명은 태양 고도 10–60°만 커버(극지 저조도 미포함).
- WAC 모자이크와 Robbins 카탈로그 간 정합 오차(수백 m급)가 라벨 위치에 유입될 수 있음
  (`docs/limitations.md` 참조).
