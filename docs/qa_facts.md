# Q&A 사실집 (docs/qa_facts.md)

발표 심사 대비. **코드·config·results에서 확인 가능한 사실만.** 추정·해석은 넣지 않는다.
숫자는 docs/results_summary.md(= results/*.json 자동 추출)와 동일 출처. 구 assumed
수치(81.4 m 등)는 사용 금지(STATUS 2026-09-01 정정).

## 1. 시스템·좌표계

- 파이프라인: Python 3-DOF(RK4) → Unity 렌더 → YOLO11n INT8 → 연관·PnP → EKF(지연 보상
  +χ² 게이트) → ZEM/ZEV → 추력. (CLAUDE.md §1, sim/loop.py)
- 좌표계 L = 착륙 목표점 원점 ENU. site = SLIM 착륙점(LROC 측정 13.3160°S, 25.2510°E).
  달 중력 1.62 m/s², R_moon 1737.4 km. (config.yaml, CLAUDE.md §2.1)
- 카메라: 핀홀 1024×1024, 수직 FOV 60°, f = H/(2·tan(θv/2)) = 886.8 px, 1 Hz(스윕 시 가변).
  (config.yaml camera, perception/camera.py)

## 2. 데이터셋·탐지

- 원본: SLDEM2015 DEM(≈59 m/px), LROC WAC 모자이크(100 m/px), Robbins 카탈로그.
  box 340×60 km, D≥1 km 카탈로그 566개. (STATUS P3, data/processed/catalog_L.csv)
- 데이터셋 1000프레임, 궤적 구간 단위 train/val 분리(프레임 무작위 분리 금지),
  태양 고도 10–60°·방위 0–360° randomization. val 112프레임. (config dataset, p5_det.json)
- 학습: yolo11n 100 epoch, imgsz 1024. mAP50-95 FP32 0.994 / INT8 0.980
  (mAP 손실 미미). s 모델도 학습(비교용). (p5_det.json, p5_det_s.json)
- INT8 PTQ: 실렌더 이미지 캘리브레이션. (scripts/export_int8.py)

## 3. 추론 지연 τ (실측, 1스레드 CPU / GPU)

| 조건 | median τ |
|---|---|
| n INT8 CPU | 195.7 ms |
| n FP32 CPU | 219.0 ms |
| s INT8 CPU | 376.5 ms |
| s FP32 CPU | 536.6 ms |
| n INT8 GPU (TensorRT) | 54.5 ms |

- GPU에선 INT8/FP16이 τ를 못 줄임(병목이 전·후처리). CPU(온보드 프록시)에서 INT8 유효.
  (results/tau_*.json, STATUS P2)
- τ_det 정의 = 전처리+추론+NMS. warmup 50, n_iter 1000. (perception/bench_tau.py)

## 4. 온보드 컴퓨트 맥락

- CPU 1스레드 벤치(CoreMark 34,876 / Dhrystone 40,654 DMIPS, 친화도 1코어, zig cc -O2):
  JAXA HR5000(SLIM 세대 200 MHz MPU, 320 DMIPS)의 **127배**, RAD750의 102배, NASA HPSC
  칩 목표(~100×RAD750)와 같은 자릿수. → 우리 τ 실측 = 차세대 온보드 프록시.
  (p7b_cpu_bench.json, docs/onboard_cpu_emulation.md)
- SLIM의 크레이터 탐지·매칭은 **CNN이 아니라** Microsemi RTG4 FPGA의 고전 방식(PCA 외형
  기저 탐지 + 삼각형 유사 매칭 TSM), 촬영→결과 ≤5 s. RTG4 DSP 피크 ~230 GOPS.
  YOLO11n INT8을 RTG4급에 올릴 때 τ ≈ 0.3~1.5 s(문헌 기반 1차 추정 — 검증값 아님).
  (docs/onboard_cpu_emulation.md §1b, 출처: Acta Astronautica Vol.226 2025)

## 5. 항법·필터·유도

- EKF 6-state [r;v], IMU 100 Hz 예측(σ_a 0.005 m/s²·bias 없음), 측정 도착 시 보정.
  지연 보상 = t_c 스냅샷 되감기 → 보정 → IMU 재전파(sim/ekf.py delayed_update).
- χ² 게이트 d²>11.345(χ²₃ 0.99) 기각. 연속 5회 기각 시 P 4배 팽창(예산 3회/비행).
  (config ekf, sim/ekf.py)
- 측정 보정(개루프, val 프레임): n INT8 σ [87.9, 81.8, 32.9] m, 오검출률 0.101,
  오프셋 381.4 m. s INT8 σ [82.2, 88.2, 30.8]·0.082 — n과 동급(측정 품질 통제).
  (measurement_model{,_s}.json)
- 유도 ZEM/ZEV, t_go=T_f−t 고정, a_max 6 m/s² 포화. (config scenario, sim/guidance.py)
- TRN 밴드 22–30 km. 하한은 iid 통계(17 km)가 아니라 실런 검증으로 확정 — 폐루프
  평가가 설계값을 바꾼 사례. 밴드 밖은 IMU 단독 전파. (config trn_band, STATUS)

## 6. 결과 (calibrated, n=200/조건, 직렬 촬영 모델)

- P6 Unity 실런 폐루프: 착륙 197.8 m, 연착륙 0.65 m/s, 사슬 τ 실측 133 ms. (p6_closed_loop.json)
- CEP vs τ: 보상 시 τ≤프레임 주기(1 s) 110.8 m 평탄 → 2 s 177.8 / 5 s 250.4 m(드롭).
  미보상 n INT8 159.9 → s FP32 323.0 → 5 s 2160.5 m. (p7b_tau_serial{,_compoff}.json)
- 오검출률 곡선(산출물 ④): fp 0→0.3에서 112.7→150.8 m, 게이트 기각률 0.009→0.319.
  (p7b_fp_sweep.json)
- 보상 성립 조건: 타임스탬프 1σ ≤20 ms 무해, 200 ms(≈τ)면 194.9 m로 미보상보다 악화,
  500 ms 349.5 m·게이트 붕괴. 카메라 5 Hz에서 n INT8(48.7 m, 드롭0) vs n FP32(65.7 m,
  드롭50%) CI 분리 — 보상 켠 채 양자화가 CEP를 가름. (p7b_jitter_sweep.json, p7b_rate_sweep.json)
- SLIM 정합성(plausibility): 측정 성공률 0.98 vs 14/14, σ수평 ≈85 m(GSD 대비 수 px),
  CEP 110.8 m vs ~10 m. 검증 아님(센서·자유도·측정 고도 구간 상이). (p7_slim_comparison.json)
- MC 500회 이상은 10월 본실험(계획서 2단계). 현재 n=200은 결선용 예비.

## 7. 신규성·선행연구

- 기여 4가지: ① 공개 폐루프 테스트베드(인식→항법→유도, results 재현) ② INT8이 mAP
  소폭 손실에도 측정 σ·오검출률·CEP 불변 ③ 지연 자체는 보상으로 흡수되고 성능에
  들어오는 경로는 측정률·타임스탬프 오차·오검출률(5 Hz에서 양자화가 CEP를 가름)
  ④ 폐루프가 설계값을 바꾼 사례(밴드 하한 17→22 km, 게이트 연쇄 기각→P 팽창).
- 합성 렌더 학습은 신규성으로 주장하지 않음(ESA PANGU, DeepMoon(Silburt 2019),
  Airbus SurRender의 확립된 방법론). 크레이터 항법 실적: Mars 2020 LVS, SLIM(고전 방식),
  DLR ATON, McLeod 2026(AI 크레이터 항법).

| 선행연구 | τ 취급 | 종단 평가점 | 폐루프 | 공개 |
|---|---|---|---|---|
| Streaming perception (2020) | 지연 인지 지표 제안 | 탐지 정확도 | 아니오 | 예 |
| Planner-centric metrics (2020) | — | 계획기 영향 | 부분 | 예 |
| DeepMoon / Silburt (2019) | — | 탐지 정밀도·재현율 | 아니오 | 예 |
| DLR ATON (2018) | — | 항법(위치·자세) 오차 | 비행시험 | 아니오 |
| SLIM VBN (Acta 2025) | 처리 ≤5 s | 착륙(비행 실증) | 예(실기) | 아니오 |
| McLeod 2026 (Astrodynamics) | — | 항법(매핑) 오차 | 시뮬 | 부분 |
| Candan & Servadio (arXiv 2606.14776) | — | 항법 오차(개루프) | 아니오 | 예 |
| **본 연구** | **FP32/INT8 실측 τ 주입** | **착륙 CEP** | **예(시뮬)** | **예** |

## 8. 자주 나올 질문

- **합성-실사 도메인 갭**: 텍스처 100 m/px vs 카메라 GSD 25~34 m/px(3~5배 업샘플),
  태양각 randomization. 10월 NAC 패치·광학계 모델로 보완 계획.
- **데이터셋 커버리지**: 밴드 22~30 km를 1 Hz로 연속 커버, 궤적 반복은 태양각만 변경,
  접근 회랑(y=0) 한정. 이는 사전 측량 착륙지 특화 = TRN 설계 특성(과적합 아님).
  train/val=SLIM 유지 근거 + 남부 고지대 held-out 일반화 test로 뒷받침. (p5_det_highlands.json)
- **통계 111 m vs 실런 198 m 갭**: 시간상관 편향(iid 미포함)·IMU 바이어스 없음(TODO 10월).
- **데모 오버레이 원 어긋남**: 원 = EKF 예측 포즈 투영(=항법 예측 오차, EKF가 보정하는
  대상). 탐지 중심은 참값 라벨 대비 median 0.52 px·p95 1.3 px.
- **유도 한계**: 무제약 ZEM/ZEV(추력 하한·활공각·질량감소 미고려), 변인 통제 목적으로
  모든 조건에 동일 고전 유도 고정. (docs/limitations.md §2)

## 9. 산출물 대응 (연구계획서 예상 결과 ①~⑤)

| # | 산출물 | 상태 |
|---|---|---|
| ① | 라벨 자동 생성 합성 데이터셋 | ✅ 1000프레임 (공개 예정) |
| ② | 경량·양자화 탐지 + FP32/INT8 벤치 | ✅ p5_det.json, tau_*.json |
| ③ | 폐루프 TRN 시뮬레이터 코드 | ✅ 저장소 공개 |
| ④ | 추론 지연·오검출률 대비 착륙 오차 곡선 | ✅ p7b_tau_serial, p7b_fp_sweep |
| ⑤ | 폐루프 착륙 시연 영상 | ✅ figs/slides/display{2,3,4}_*.mp4 |
