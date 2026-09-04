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
- 고전 베이스라인(PCA 외형 기저 템플릿, SLIM 계열 비DL 근사) 비교: 같은 val·같은 지표
  구현에서 mAP50-95 0.093 vs YOLO INT8 0.983, recall 0.256 vs 0.999, τ 218.8 vs 151.6 ms.
  측정 σ수평 460 vs 88 m·오검출률 0.151 vs 0.101 → **CEP 664.4 m vs 126.6 m(5.2배)**.
  τ를 서로 바꿔 넣어도 CEP 불변(둘 다 프레임 주기 1 s 미만 → 보상이 흡수) — 계열 격차는
  지연이 아니라 측정 품질로 들어온다. (p7c_det_compare, p7c_cep_compare, docs/classic_baseline.md)
- SLIM 정합성(plausibility): 측정 성공률 0.98 vs 14/14, σ수평 ≈85 m(GSD 대비 수 px),
  CEP 110.8 m vs ~10 m. 검증 아님(센서·자유도·측정 고도 구간 상이). (p7_slim_comparison.json)
- **P9 매트릭스(탐지기 5 × 등급 3 × 보상 2, n=200, 정밀 지도 가정)**: ① 차세대+보상은
  YOLO 4종 118~127 m 동급(τ<주기 → 모델·양자화 무차별), classic 664 m(측정 품질 지배)
  ② 현세대(GR740급 ×96)/구세대(HR5000급 ×127)는 τ 19~68 s → 측정 2~6개, CEP 520~2159 m
  — SW 탐지 TRN 성립 불가 ③ 미보상은 τ≥19 s 전 조건 ~2333 m 동일(게이트 전량 기각 =
  IMU 드리프트 바닥) ④ 같은 mAP 0.99가 등급 따라 118→1017 m — mAP 단독으로는 착륙
  성능을 못 정렬. mAP→σ는 수식이 아니라 탐지기별 개루프 보정 실측 사용.
  (p9_matrix.json, figs/p9_matrix_heatmap.png, p9_map_vs_cep.png)
- **P9 해석 규칙(칸 간 순위 주의)**: ① 차세대 YOLO 4칸(118~127 m)은 CI가 거의 전부
  겹침 — 순위 없음. ② 붕괴 영역(τ 19~68 s)에서 FP32/INT8 우위가 등급마다 뒤집히는
  것은 탐지기 품질이 아니라 **측정 스케줄 앨리어싱**: 직렬 1 Hz에서 τ가 촬영 격자와
  간섭해 밴드(100 s) 내 마지막 측정 시각 t_last가 τ에 따라 53~97 s로 튀고, CEP가
  t_last와 거의 단조 대응(t_last 96 s→567, 97→520 vs 75→820, 53→1017 m). 즉 붕괴
  영역의 칸 간 차이는 "τ 위상 운"이며 탐지기 비교로 읽으면 안 됨 — 유효한 결론은
  영역 수준(평탄/붕괴/바닥)뿐. (p9_matrix.json conditions의 tau_s·cep_m로 재현)
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

- 고전 vs AI 비교는 같은 폐루프에서 계열만 바꿔 mAP·CEP를 동시에 재는 방식이다. 위 표의
  선행연구 중 두 축을 함께 보고한 사례는 없다. (docs/classic_baseline.md)

## 8. 자주 나올 질문

- **합성-실사 도메인 갭**: 텍스처 100 m/px vs 카메라 GSD 25~34 m/px(3~5배 업샘플),
  태양각 randomization. 10월 NAC 패치·광학계 모델로 보완 계획.
- **데이터셋 커버리지**: 밴드 22~30 km를 1 Hz로 연속 커버, 궤적 반복은 태양각만 변경,
  접근 회랑(y=0) 한정. 이는 사전 측량 착륙지 특화 = TRN 설계 특성(과적합 아님).
  train/val=SLIM 유지 근거 + 남부 고지대 held-out 일반화 test로 뒷받침. (p5_det_highlands.json)
- **통계 111 m vs 실런 갭**: P8 실런 MC(n=200, wallclock τ)로 정량화 — 실런 CEP 297.8 m
  [284.7, 312.4] = 통계 110.8 m의 2.7배(P6 단일런 197.8 m은 이 분포 안의 한 표본).
  갭의 지배 성분은 분산이 아니라 **계통 바이어스**: 평균 오프셋 크기 290.0 m ≈ CEP,
  자기 중심 기준 R95/CEP 1.39(조밀) vs 통계 2.32(등방). iid 통계 모델이 만들 수 없는
  반복성 오프셋 — 후보: 합성 씬·카탈로그 정합 오차(맵타이 유사), 시간상관, IMU bias
  미모델(10월 분해 실험). 프레임 단위 PnP 산포(고도 bin 로버스트 σ 44~74 m)는 오히려
  데이터셋 iid(85~186 m)보다 작음 — 문제는 산포가 아니라 바이어스. 발산 1/200(seed 14,
  초기 연관 실패 = lost-in-space 한계 실증, 집계 분리). (p8_summary.json, p8_unity_mc.json)
- **바이어스의 정체와 보정(P8b)**: 진단(참값 pose 개루프, 태양각 2종 일치 → 조명 아님)
  결과 씬-카탈로그 정합 오차가 East를 따라 +60→−80→+120 m로 굽이침 — 변화율을 EKF가
  속도로 흡수 후 밴드 이탈 뒤 250 s coast에서 증폭(1.2 m/s × 250 s ≈ 290 m). East 10 km
  bin 사전 보정 테이블 적용(실제 임무의 pre-flight 맵타이 보정에 해당) 시 **CEP 297.8 →
  133.2 m**, 통계 110.8 m와 CI 인접 — 남은 갭은 잔여 정합(bin 조도)·시간상관.
  "iid로는 시간상관·계통 성분이 안 보인다 → 폐루프 실런 평가 필요"의 완결 실증.
  (p8_reg_diag.json, registration_correction.json, p8_unity_mc_corr.json, p8_summary.json)
- **정합 오차의 근본 원인**: Unity 배제 검사(figs/p8_texture_catalog_overlay.png,
  scripts/make_reg_overlay.py) — 원본 WAC 텍스처 위에 카탈로그 원을 직접 그리면
  크레이터 테두리와 1~4 px(100~400 m)씩 지역마다 다른 방향으로 어긋남. Robbins
  카탈로그와 WAC 모자이크는 서로 다른 파생 산물(카탈로그 위치 불확실성 수백 m급,
  모자이크 스트립별 정사보정 잔차)이라 공개 전역 산물 체인에서 예상되는 수준.
  실제 임무의 map-tie 오차 항에 대응하며, 임무들은 착륙지 전용 고해상 지도
  (예: Mars 2020 LVS의 사전 제작 기준지도)로 이 항을 줄인다 — 본 연구의 10월
  NAC 재정합 계획이 그 대응. 탐지기는 화면의 시각적 테두리를 따르고 PnP는
  카탈로그 좌표에 고정되므로, 지도-영상 불일치가 프레임 공통 오프셋으로 측정에 유입.
- **데모 오버레이 원 어긋남**: 원 = EKF 예측 포즈 투영(=항법 예측 오차, EKF가 보정하는
  대상). 탐지 중심은 참값 라벨 대비 median 0.52 px·p95 1.3 px.
- **유도 한계**: 무제약 ZEM/ZEV(추력 하한·활공각·질량감소 미고려), 변인 통제 목적으로
  모든 조건에 동일 고전 유도 고정. (docs/limitations.md §2)
- **왜 EKF? 모델이 선형인데(KF와 동일)**: 맞다. 동역학은 g 상수·F 상수 행렬로 선형이고,
  카메라 투영의 비선형성은 PnP가 필터 밖에서 흡수해 z=r_PnP ∈ ℝ³ 선형 유사측정이 된다
  (loosely-coupled). 따라서 현 구현은 자코비안 선형화가 없는 선형 KF와 수학적으로 동일
  (sim/ekf.py docstring에 명기). EKF 명칭은 계약 §2.4·TRN 문헌 관례이며, 10월 확장
  (자세·IMU bias 상태, 픽셀 직접 측정 tightly-coupled 시 H=투영 자코비안)에서 실제 EKF가 된다.

## 9. 산출물 대응 (연구계획서 예상 결과 ①~⑤)

| # | 산출물 | 상태 |
|---|---|---|
| ① | 라벨 자동 생성 합성 데이터셋 | ✅ 1000프레임 (공개 예정) |
| ② | 경량·양자화 탐지 + FP32/INT8 벤치 | ✅ p5_det.json, tau_*.json |
| ③ | 폐루프 TRN 시뮬레이터 코드 | ✅ 저장소 공개 |
| ④ | 추론 지연·오검출률 대비 착륙 오차 곡선 | ✅ p7b_tau_serial, p7b_fp_sweep |
| ⑤ | 폐루프 착륙 시연 영상 | ✅ figs/slides/display{2,3,4}_*.mp4 |
