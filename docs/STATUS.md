# STATUS

## 자율 진행 세션 (2026-08-26 밤) — P2 완료, P3~P7 코드 선행

사용자 부재 중 자동 진행. 실행이 막힌 단계는 코드·문서만 작성하고 blocker를 명시한다.

- **P3 (코드 완료, 실행 대기)**: perception/camera.py(project/backproject, §2.2 완성),
  perception/labeler.py(§2.3 포함 조건·클리핑), data/catalog.py(등장방형 변환 + Robbins CLI),
  data/crop.py(DEM/텍스처 크롭·리샘플·Unity RAW), scripts/analyze_altitude_band.py.
  테스트 15개(camera 4, labeler 5, catalog 4, tau_sampler 2(+skip 1)) 통과.
  **blocker**: data/raw/ 원본(SLDEM 타일, WAC 모자이크, Robbins CSV)과 config site lat0/lon0 확정 —
  사용자가 채우면 `data/crop.py → data/catalog.py → analyze_altitude_band.py` 순서로 실행.
- **P4 (파일 작성 완료, GUI 대기)**: unity/Assets/Editor/SceneBuilder.cs("LunarTRN/Build Scene"),
  unity/Assets/Scripts/RenderServer.cs(TCP, L→Unity 매핑, 길이 접두 PNG 프로토콜),
  unity/client.py(재시도·타임아웃), scripts/check_projection.py(Hough 5크레이터 픽셀 오차),
  unity/README.md(번호 절차 + 흔한 오류 7종). **blocker**: Unity Editor 수동 작업(README 절차).
- **P5 (핵심 코드 완료, 나머지 대기)**: perception/associate.py(게이트·모호성 기각),
  perception/pnp.py(RANSAC+LM 정밀화; 무잡음 <1e-3 m, 이상치 30% 복원 테스트 통과),
  perception/detect.py(torch/ORT/TRT 래퍼), scripts/make_dataset.py·train.py·export_int8.py·
  eval_det.py·calibrate_measurement.py 구현(Unity 서버 필요, 미실행).
- **P6 (코드 완료, 미실행)**: sim/measurement.py UnityMeasurementModel(렌더→탐지→연관→PnP,
  τ wallclock 실측), sim/loop.py measurement="unity" + run_closed_loop CLI --detector/--frames-dir.
- **P7 (예비 실행 완료)**: scripts/aggregate_mc.py(명령 세트 출력 + 집계·그림).
  assumed measurement stats, τ 9지점(스윕 6 + 실측 median 3) × 보상/미보상, n=200/조건 (총 3,600회):
  보상 시 CEP 31.1~31.6 m로 τ∈[0.02, 5] s 전 구간 평탄. 미보상 시 τ 비례 증가 —
  실측 τ 지점 기준: TensorRT INT8(20 ms) 33.5 m, ORT CPU INT8(179 ms) 89.9 m,
  ORT CPU FP32(207 ms) 103.1 m; τ=5 s에선 1776 m.
  → results/p7_mc_*.json(18), results/p7_mc.json, figs/p7_cep_vs_tau.png, p7_scatter_*(18).
  P5 보정 후 calibrated 통계로 재실행 필요("preliminary" 표기 유지).
- 테스트: `pytest -q` 47 passed (11.7 s)

## P2 · τ 벤치마크 (2026-08-26)

- 완료: perception/bench_tau.py — ONNX export(yolo11n 사전학습, imgsz 1024, batch 1),
  INT8 static QDQ 양자화(백엔드별 변형: ORT CPU는 U8S8 비대칭, TRT는 S8S8 대칭+bias FP,
  탐지 헤드 model.23 제외 — 헤드 양자화 시 탐지 소멸), TensorRT 엔진(FP32 / explicit QDQ INT8),
  τ_det = 전처리+추론+NMS 프로토콜(warmup 50, n_iter 1000), sanity(IoU), docs/bench_protocol.md
- 테스트 결과: `pytest -q` 43 passed. tests/test_tau_sampler.py — empirical 스키마 로드 검증
- 핵심 수치 (RTX 5060 Ti / Ryzen 5 5600X 1스레드, 캘리브레이션 "temporary"):
  | 백엔드 | median | p95 | sanity IoU |
  |---|---|---|---|
  | TensorRT FP32 | 20.55 ms | 22.56 ms | — |
  | TensorRT INT8 | 20.02 ms | 22.25 ms | 0.949 ✔ |
  | ORT CPU FP32 | 206.81 ms | 211.65 ms | — |
  | ORT CPU INT8 | 179.39 ms | 190.31 ms | 0.922 ✔ |
- 생성 파일: results/tau_{trt,ort_cpu}_{fp32,int8}.json, figs/p2_tau_hist.png, logs/p2_bench.log
- 특이사항: ① PyPI tensorrt 대신 tensorrt-cu12 11.2.1.2 사용 중 — TRT 11은 implicit INT8
  캘리브레이션 API가 제거되어 explicit QDQ 경로로 구현 ② TRT INT8 제약: 대칭 zero-point만,
  Int32 bias DQ 불가, attention(model.10) 0차원 상수 Q/DQ 불가 → 해당 부분 제외
  ③ 캘리브레이션이 임시 이미지(ultralytics 샘플 증강)이므로 P5에서 실제 렌더 이미지로 재실행 필요
- 다음 단계: P3 데이터 파이프라인 (원본 데이터·site 확정 대기)

## P1 · 시뮬 코어 (2026-08-26)

- 완료:
  - sim/dynamics.py `rk4_step`, sim/guidance.py `zem_zev`(방향 유지 포화, t_go_min 유지는 loop 담당)
  - sim/ekf.py `EKF`: predict/update(χ² 게이트)/`delayed_update`(링버퍼 복원→보정→재전파, 중첩 지연 시 버퍼 스냅샷 갱신)
  - sim/measurement.py `StatMeasurementModel`(σ_xyz 가우시안 + fp_rate 무작위 방향 fp_offset 이상치), `TauSampler`(constant|empirical), `measurement_R`(calibrated 파일 없으면 assumed)
  - sim/loop.py `run_closed_loop`(measurement: stat|truth|unity(P6 예약)), sim/mc.py(CEP·95% 타원·부트스트랩 CI·result_meta)
  - scripts/run_closed_loop.py, sweep_tau.py, run_mc.py (--config --seed --out CLI)
- 테스트 결과: `pytest -q` 27 passed (11.6 s). RK4 vs 해석해 <1e-6 m, 참값+ZEM/ZEV 착륙 오차 0.0013 m·속도 0.077 m/s·|a_T|≤a_max,
  NEES 50 seed 시간평균이 χ²₆ 95% 구간 내, 지연 보상 τ=1 s 무지연과 1e-6 이내 일치·미보상 |v|τ 편향 재현, 게이트 10σ 기각/1σ 통과
- 생성 파일: results/p1_closed_loop.json, results/p1_tau_sweep.json, figs/p1_trajectory.png, figs/p1_est_error.png, figs/p1_tau_sweep.png, logs/p1_tau_sweep.log
- 핵심 수치(assumed measurement stats, n=50/점): 지연 보상 시 CEP ≈ 34 m로 τ∈[0.05,5] s 전 구간 평탄,
  미보상 시 τ=0.05 s→42 m, 0.5 s→229 m, 5 s→1763 m (τ 비례 증가) — 연구 주장(τ 영향 정량화) 골격 확인
- 특이사항: PyYAML이 config의 `1.0e6`을 문자열로 읽음(YAML 1.1 형식 제약) → 코드에서 float 변환으로 처리
- 수동 작업 필요: 없음
- 다음 단계: P2 — perception/bench_tau.py (TensorRT/ORT FP32·INT8 τ 벤치마크, results/tau_*.json)

## P0 · 환경·뼈대 (2026-08-26)

- 완료:
  - CLAUDE.md §5 레이아웃 디렉터리·스텁 모듈 생성 (파일마다 docstring 1줄)
  - `.venv` (Python 3.12.10) + torch 2.11.0+cu128 / torchvision 0.26.0+cu128 — RTX 5060 Ti가 sm_120으로 인식됨
  - ultralytics 8.4.129, tensorrt 11.2.1.2 (cu12), onnxruntime 1.29.0, openvino 2026.3.0, opencv 5.0.0, rasterio 1.5.1, scipy, matplotlib, pyyaml, pytest 설치
  - `perception/camera.py` `K_cam(cfg)` 구현 (f = H/(2·tan(θ_v/2)))
  - `scripts/check_env.py` → `results/env.json` (GPU sm_120, 13개 패키지 import 전부 OK)
  - .gitignore, README.md, pytest.ini, requirements.txt
- 테스트 결과: `pytest -q` 17 passed (0.2 s)
- 생성 파일: sim/·perception/·data/·unity/·scripts/·tests/ 스텁 일체, results/env.json
- 특이사항: PyPI `tensorrt` 메타패키지(sdist)가 빌드 격리 안에서 1.6 GB 휠을 받다 멈춤(1시간+ 정지 2회).
  pypi.nvidia.com에서 `tensorrt_cu12_bindings`/`tensorrt_cu12_libs` 휠을 curl로 직접 받아 설치 후
  `tensorrt-cu12`를 `--no-deps`로 설치하여 해결. requirements.txt에 절차 주석 있음.
- 수동 작업 필요: 없음
- 다음 단계: P1 — sim/dynamics.py(RK4) + sim/guidance.py(ZEM/ZEV) 구현, 참값 상태 폐루프로
  scenario 기준 1 m 이내 착륙·|a_T| ≤ a_max 실현가능성 테스트
