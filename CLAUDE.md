# lunar-trn — CLAUDE.md

온보드 경량 AI 크레이터 탐지 기반 달 착륙 TRN(Terrain Relative Navigation, 지형상대항법) 폐루프 시뮬레이터.
핵심 주장 한 문장: **AI 탐지기를 mAP가 아닌 착륙 오차 분포(CEP)로 평가한다.**
추론 지연 τ와 오검출률이 착륙 오차 분포에 미치는 영향을 몬테카를로로 정량화하는 것이 목적이다.

## 0. 이 파일의 지위
- 모든 세션에서 유효한 규칙이다. 세션 프롬프트(docs/PROMPTS.md)와 충돌하면 이 파일이 우선한다.
- §2 계약(Contract)은 Claude가 수정하지 않는다. 수정이 필요하면 `CONTRACT CHANGE PROPOSAL:`로 시작하는 제안만 출력하고 멈춘다.
- 숫자 파라미터는 전부 config.yaml. 이 파일에는 규칙과 정의만 둔다.
- 원본 참조 문서: docs/ref/온보드 경량 AI 크레이터 탐지 기반 달 착륙 지형상대항법(TRN)의 폐루프 성능 정량화.pdf (연구 범위의 계약 원본), docs/ref/[붙임] KASA 우주항공 학술 경연대회 대학부 모집공고.pdf (대회 규정 원본). 세션 중 필요하면 @docs/ref/ 로 그때만 불러온다. 매 세션 자동 로드하지 않는다.

## 1. 범위 (변경 금지)
- 파이프라인: Python 3-DOF 동역학(참값, RK4) → Unity 렌더(센서 모사 전용) → YOLO 계열 경량 탐지 + INT8 PTQ → 카탈로그 매칭·PnP → EKF(지연 보상 + χ² 게이팅) → ZEM/ZEV 유도 → 추력 명령
- 데이터: SLDEM2015 DEM, LROC WAC 100 m 모자이크, Robbins 카탈로그(D ≥ D_min), 투영 자동 라벨, 태양각 domain randomization
- 평가: Tier 1 폐루프(측정 통계 보정) → Tier 2 몬테카를로(CEP, 95% 오차 타원), FP32 vs INT8
- 결선(9/12) 전 허용되는 최소 구현. 해당 코드에 `# TODO(oct):` 표시:
  - 데이터 연관: EKF 예측 pose로 카탈로그를 투영한 뒤 최근접 연관 (기하 불변량 lost-in-space 매칭은 10월)
  - t_go = T_f − t 고정 (최적 t_go 탐색은 10월)
  - 카메라 자세 nadir 고정 (3-DOF에는 자세 상태가 없음)
  - EKF 6-state, IMU bias 상태 없음 (10월 검토)
- 위 목록 밖의 단순화·대체(다른 렌더러, 다른 탐지기 계열, 다른 필터)는 하지 말고 물어본다.

## 2. 계약 (Contract) — 좌표계·기호·인터페이스

### 2.1 좌표계·단위
- 모든 내부 값은 SI(m, s, rad). 각도는 config·CLI 입출력에서만 deg.
- L 프레임: 착륙 목표점 원점 ENU(East, North, Up). 시뮬·EKF·유도·카탈로그·PnP 출력 전부 L.
- 달 중력 g = [0, 0, −1.62] m/s² 상수. 달 반경 R_m = 1737.4 km.
- 카탈로그 lat/lon → L: 등장방형 평면 근사. x = R_m·(lon − lon0)·cos(lat0), y = R_m·(lat − lat0), z = DEM(lat, lon) − DEM(lat0, lon0). 시뮬·라벨러·Unity가 같은 근사를 쓴다(내부 일관성이 목적).
- Unity 좌표계 변환은 unity/Assets/Scripts/RenderServer.cs 안에서만. 매핑: Unity x = East, y = Up, z = North. L 원점 = Unity 월드 (0, 0, 0).
- 태양각: sun_az는 북에서 시계방향(deg), sun_el은 지평선 기준 고도각(deg).

### 2.2 카메라
- 핀홀. 해상도 W×H, 수직 FOV θ_v. f = H / (2·tan(θ_v/2)), c_x = W/2, c_y = H/2.
- 내부 파라미터 행렬 K_c = [[f, 0, c_x], [0, f, c_y], [0, 0, 1]]. 칼만 이득 K와 이름을 섞지 않는다.
- 카메라 프레임 C: x_C = East, y_C = South, z_C = Down. R_{C←L} = diag(1, −1, −1). 카메라 원점 = 착륙선 위치 r.
- 투영: p_C = R_{C←L}·(p_L − r), u = f·x_C/z_C + c_x, v = f·y_C/z_C + c_y. z_C ≤ 0인 점은 무효.
- 이 규칙은 perception/camera.py 한 곳에만 구현한다. 라벨러·PnP·Unity 검증은 모두 그 모듈을 import한다.

### 2.3 카탈로그·라벨
- data/processed/catalog_L.csv 열: id, x, y, z, D (m). D ≥ D_min만.
- 라벨은 YOLO txt `0 cx cy w h`(정규화). bbox = 투영된 원(직경 D)의 외접 사각형(nadir 가정). 포함 조건: 중심이 화면 안, 투영 직경 ≥ p_min px, bbox의 50% 이상이 화면 안(경계로 클리핑).
- 이미지 `{traj_id}_{frame_id:05d}.png`, 같은 stem의 txt. 프레임별 pose·태양각은 poses.csv(t, x, y, z, sun_az_deg, sun_el_deg).

### 2.4 상태·측정·필터
- 참값 상태 x = [r; v] ∈ ℝ⁶ (r 위치, v 속도, L). ṙ = v, v̇ = g + a_T. RK4, Δt_IMU = 1/f_IMU.
- IMU: a_IMU = a_T + n_a, n_a ~ N(0, σ_a² I₃) (샘플당 백색잡음, bias 없음).
- EKF 추정 x̂ = [r̂; v̂], 공분산 P. 예측은 IMU 주기, 보정은 측정 도착 시.
  - F = [[I₃, Δt·I₃], [0₃, I₃]], G = [[½Δt²·I₃], [Δt·I₃]], Q = G·σ_a²·Gᵀ.
- 측정 z = r_PnP ∈ ℝ³ (L 위치). 촬영시각 t_c, 도착시각 t_c + τ. valid 플래그(대응쌍 ≥ 4, RANSAC 인라이어 ≥ n_min).
- 측정 잡음 공분산 R = diag(σ_x², σ_y², σ_z²). results/measurement_model.json이 있으면 로드, 없으면 config 가정값을 쓰고 모든 그림 제목에 "assumed measurement stats"를 넣는다.
- 보정: ν = z − H·x̂⁻, S = H·P⁻·Hᵀ + R, K = P⁻·Hᵀ·S⁻¹, x̂ = x̂⁻ + K·ν, P = (I − K·H)·P⁻. H = [I₃ 0₃].
- 게이트: d² = νᵀ·S⁻¹·ν. d² > χ²₃(0.99) = 11.345 이면 기각하고 로그에 남긴다.
- 지연 보상: (t, x̂, P, a_IMU) 링버퍼 보관 → 측정 도착 시 t_c 시점 상태로 되돌려 보정 → 버퍼의 IMU 입력으로 현재 시각까지 재전파. 미보상 옵션은 비교 실험용으로만 유지.
- TRN 측정은 고도 구간 [h_min, h_max] 안에서만 생성. 밖은 IMU 전파만.
- τ 샘플러 모드: constant | empirical(results/tau_*.json의 samples_s 리샘플). 스윕은 constant.

### 2.5 유도
- ZEM/ZEV, t_go = T_f − t. ZEM = r_f − (r̂ + v̂·t_go + ½·g·t_go²), ZEV = v_f − (v̂ + g·t_go).
- a_T = (6/t_go²)·ZEM − (2/t_go)·ZEV. |a_T| > a_max이면 방향 유지 포화. t_go < t_go_min이면 직전 명령 유지.
- r_f = 0, v_f = 0 (L 원점 연착륙). 착륙 판정 r_z ≤ 0 (평면 착륙, 지형 충돌 미고려). 착륙 오차 = 착륙 시점 (r_x, r_y), 착륙 속도도 기록.

### 2.6 기호 충돌 금지
K 칼만 이득 / K_c 카메라 내부 행렬 / P 추정 공분산 / R 측정 잡음 공분산 / R_m 달 반경 / R_{C←L} 회전 / τ 지연 / t_c 촬영시각 / t_go 잔여 비행시간 / ν 혁신(innovation) / S 혁신 공분산.
코드 변수명도 따른다: K_gain, K_cam, P_cov, R_meas, R_moon, R_cam_from_L, tau, t_c, t_go, nu, S_inn.

## 3. 코드 규칙
- Python ≥ 3.11. numpy, scipy, opencv-python, ultralytics, pytest, matplotlib, pyyaml, rasterio. 타입힌트. 함수당 한 가지 일. 모듈은 §5 레이아웃.
- 모든 스크립트는 argparse CLI + `--config config.yaml` + `--seed` + `--out`. 같은 인자면 같은 결과(결정론). rng는 np.random.default_rng(seed)를 인자로 전달, 전역 시드 금지.
- 매직 넘버 금지. 숫자는 config.yaml에서만.
- 그림 → figs/, 수치 → results/*.json. 모든 results json에 meta: git_hash, config_hash, timestamp, hardware. 문서·슬라이드의 숫자는 results/에서만 가져온다. 스크립트가 만들지 않은 숫자는 어디에도 쓰지 않는다. 미정은 "TBD".
- 테스트 우선: 새 모듈은 tests/에 테스트를 먼저 쓰고 통과할 때까지 구현. `pytest -q` 전체 60초 이내, 느린 것은 `@pytest.mark.slow`.
- 긴 작업(학습, 대량 렌더, MC)은 tmux/nohup으로 돌릴 스크립트로 만들고 logs/에 로그, 체크포인트 지원. 세션 안에서 완료를 기다리지 않는다. 실행 명령을 출력하면 내가 돌린다.
- Unity: unity/ 아래 C#·Python 클라이언트·README만 작성. Editor GUI 작업은 unity/README.md에 사람이 따라 할 수 있게 번호 매겨 적는다.
- 공개 저장소 전제: 사람 이름, 소속, 로컬 절대경로 금지. data/raw, runs/, logs/, results/large/, unity/Library, unity/Temp, *.engine은 .gitignore.
- 커밋은 단계 끝마다 `P<n>: <요약>`. 히스토리 재작성 금지.

## 4. 세션 규칙
- 시작: 이 파일 읽기 → `pytest -q` → 현재 상태 5줄 요약 → 이번 세션 계획 10단계 이내 → 진행.
- 멈추고 물어볼 때: 데이터 삭제, §2 계약 변경, §1 범위 밖 대체, 대용량 외부 다운로드, config.yaml의 시나리오·고도 구간 값 변경.
- 시간 상자: 같은 오류로 2시간(또는 시도 5회) 진전이 없으면 멈추고, 범위 안에서 가장 단순한 대안을 제안한다.
- 끝: docs/STATUS.md 갱신(완료 / 테스트 결과 / 생성 파일 / 수동 작업 필요 / 다음 단계) → 커밋.
- 일정(참고): 9/3(목) 정오 Unity 판정, 9/6(일) 저녁 Plan A/B 판정, 9/10(목) 슬라이드 동결, 9/12(토) 결선. 신규 기능은 9/4까지.

## 5. 레이아웃
```
lunar-trn/
  CLAUDE.md  config.yaml  README.md  requirements.txt
  sim/         dynamics.py  guidance.py  ekf.py  measurement.py  loop.py  mc.py
  perception/  camera.py  labeler.py  detect.py  associate.py  pnp.py  bench_tau.py
  data/        crop.py  catalog.py  raw/(gitignore)  processed/
  unity/       Assets/Editor/SceneBuilder.cs  Assets/Scripts/RenderServer.cs  client.py  README.md
  scripts/     check_env.py  run_closed_loop.py  sweep_tau.py  run_mc.py  make_dataset.py
               train.py  export_int8.py  eval_det.py  calibrate_measurement.py
               analyze_altitude_band.py  check_projection.py  make_slide_assets.py
  tests/  docs/(PROMPTS.md STATUS.md)  results/  figs/  logs/  frames/
```
