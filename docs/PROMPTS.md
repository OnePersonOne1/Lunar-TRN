# docs/PROMPTS.md — Claude Code 세션 프롬프트

## 운용 방법
- 저장소 루트에 CLAUDE.md와 config.yaml을 두고 루트에서 `claude`를 실행한다. CLAUDE.md는 매 세션 자동 로드된다.
- 단계(P0~P8)마다 `/clear`로 대화를 비우고 아래 프롬프트를 통째로 붙여 넣는다. CLAUDE.md는 유지된다.
- 세션이 길어지면 `/compact`. 규칙을 안 지키는 것 같으면 `/memory`로 CLAUDE.md가 로드됐는지 먼저 확인한다.
- 17일 일정이나 발표 스토리는 CLAUDE.md에 넣지 않는다(매 세션 토큰만 먹는다). 필요하면 docs/에 두고 `@docs/파일명`으로 그때만 불러온다.
- 학습·대량 렌더·MC는 Claude가 명령만 출력하고 내가 tmux에서 돌린다. 끝나면 다음 세션에서 "logs/xxx.log 읽고 결과 집계"로 시작한다.
- Claude는 config.yaml의 site / scenario / trn_band를 스스로 못 바꾼다. 바꿔야 하면 값을 제안하게 하고 내가 수정한다.

---

## P0 · 환경·뼈대 (8/27)
```
CLAUDE.md와 config.yaml을 읽어라. 이번 세션은 P0(환경·뼈대)다.
1) CLAUDE.md §5 레이아웃대로 디렉터리와 빈 모듈(파일마다 docstring 1줄)을 만들어라.
2) requirements.txt 작성. torch는 CUDA 12.8 이상 빌드(cu128 인덱스)로 설치해라. GPU가 RTX 5060 Ti(sm_120)라 구버전 휠은 "not compatible" 에러가 난다. 설치를 실행하고 실패하면 원인과 대안을 보고해라. 나머지: ultralytics, tensorrt, onnxruntime, openvino, opencv-python, rasterio, scipy, matplotlib, pyyaml, pytest.
3) scripts/check_env.py: torch 버전, CUDA 버전, GPU 이름, compute capability, 각 패키지 import 가능 여부를 results/env.json에 기록.
4) tests/test_smoke.py: config.yaml 로드, 모든 모듈 import, K_c가 f = H/(2·tan(θ_v/2))로 계산되는지 검증(perception/camera.py에 K_cam(cfg)만 먼저 구현).
5) .gitignore(CLAUDE.md §3 목록), README.md 뼈대(제목·목적 한 문장·레이아웃, 사람 이름·소속 없음), docs/STATUS.md 초기화.
6) git init 후 커밋 "P0: scaffold".
합격: pytest -q 통과, results/env.json에 GPU가 sm_120으로 인식됨.
```

## P1 · 시뮬 코어 (8/27~29)
```
CLAUDE.md §2 계약과 config.yaml을 읽어라. 이번 세션은 P1(시뮬 코어)이다. 테스트를 먼저 쓰고 구현해라.
구현:
- sim/dynamics.py: rk4_step(x, a_T, dt, g), 참값 전파. 상태 x=[r;v].
- sim/guidance.py: zem_zev(r_hat, v_hat, t_go, cfg) → a_T. §2.5 그대로(포화는 방향 유지).
- sim/ekf.py: class EKF — predict(a_imu, dt), update(z, R_meas), delayed_update(z, t_c, R_meas): 링버퍼로 t_c 시점 상태 복원 → 보정 → 버퍼 IMU로 재전파. gate(nu, S_inn) → bool. F, G, Q는 §2.4.
- sim/measurement.py: StatMeasurementModel(cfg, rng): 참값 r에서 z 생성(σ_xyz 가우시안, 확률 fp_rate로 무작위 방향 fp_offset 이상치), valid 플래그. TauSampler(cfg): constant | empirical.
- sim/loop.py: run_closed_loop(cfg, seed, tau=None, fp_rate=None, delay_comp=None, measurement="stat") → dict(landing_xy, landing_v, traj_true, traj_est, est_error, gate_log, tau_log). measurement="unity"는 P6에서 채우도록 인터페이스만.
- sim/mc.py: multiprocessing seed 병렬, CEP(반경 오차 중앙값), 95% 오차 타원(공분산 고유분해), 부트스트랩 95% CI.
- scripts/run_closed_loop.py, scripts/sweep_tau.py, scripts/run_mc.py — CLI, --config --seed --out.
테스트(tests/test_sim.py):
- 자유낙하(a_T=0) RK4 vs 해석해: 100 s 후 위치 오차 < 1e-6 m.
- 참값 상태 + ZEM/ZEV: config scenario에서 착륙 오차 < 1 m, 착륙 속도 < 0.1 m/s, 전 구간 |a_T| ≤ a_max. 실패하면 시나리오가 비현실적이라는 뜻이다. config를 고치지 말고 어떤 값을 얼마로 바꿔야 하는지 보고하고 멈춰라.
- EKF 일관성: 통계 측정모델 50 seed에서 시간 평균 NEES가 χ²₆ 95% 구간 안.
- 지연 보상: τ=1 s 무잡음에서 delayed_update 결과가 무지연 update와 1e-6 이내 일치, 미보상은 |v|·τ 규모의 편향 발생.
- 게이트: 10σ 이상치 기각, 1σ 정상치 통과.
산출: figs/p1_trajectory.png(참값·추정·목표, 3면도), figs/p1_est_error.png, figs/p1_tau_sweep.png(τ별 CEP, 보상/미보상 두 곡선, 제목에 "assumed measurement stats"), results/p1_tau_sweep.json.
끝: STATUS.md 갱신, 커밋 "P1: sim core".
```

## P2 · τ 벤치마크 (8/30)
```
이번 세션은 P2(τ 벤치마크)다. CLAUDE.md §3 결과 기록 규칙을 지켜라. 학습 전이므로 사전학습 가중치(config detector.model)를 쓴다.
- perception/bench_tau.py: 모델 → ONNX export. 백엔드 (a) TensorRT FP32 / INT8(캘리브레이션 config bench.calib_images장. data/calib/가 없으면 임시 이미지를 쓰고 json meta에 "calib": "temporary" 기록), (b) ONNX Runtime CPU FP32 / static INT8, intra_op_num_threads = config bench.cpu_threads.
- 프로토콜: batch 1, imgsz config, warm-up bench.warmup, 측정 bench.n_iter. τ_det = 전처리 + 추론 + NMS. median, p95, mean, std, 히스토그램. meta에 GPU명, CPU명, 스레드 수, 드라이버·TensorRT·ORT 버전, 전원 모드.
- sanity: 같은 20장에서 INT8 vs FP32 박스 매칭 IoU 평균 ≥ 0.9. 미달이면 json에 실패로 기록하고 원인 추정.
- 출력 스키마(TauSampler empirical이 읽음): results/tau_{backend}_{precision}.json = {"samples_s": [...], "median_s", "p95_s", "mean_s", "std_s", "meta": {...}}. figs/p2_tau_hist.png(4조합 한 그림).
- docs/bench_protocol.md: 사람이 재현할 수 있는 절차.
- sim/measurement.py TauSampler empirical 모드가 이 파일을 읽는지 테스트 추가(tests/test_tau_sampler.py).
끝: STATUS.md, 커밋 "P2: tau benchmark".
```

## P3 · 데이터·라벨러 (8/31~9/1)
```
이번 세션은 P3(데이터·라벨러)다. 원본은 내가 data/raw/에 내려받아 뒀다(SLDEM2015 타일, LROC WAC 100 m 모자이크 타일, Robbins 카탈로그 CSV). 파일명은 ls로 확인해라. config site의 lat0/lon0는 내가 채웠다.
- data/crop.py: config site(lat0, lon0, box_km)로 DEM·모자이크를 잘라 §2.1 등장방형 평면 근사 L 격자로 리샘플(rasterio). 출력 data/processed/dem_L.npz(x, y 격자, z), texture_L.png(북이 위, 동이 오른쪽), Unity용 16-bit RAW heightmap(해상도 config unity.terrain_heightmap_res, 정사각 2^n+1, 부족한 쪽은 패딩) + heightmap_meta.json(m 단위 크기, z 최소/최대, 원점 위치).
- data/catalog.py: Robbins CSV → box 안, D ≥ D_min, x·y(§2.1), z(DEM 이중선형 보간) → data/processed/catalog_L.csv. 열 이름이 배포본과 다르면 매핑을 보고해라.
- perception/camera.py: §2.2 완성. K_cam(cfg), project(points_L, r) → (u, v, valid), backproject_ray(u, v, r) → 단위 방향벡터(L).
- perception/labeler.py: poses.csv → 프레임별 YOLO txt(§2.3 포함 조건) + overlay PNG(원과 id).
- scripts/analyze_altitude_band.py: 착륙지 상공 공칭 궤적(config scenario) 위 고도별로 시야 내 카탈로그 크레이터 수와 최소 투영 직경(px)을 계산. figs/p3_altitude_band.png. 조건(시야 내 ≥ trn_band.n_min_in_view, 투영 직경 ≥ p_min)을 만족하는 [h_min, h_max]를 results/p3_altitude_band.json에 기록. config trn_band는 바꾸지 말고 제안만 해라.
- figs/p3_registration.png: 카탈로그 원을 texture_L.png 위에 overlay(정합 육안 확인용).
테스트(tests/test_camera.py): 투영→역투영 round-trip, 카메라 바로 아래 점이 (c_x, c_y), 동쪽 점은 u 증가, 북쪽 점은 v 감소, z_C ≤ 0 무효. tests/test_labeler.py: 화면 밖 크레이터 제외, 경계 클리핑.
끝: STATUS.md, 커밋 "P3: data pipeline".
```

## P4 · Unity 렌더 서버 (9/2~9/3 정오, 36시간 time-box)
```
이번 세션은 P4(Unity 렌더 서버)다. C#, Python 클라이언트, 수동 절차 문서만 작성한다. Editor 작업은 내가 한다. 최소 구성만.
- unity/Assets/Editor/SceneBuilder.cs: 메뉴 "LunarTRN/Build Scene". data/processed/heightmap_meta.json과 RAW를 읽어 TerrainData.SetHeights로 Terrain 생성(가로세로·고도 스케일 m 단위). texture_L.png를 TerrainLayer로 지정(타일링 = 지형 크기, 방향 확인). Directional Light "Sun". Camera(수직 FOV = config camera.fov_v_deg, RenderTexture W×H, far clip ≥ 200 km). RenderServer 컴포넌트 부착. L 원점(착륙 목표점)이 Unity 월드 (0, 0, 0)에 오도록 Terrain 위치 오프셋.
- unity/Assets/Scripts/RenderServer.cs: TCP 서버(config unity.port). 요청 JSON {frame_id, t, r_L:[x,y,z], sun_az_deg, sun_el_deg} → 카메라 위치(L→Unity: x=E, y=U, z=N), nadir 자세(forward = −Up, 이미지 상단 = 북), 태양 방향(§2.1 정의) 설정 → 렌더 → PNG 바이트(4바이트 big-endian 길이 접두). 예외는 {"error": ...}.
- unity/client.py: render(r_L, sun_az_deg, sun_el_deg, frame_id) → numpy 이미지. 타임아웃(config unity.timeout_s)·재시도 3회.
- scripts/check_projection.py: 알려진 pose에서 렌더 → camera.py로 카탈로그 원 overlay → figs/p4_projection_check.png. 가장 큰 크레이터 5개에 대해 예측 중심 vs 렌더 상 중심(Hough 원 검출 또는 템플릿) 픽셀 오차를 results/p4_projection_check.json에 기록. 5 px 이내면 합격.
- unity/README.md: Unity 버전, 프로젝트 생성, Assets 복사, 메뉴 실행, Play, 포트 확인, 흔한 오류(좌수계 뒤집힘, 고도 스케일, 텍스처 상하/좌우 반전, RAW 바이트 오더)를 번호 매겨 적어라.
파일 작성 후 멈춰라. 내가 GUI 작업을 마치고 "서버 떠 있음"이라고 하면 check_projection부터 실행한다.
```

### P4 후속 (GUI 완료 후)
```
Unity 서버가 config unity.port에 떠 있다. scripts/check_projection.py를 실행하고 결과를 해석해라. 5 px 초과면 원인(좌우/상하 반전, 스케일, 오프셋)을 좁혀 RenderServer.cs 또는 SceneBuilder.cs 수정안을 제시해라. camera.py는 건드리지 않는다.
```

## P5 · 데이터셋·학습·PTQ·개루프 보정 (9/3 오후~9/4)
```
이번 세션은 P5다. Unity 서버는 떠 있다.
- scripts/make_dataset.py: sim으로 공칭 궤적(TRN 구간만) pose 생성 + 태양각 무작위(config dataset) → unity/client.render → 이미지·YOLO 라벨(labeler)·poses.csv. N = config dataset.n_frames. train/val은 궤적 구간 단위로 분리(프레임 무작위 분리 금지). data/dataset.yaml. figs/p5_overlay_{k}.png 6장. 대량 렌더 명령은 출력만 하고 내가 tmux로 돌린다.
- scripts/train.py: ultralytics 학습(config detector), seed 고정, logs/train.log. 명령만 출력.
- scripts/export_int8.py: best.pt → TensorRT INT8(캘리브레이션 = train 이미지 config bench.calib_images장) + ONNX Runtime CPU INT8. data/calib/도 이 이미지로 채운다.
- scripts/eval_det.py: val에서 FP32/INT8의 mAP50, mAP50-95, config conf에서 정밀도·재현율·FP율(프레임당 오검출 수) → results/p5_det.json. 이어서 학습 모델로 bench_tau 재실행(P2 스키마).
- perception/associate.py: 예측 pose로 카탈로그 투영 → 탐지 중심과 최근접 연관(게이트 반경 px, 후보 2개 이상이면 그 탐지는 기각). 파일 상단에 # TODO(oct): 기하 불변량 매칭.
- perception/pnp.py: cv2.solvePnPRansac(3D 크레이터 중심 ↔ 2D 탐지 중심, K_c, 왜곡 없음). 대응쌍 ≥ 4, 인라이어 ≥ measurement.n_min_inliers → r_PnP, valid, 재투영 오차.
- scripts/calibrate_measurement.py: val 프레임에 대해 "참값 pose + 예측 오차 모사(config ekf.x0_error 규모의 무작위 섭동)"로 연관·PnP 실행 → 축별 오차 σ·편향, 이상치 비율(|err| > 3σ), valid 비율 → results/measurement_model.json(sim/measurement.py가 읽는 스키마), figs/p5_pnp_error_hist.png. FP32/INT8 각각.
테스트: tests/test_pnp.py — 합성 대응쌍(잡음 0)에서 r_PnP 오차 < 1e-3 m, 이상치 30% 섞어도 RANSAC이 참값 복원.
끝: STATUS.md, 커밋 "P5: dataset, detector, calibration".
```

## P6 · Unity-in-the-loop 통합, Plan A (9/5~9/6)
```
이번 세션은 P6(폐루프 통합)다. 9/6 저녁까지 1회 완주를 못 하면 blocker를 STATUS.md에 적고 멈춘다. 완벽함보다 완주.
- sim/loop.py measurement="unity": TRN 구간의 카메라 주기마다 참값 r과 태양각으로 client.render → perception/detect.py(INT8 엔진, config) → associate(EKF 예측 pose) → pnp → z(t_c) → τ 적용(모드: wallclock | empirical | constant) → delayed_update → guidance.
- 프레임별 로그: n_det, n_match, 인라이어, PnP 오차(참값 대비), 게이트 채택 여부, τ.
- 산출: results/p6_run_{seed}.json, figs/p6_trajectory.png, figs/p6_est_error.png, frames/p6/{frame_id}.png(렌더 + 탐지 박스 + 투영 카탈로그 overlay, 영상용).
- scripts/run_closed_loop.py에 --measurement unity, --detector fp32|int8 옵션.
합격: 착륙 판정까지 1회 완주, 결과 파일 생성. 실패해도 어디까지 갔는지(몇 프레임, 어느 단계) STATUS.md에 남긴다.
끝: STATUS.md에 Plan A 성공/실패 명시, 커밋 "P6: integration".
```

## P7 · 예비 몬테카를로 (9/6 밤~9/7)
```
이번 세션은 P7(예비 MC)이다. 계산은 내가 tmux에서 돌린다.
- scripts/run_mc.py 실행 명령 세트를 출력해라: measurement="stat" + results/measurement_model.json(보정 통계, FP32/INT8 각각), τ ∈ config tau.sweep_s ∪ {실측 median: TensorRT INT8, ORT CPU FP32, ORT CPU INT8}, delay_comp ∈ {true, false}, n = config mc.n_runs. 조건별 results/p7_mc_{cond}.json.
- 집계 스크립트(scripts/aggregate_mc.py): results/p7_mc.json(조건별 CEP, 95% 타원, 시행 수, 부트스트랩 95% CI), figs/p7_cep_vs_tau.png(실측 τ 지점을 마커로 표시, 미보상 곡선 병기), figs/p7_scatter_{cond}.png.
- SLIM 정합성 비교(site가 SLIM(13.3°S, 25.2°E)일 때): results/p7_slim_comparison.json —
  우리 시뮬의 CEP·고도별 EKF 위치 오차를 SLIM 공개 비행 결과와 나란히 표로.
  외부 기준값(출처 명기: "Vision-based navigation and obstacle detection flight results in
  SLIM lunar landing", Acta Astronautica Vol.226 (2025)): VBN 총 14회 전부 성공(CST1/2, VLD1/2),
  고도 500 m 수평 항법 오차 <1 m, 고도 50 m 시점 착륙 정밀도 ~10 m (목표 100 m).
  주의: SLIM 실제 착지점 이탈 ~55 m는 엔진 노즐 탈락(추진 고장) 영향이므로 비교 기준으로
  쓰지 않는다. 주장 수위는 "validation"이 아니라 "실제 임무의 공개 비행 결과와 자리수·경향이
  일관(plausibility)"으로 제한한다 — 센서 구성(지도 매칭+레이더+LRF vs YOLO+PnP)과
  자유도(6-DOF vs 3-DOF)가 다르다.
- 모든 그림 제목에 "preliminary, n=…; full MC scheduled Oct".
끝: STATUS.md, 커밋 "P7: preliminary MC".
```

## P7b · 직렬 처리량 모델·오검출률 스윕·s 보정·SLIM 표 (9/2~9/3)
```
이번 세션은 P7b다. 계산은 내가 tmux에서 돌린다. 신규 기능 마감은 9/4 24시이고 이 세션과 (조건부) P7c가 마지막이다. 시작 규칙(CLAUDE.md §4)대로 pytest 후 상태 5줄, 계획 10단계 이내. 시작 전에 미커밋 문서 변경(docs/limitations.md, docs/team_status.md 등)이 있으면 "docs: …"로 먼저 따로 커밋해 P7b 커밋에 섞이지 않게 한다. 아래 순서를 지키고, 한 항목이 2시간 또는 5회 시도에서 막히면 그 항목을 버리고 다음으로 간다.
0) 전제 전환. config measurement.mode를 calibrated로 바꾸고 시작한다(run_mc에 mode CLI가 없어 config가 유일한 스위치다 — 이 세션의 기준 조건 전부가 보정 통계 기준). 겸사: scripts/run_mc.py의 산포도 제목 "(assumed measurement stats)" 하드코딩을 measurement_R의 실제 소스에 따라 assumed/calibrated로 표기하게 고친다(§2.4 표기 규칙).
1) 직렬 처리 모델. config에 tau.serial(기본 false)을 추가한다. sim/loop.py에서 촬영 시점 t_next가 직전 프레임의 도착 시각 busy_until보다 앞이면 촬영을 건너뛰고 n_dropped를 센다. 촬영하면 busy_until = t_next + τ_k. τ 샘플은 실제 촬영할 때만 추출한다(건너뛴 프레임에서 rng를 소비하지 않는다 — empirical 모드 결정론). unity 모드의 wallclock τ에도 같은 규칙. 반환 dict에 n_meas와 n_dropped 추가. serial false는 기존 결과를 비트 단위로 재현해야 한다. 테스트를 먼저 쓴다: 같은 seed에서 serial false의 landing_xy가 P7과 동일, 카메라 1 Hz·τ=2.5 s 상수·serial true에서 밴드 내 측정 수가 false 대비 약 1/3.
2) 보상 조건 τ 스윕(직렬). config에 tau.sweep_serial_s = [0.05, 1.0, 2.0, 5.0]와 tau.measured_points(파일 경로·라벨 쌍: results/tau_trt_int8.json "n INT8 GPU", results/tau_ort_cpu_int8.json "n INT8 CPU", results/tau_ort_cpu_fp32.json "n FP32 CPU", results/s/tau_ort_cpu_int8.json "s INT8 CPU", results/s/tau_ort_cpu_fp32.json "s FP32 CPU")을 추가하고, scripts/sweep_tau.py에 --serial on/off, --comp on/off/both, --measured-points를 붙인다. 실측 점의 τ는 파일의 median_s를 읽는다. 조건: 격자 4점+실측 5점, delay_comp on, serial on, n = mc.n_runs. results/p7b_tau_serial.json에 조건별 tau_s, label, cep_m, cep_ci95_m, ellipse95, n_runs, mean_n_meas, mean_n_dropped, mean_gate_accept. figs/p7b_cep_vs_tau_serial.png: x축 log τ, 왼쪽 축 CEP+CI, 오른쪽 축 밴드 내 측정 수, 1/camera.rate_hz 위치에 세로선, 실측 5점 라벨. 미보상 곡선은 그리지 않는다.
3) 오검출 오프셋 불일치 수정. sim/measurement.py StatMeasurementModel이 fp_offset을 config(2000 m)에서만 읽는다. measurement.mode가 calibrated이고 파일에 fp_offset_med_m(현재 381.4 m)이 있으면 그 값을 쓰도록 고치고, 사용값을 반환 dict와 results meta에 fp_offset_used_m으로 남긴다. 기준 조건(τ = n INT8 CPU median, comp on, serial on)을 보정 오프셋으로 재실행 → results/p7b_baseline.json, 2000 m 결과와의 차이를 STATUS에 숫자로만 적는다.
4) 오검출률 스윕 — 연구계획서 산출물 ④, 이 세션의 최우선 신규 산출물. config에 measurement.fp_sweep = [0.0, 0.05, 0.10, 0.20, 0.30]을 추가하고 scripts/sweep_fp.py를 sweep_tau.py 구조로 만든다. τ = n INT8 CPU median, comp on, serial on, 오프셋은 보정값, n = mc.n_runs. results/p7b_fp_sweep.json에 조건별 CEP, CI, 게이트 기각 비율, 수락 측정 수. figs/p7b_cep_vs_fp.png: 왼쪽 축 CEP, 오른쪽 축 기각 비율, 보정 오검출률 0.101 위치에 세로선.
5) s 모델 측정 보정. scripts/calibrate_measurement.py를 s INT8 ONNX로 val 프레임에 돌려 results/measurement_model_s.json을 만든다 — 경로 주의: runs/export_s/crater_int8_ort.onnx (runs/export/는 n 모델이다). Unity 불필요(저장된 val 프레임 사용). 그 σ·오검출률로 기준 조건 MC 1회 → results/p7b_mc_s.json, n 기준 조건과 나란히 STATUS에 적는다. (근거: "mAP 동급인데 CEP 차이" 주장은 s를 s 자신의 측정 통계로 평가해야 성립한다.)
6) SLIM 비교표. P7 프롬프트의 results/p7_slim_comparison.json이 실행되지 않았다. 같은 명세·같은 주장 수위(plausibility)로 만든다.
7) (시간 남으면) ΔV. run_closed_loop 반환에 delta_v_mps = Σ|a_T|·dt를 추가하고 run_mc·aggregate_mc가 평균·p50·p95를 집계. 기준 조건과 measurement="truth" 실행(P1 시나리오, 이상 ΔV)의 ΔV를 results/p7b_deltav.json에 저장. 해석은 쓰지 않는다(팀원 2의 로켓 방정식 입력용). 밀리면 이 항목만 버린다.
끝: STATUS.md 갱신, 커밋 "P7b: serial throughput model, fp sweep, s calibration, SLIM table (+deltaV)".
```

## P7c · 카메라 지향·PnP 방식 민감도 (9/4, 조건부)
```
착수 조건: P7b가 9/3 안에 완료·검증(pytest 통과, 산출물 확인)됐을 때만 이 세션을 연다. 아니면 건너뛰고 "문서 갱신" 세션으로 간다 — 이 실험은 핵심 주장(τ·오검출→CEP) 바깥의 민감도 분석이고, 안 하면 limitations.md 서술로 방어한다. 슬라이드 동결 6일 전에 전 모듈이 import하는 camera.py를 건드리는 작업임을 유념하고, 비트 재현 테스트가 깨지면 즉시 되돌린다.
이번 세션은 P7c다. 사용자 승인(2026-08-31): 계약 §2.2에 카메라 지향 파라미터를 추가하는 것을 허용한다. 자세 상태·자세 동역학·자이로·EKF 상태 확장은 여전히 §1 범위 밖이며 하지 않는다. CLAUDE.md §2.2를 아래 (1)에 맞게 고치고 개정 날짜를 적는다.
1) 카메라 지향. perception/camera.py에서 R_cam_from_L(theta_deg) 함수화: R_{C←L} = R_att(θ)·diag(1, −1, −1), R_att는 East 축(x_C) 기준 회전 — 시선이 남북(교차트랙)으로 기운다. 의도된 선택이다: 교차트랙 축은 속도·지연으로 인한 along-track(East) 오차와 결합하지 않아 순수 기하 민감도를 분리하며, 이론선 h·Δθ는 축 선택과 무관하다. config에 camera.pointing_known_deg(기본 0.0). project와 backproject_ray가 이 함수를 쓰고, θ=0에서 기존 결과가 비트 단위로 재현되어야 한다. 테스트: θ=0 재현, θ=1°에서 nadir 지상점 투영이 약 f·tan(1°) px 이동.
2) 자세 고정 PnP. perception/pnp.py에 solve_pnp_known_R 추가. 회전 기지 시 (u−c_x)·z_C − f·x_C = 0, (v−c_y)·z_C − f·y_C = 0이 r에 선형 → 최소제곱, RANSAC은 재투영 오차 기준 기존 임계값. config에 measurement.pnp_mode = free | known_att(기본 free). 테스트: 무잡음 투영에서 r 복원 1e-6 m 이내, 이상치 20%에서도 복원.
3) 스윕(축소 기본). config에 camera.pointing_sweep_deg = [0.0, 0.1, 0.3] (기본. 3까지 순조로우면 0.05와 1.0을 추가한다). calibrate_measurement.py에 --pnp-mode와 --pointing-known-deg를 붙여 val 프레임(nadir 렌더)에 대해 known_att 모드로 Δθ 각 값, free 모드로 Δθ 0과 최대값을 돌린다. 렌더는 nadir이고 필터가 아는 자세만 Δθ만큼 틀린 지식 오차 실험이다. 각 조건의 σ·오검출률을 results/p7c_pointing_{mode}_{deg}.json에 저장하고, 각 통계로 기준 조건 MC(τ = n INT8 CPU median, comp on, serial on)를 돌려 results/p7c_pointing_sweep.json에 모은다. figs/p7c_sigma_cep_vs_pointing.png: x축 Δθ, 왼쪽 축 수평 σ, 오른쪽 축 CEP, known_att/free 마커 구분, h·Δθ(h = trn_band 중앙값) 이론선 점선. 해석 문장은 쓰지 않는다.
4) (선택, 9/4 24시 전 여유 시에만) 구동기 오차. config에 actuator.dir_err_deg(기본 0)와 actuator.scale_err(기본 0)를 추가하고, sim/loop.py에서 참값 동역학과 IMU에 들어가는 a_T에만 회전·스케일 적용(유도 명령은 그대로). 1°·0.05로 기준 조건 MC 1회 → results/p7c_actuator.json. 시간 없으면 하지 않는다.
끝: STATUS.md 갱신, 커밋 "P7c: camera pointing and PnP mode sensitivity".
```

## 문서 갱신 (P7b/P7c 직후, 9/5 오전)
```
1) docs/limitations.md §3(기타 범위 한계)에 추가 — 절 번호 주의: §2는 유도 법칙 한계다. 항목: 촬영 시각 불확실성 미포함 / 병렬 파이프라인 가정(serial false)과 직렬 결과의 차이(results/p7b_tau_serial.json 수치 인용) / TRN 밴드 하한 22 km 아래 IMU 단독 구간 / 카메라 지향 nadir 가정·PnP 자유 회전(P7c를 했으면 결과 파일 참조, 건너뛰었으면 미정량 한계로 서술) / 구동기 이상화(짐벌·추력 편향·질량 변화 없음).
2) P8 프롬프트의 신규성 절 교체. 기여는 네 가지: ① 공개 폐루프 테스트베드 ② INT8이 mAP 손실에도 측정 σ·오검출률·CEP를 바꾸지 않음 ③ 촬영 시각 보상 필터에서 지연 자체는 CEP를 바꾸지 않았고 지연이 들어오는 경로는 측정률과 오검출임 — 단 이 문장은 results/p7b_tau_serial.json 수치로 확인한 뒤 채택하고, 결과가 다르면(직렬에서 τ 증가 시 CEP 상승 등) 문장을 결과에 맞춘다 ④ 폐루프가 설계값을 바꾼 사례(h_min 17 vs 22 km, 게이트 연쇄 기각과 공분산 팽창). "실측 τ 주입"과 미보상 곡선의 n 대 s 배율은 차별점으로 쓰지 않는다. 같은 절의 구 밴드 수치 "16.5~30 km"도 22~30 km로 갱신한다. 선행연구 표: streaming perception 2020, planner-centric metrics 2020, LunaNet 2020, AST 2022, Aerospace 2025, ION NAVIGATION 2024, arXiv 2606.14776(Candan & Servadio, DL 크레이터 TRN — 궤도 영상·개루프)을 τ 취급 | 종단점 | 폐루프 여부 | 공개 여부 4열로 넣는다.
3) docs/symbols.md는 P8 산출물이라 아직 없다. 없으면 이 세션에서 생성하고 θ(카메라 피치), Δθ(지향 지식 오차), busy_until, n_dropped, delta_v_mps를 포함한다(§2 전체 대응표는 P8에서 마저 채운다).
4) P8 프롬프트에 "docs/qa_facts.md 생성 시 P7b·P7c results 파일을 숫자 출처에 포함한다"를 한 줄 추가한다.
커밋 "docs: positioning update, limitations".
```

## P8 · 발표 자산·Q&A 준비 (9/8~9/10)
```
@docs/ref/온보드 경량 AI 크레이터 탐지 기반 달 착륙 지형상대항법(TRN)의 폐루프 성능 정량화.pdf 를 읽고, 슬라이드 자산이 제출된 계획서 범위를 벗어나지 않는지 확인하면서 작업해라.

이번 세션은 P8(발표 자산)이다. 슬라이드 자체는 만들지 않는다.
- scripts/make_slide_assets.py: results/*.json에서 슬라이드에 쓸 숫자를 뽑아 docs/results_summary.md에 "숫자 | 출처 파일 | 생성 시각 | git hash" 표로 정리. 출처 없는 숫자는 넣지 않는다. figs/ 중 슬라이드용 그림을 figs/slides/slide_XX_주장.png로 복사.
- 시연 영상: frames/p6/(Plan A) 또는 p1·p7 결과 애니메이션(Plan B)을 ffmpeg로 30초 이내 mp4. 자막·로고 없음.
- docs/symbols.md: CLAUDE.md §2 기호 ↔ 코드 변수 ↔ 구현 파일:줄 대응표.
- docs/qa_facts.md: 코드·config·results에서 확인 가능한 사실만(카메라 파라미터, 고도 구간과 근거, 데이터셋 크기·분리 방식, 학습 설정, 벤치마크 하드웨어, MC 시행 수·CI). 추정과 해석은 넣지 않는다.
- docs/qa_facts.md에 "신규성 포지셔닝·선행연구" 절을 반드시 포함 (2026-08-28 결정):
  - 합성 렌더 학습은 신규성으로 주장하지 않는다 — 확립된 방법론을 따랐다고 말한다:
    ESA PANGU(행성 표면 합성 렌더 기반 비전 항법 개발·검증 표준 도구),
    DeepMoon(Silburt et al. 2019, DEM 렌더로 CNN 크레이터 탐지 학습) 및 후속 합성→실사 연구,
    Airbus SurRender. 크레이터 기반 항법 실적: Mars 2020 LVS(지도 상대 항법),
    SLIM 크레이터 매칭(고전 영상처리, 비딥러닝).
  - 이 연구의 기여로 내세울 것: "탐지기를 mAP가 아니라 착륙 CEP로 평가" —
    INT8 양자화 탐지기의 τ 실측(하드웨어) → 실측 τ·오검출률을 폐루프 EKF+유도에 주입 →
    MC로 CEP 정량화. 인식→항법→유도를 관통하는 평가 사슬이 신규성이다.
  - Q&A 대비: 합성-실사 도메인 갭 질문에는 숫자로 답한다(텍스처 100 m/px vs
    카메라 GSD 19~34 m/px, 3~5배 업샘플; 태양각 randomization; 한계는 10월 NAC 패치·
    광학계 모델로 보완 계획). 데이터셋 커버리지 질문에는: 밴드 16.5~30 km를 1 Hz
    148프레임(고도 간격 40~90 m)으로 연속 커버, 궤적 반복 7회는 태양각만 변경,
    접근 회랑(y=0) 한정임을 선제 명시. 과적합 질문에는: train/val=SLIM(배포 도메인) 유지
    근거 + 남부 고지대 held-out 일반화 test(STATUS 2026-08-28 항목) 결과 인용.
끝: STATUS.md, 커밋 "P8: slide assets".
```

---

## Plan B 메모 (P6 실패 시)
P7은 어차피 measurement="stat" + 보정 통계로 돌아간다. 슬라이드에서는 "개루프 측정 통계 보정(P5) → 보정 통계 폐루프(P1/P7)"로 사슬을 보여 주고, 실시간 결합은 10월 일정이라고 명시한다. P6 부분 완주 로그도 증거다.
