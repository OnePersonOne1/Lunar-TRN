# STATUS

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
