# STATUS

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
