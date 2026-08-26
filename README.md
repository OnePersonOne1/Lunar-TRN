# lunar-trn

온보드 경량 AI 크레이터 탐지 기반 달 착륙 TRN(지형상대항법) 폐루프 시뮬레이터 — AI 탐지기를 mAP가 아닌 착륙 오차 분포(CEP)로 평가한다.

## 레이아웃

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
  tests/  docs/  results/  figs/  logs/  frames/
```

## 환경 설정

Python ≥ 3.11, NVIDIA GPU(CUDA 12.8 이상 빌드 필요).

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\check_env.py     # results/env.json 생성
.venv\Scripts\python -m pytest -q
```
