# τ 벤치마크 재현 절차 (P2)

τ_det = 전처리(letterbox+정규화) + 추론 + NMS, batch 1. 모든 수치 파라미터는 config.yaml `bench`/`detector`에서 온다.

## 조건 고정

1. 노트북/데스크톱 전원 모드를 확인하고 기록한다(스크립트가 `powercfg /getactivescheme`을 meta에 자동 기록).
   측정 중 전원 모드·배터리 상태를 바꾸지 않는다.
2. 측정 중 다른 GPU/CPU 부하(브라우저 영상, 학습 등)를 띄우지 않는다.
3. CPU 측정은 `bench.cpu_threads`(기본 1) 스레드로 고정된다(ORT `intra_op_num_threads`).

## 실행

```
.venv\Scripts\python perception\bench_tau.py --config config.yaml --seed 0 --out results --fig figs\p2_tau_hist.png
```

스크립트가 수행하는 일:

1. `detector.model`(사전학습 가중치) 로드 → ONNX export (`imgsz`, batch 1, static).
2. 캘리브레이션 이미지: `data/calib/`가 있으면 그중 `bench.calib_images`장,
   없으면 ultralytics 내장 샘플 증강 임시 이미지(meta에 `"calib": "temporary"`).
3. TensorRT 엔진 빌드: FP32, INT8(엔트로피 캘리브레이션). ORT용 INT8은 static QDQ 양자화.
4. sanity: 같은 20장에서 INT8 vs FP32 박스 매칭 IoU 평균 ≥ 0.9 확인, 결과를 각 json meta에 기록.
5. 각 조합(warm-up `bench.warmup`회 → `bench.n_iter`회 측정) →
   `results/tau_{trt,ort_cpu}_{fp32,int8}.json` (samples_s, median_s, p95_s, mean_s, std_s, meta)
   + `figs/p2_tau_hist.png`.

meta에는 GPU명, 드라이버, CPU, 스레드 수, TensorRT/ORT/torch/ultralytics 버전, 전원 모드, git hash,
config hash가 기록된다. 숫자를 문서에 옮길 때는 이 json에서만 가져온다.

## 시뮬레이션 연결

`config.yaml`의 `tau.mode: empirical`, `tau.empirical_file: results/tau_ort_cpu_int8.json`으로 두면
폐루프의 τ가 이 분포(samples_s)에서 리샘플된다. 파일 경로만 바꾸면 다른 백엔드 분포를 쓴다.
