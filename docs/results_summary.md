# 슬라이드 숫자 요약 (자동 생성 — scripts/make_slide_assets.py)

모든 값은 results/*.json에서 스크립트가 추출한다. 손으로 고치지 말 것.
TBD = 해당 결과 파일/필드 부재. 구 assumed 수치는 매니페스트에서 제외됨.

| 항목 | 값 | 출처 파일 | 생성 시각(UTC) | git |
|---|---|---|---|---|
| val 프레임 수 | 112 | `p5_det.json` | 2026-08-28T14:59:20 | `13d93d03` |
| 탐지 mAP50-95 (n INT8) | 0.980 | `p5_det.json` | 2026-08-28T14:59:20 | `13d93d03` |
| 탐지 mAP50-95 (n FP32) | 0.994 | `p5_det.json` | 2026-08-28T14:59:20 | `13d93d03` |
| 측정 σ 수평 [m] (n INT8) | 87.9 | `measurement_model.json` | 2026-08-29T03:49:12 | `f4b28d25` |
| 보정 오검출률 (n INT8) | 0.101 | `measurement_model.json` | 2026-08-29T03:49:12 | `f4b28d25` |
| τ n INT8 CPU [ms] | 195.7 | `tau_ort_cpu_int8.json` | 2026-08-28T23:10:00 | `1c67e56e` |
| τ n FP32 CPU [ms] | 219.0 | `tau_ort_cpu_fp32.json` | 2026-08-28T23:06:35 | `1c67e56e` |
| τ s INT8 CPU [ms] | 376.5 | `s/tau_ort_cpu_int8.json` | 2026-08-28T23:33:20 | `1c67e56e` |
| τ GPU INT8 [ms] | 54.5 | `tau_trt_int8.json` | 2026-08-28T23:02:45 | `1c67e56e` |
| CPU 1스레드 CoreMark | 34876 | `p7b_cpu_bench.json` | 2026-09-01T05:30:32 | `74c81b9f` |
| HR5000 대비 배율 | 127x | `p7b_cpu_bench.json` | 2026-09-01T05:30:32 | `74c81b9f` |
| P6 실런 착륙 오차 [m] | 197.8 | `p6_closed_loop.json` | 2026-08-29T03:49:57 | `6d8ce18d` |
| P6 연착륙 속도 [m/s] | 0.65 | `p6_closed_loop.json` | 2026-08-29T03:49:57 | `6d8ce18d` |
| CEP 보상 τ≤1s [m] | 110.8 | `p7b_baseline.json` | 2026-09-01T04:59:39 | `74c81b9f` |
| CEP 미보상 n INT8 [m] | 159.9 | `p7b_tau_serial_compoff.json` | 2026-09-01T05:28:17 | `74c81b9f` |
| CEP 미보상 s FP32 [m] | 323.0 | `p7b_tau_serial_compoff.json` | 2026-09-01T05:28:17 | `74c81b9f` |
| 오검출 0.3 시 CEP [m] | 150.8 | `p7b_fp_sweep.json` | 2026-09-01T05:08:30 | `74c81b9f` |
| 5Hz n INT8 CEP [m] | 48.7 | `p7b_rate_sweep.json` | 2026-09-01T06:34:19 | `fed24586` |
| 5Hz n FP32 CEP [m] | 65.7 | `p7b_rate_sweep.json` | 2026-09-01T06:34:19 | `fed24586` |
| 지터 200ms 시 CEP [m] | 194.9 | `p7b_jitter_sweep.json` | 2026-09-01T06:16:51 | `fed24586` |
| ΔV truth [m/s] | 1109.5 | `p7b_deltav.json` | 2026-09-01T03:03:48 | `5a3f9c01` |
| 고전 PCA mAP50-95 | 0.093 | `p7c_det_compare.json` | 2026-09-03T05:20:47 | `42961e04` |
| 고전 PCA recall | 0.256 | `p7c_det_compare.json` | 2026-09-03T05:20:47 | `42961e04` |
| τ 고전 PCA [ms] | 218.8 | `p7c_det_compare.json` | 2026-09-03T05:20:47 | `42961e04` |
| 측정 σ 수평 [m] (고전) | 460.4 | `measurement_model_classic.json` | 2026-09-03T05:23:28 | `42961e04` |
| 보정 오검출률 (고전) | 0.151 | `measurement_model_classic.json` | 2026-09-03T05:23:28 | `42961e04` |
| CEP 고전 PCA [m] | 664.4 | `p7c_cep_compare.json` | 2026-09-03T05:30:23 | `42961e04` |
| CEP YOLO INT8 (동일 조건) [m] | 126.6 | `p7c_cep_compare.json` | 2026-09-03T05:30:23 | `42961e04` |
| 실런 MC CEP [m] | 297.8 | `p8_unity_mc.json` | 2026-09-03T14:56:25 | `33ec726b` |
| 실런 MC CEP CI 하한 [m] | 284.7 | `p8_unity_mc.json` | 2026-09-03T14:56:25 | `33ec726b` |
| 실런 MC CEP CI 상한 [m] | 312.4 | `p8_unity_mc.json` | 2026-09-03T14:56:25 | `33ec726b` |
| 실런 τ 중앙값 [ms] | 184.8 | `p8_unity_mc.json` | 2026-09-03T14:56:25 | `33ec726b` |
| 실런 MC 바이어스(평균 오프셋 크기) [m] | 290.0 | `p8_summary.json` | 2026-09-03T14:56:36 | `33ec726b` |
| 실런 MC 발산 런 수 (/200) | 1 | `p8_unity_mc.json` | 2026-09-03T14:56:25 | `33ec726b` |
| 실런 MC CEP 정합보정 후 [m] | 133.2 | `p8_unity_mc_corr.json` | 2026-09-03T14:56:24 | `33ec726b` |
| 실런 MC CEP 정합보정 CI 하한 [m] | 120.7 | `p8_unity_mc_corr.json` | 2026-09-03T14:56:24 | `33ec726b` |
| 실런 MC CEP 정합보정 CI 상한 [m] | 139.5 | `p8_unity_mc_corr.json` | 2026-09-03T14:56:24 | `33ec726b` |
| 정합보정 후 잔여 바이어스(중앙값) [m] | 101.5 | `p8_summary.json` | 2026-09-03T14:56:36 | `33ec726b` |
| 정합 진단 오차 로버스트 σ East [m] | 96.8 | `p8_reg_diag.json` | 2026-09-03T12:51:39 | `33ec726b` |
| P9 차세대 n INT8 CEP [m] | 126.6 | `p9_matrix.json` | 2026-09-04T02:17:18 | `20e389e3` |
| P9 차세대 s FP32 CEP [m] | 121.2 | `p9_matrix.json` | 2026-09-04T02:17:18 | `20e389e3` |
| P9 차세대 classic CEP [m] | 664.4 | `p9_matrix.json` | 2026-09-04T02:17:18 | `20e389e3` |
| P9 구세대 n INT8 CEP [m] | 763.4 | `p9_matrix.json` | 2026-09-04T02:17:18 | `20e389e3` |
| P9 구세대 n INT8 미보상 CEP [m] | 2332.5 | `p9_matrix.json` | 2026-09-04T02:17:18 | `20e389e3` |
