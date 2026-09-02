# 기호 ↔ 코드 대응표 (docs/symbols.md)

CLAUDE.md §2 계약 기호와 코드 변수의 대응. 줄 번호는 2026-09-02 기준 — 리팩터링 시
함수 앵커(파일·함수명)를 우선 신뢰한다. §2 전체 대응표는 P8에서 마저 채운다.

## §2.6 충돌 금지 기호 (계약 확정분)

| 기호 | 뜻 | 코드 변수 | 구현 위치 |
|---|---|---|---|
| K | 칼만 이득 | `K_gain` | sim/ekf.py `EKF.update` (89행) |
| K_c | 카메라 내부 행렬 | `K_cam` | perception/camera.py `K_cam` |
| P | 추정 공분산 | `P_cov` | sim/ekf.py `EKF.__init__` |
| R | 측정 잡음 공분산 | `R_meas` | sim/measurement.py `measurement_R` |
| R_m | 달 반경 | `R_moon` | config `moon.R_m`, data/catalog.py |
| R_{C←L} | L→C 회전 | `R_cam_from_L` | perception/camera.py |
| τ | 추론 지연 | `tau`, `tau_k` | sim/loop.py 촬영 블록, sim/measurement.py `TauSampler` |
| t_c | 촬영 시각 | `t_c` (`t_c_used`) | sim/loop.py 촬영 블록, sim/ekf.py `delayed_update` |
| t_go | 잔여 비행시간 | `t_go` | sim/guidance.py `zem_zev`, sim/loop.py |
| ν | 혁신(innovation) | `nu` | sim/ekf.py `EKF.update` (77행) |
| S | 혁신 공분산 | `S_inn` | sim/ekf.py `EKF.update` (78행) |

## P7b에서 추가된 양 (계약 밖 — 구현 정의)

| 기호/이름 | 뜻 | 코드 변수 | 구현 위치 |
|---|---|---|---|
| busy_until | 직전 프레임 처리가 끝나는 시각 (직렬 처리 모델) | `busy_until` | sim/loop.py:122 |
| n_dropped | 처리 중(busy)이라 촬영을 생략한 프레임 수 | `n_dropped` | sim/loop.py:124 |
| ΔV | Σ\|a_T\|·dt (추력 가속도 적분, 질량 미반영) | `delta_v` → 반환 `delta_v_mps` | sim/loop.py:125 |
| σ_t | 촬영 타임스탬프 오차 1σ | `t_c_jitter` → `jitter` | sim/loop.py:55, config `tau.t_c_jitter_s` |

## P7c 예약 (미구현 — 실행 시 채움)

| 기호 | 뜻 | 예정 변수 |
|---|---|---|
| θ | 카메라 피치(지향각, nadir=0) | `cam_pitch` (예정) |
| Δθ | 지향 지식 오차 (참값 지향 vs 투영에 쓰는 지향) | `att_err` (예정) |
