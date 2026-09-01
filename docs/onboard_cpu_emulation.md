# 온보드 CPU 성능 재현 방안 검토 (2026-09-01)

목적: τ 벤치를 돌린 지상 CPU(1스레드)와 실제 우주용 프로세서의 성능 격차를 정량화하고,
그 격차를 시뮬레이션에 반영하는 방법을 정한다. 수치 출처: results/p7b_cpu_bench.json,
results/tau_*.json, results/p7b_tau_serial{,_compoff}.json.

## 1. 성능 격차 (실측)

1스레드(친화도 1코어 고정, zig cc -O2): CoreMark 34,876 iter/s, Dhrystone 40,654 DMIPS.

| 기준 프로세서 | 공표 성능 | 우리 1스레드 대비 |
|---|---|---|
| JAXA HR5000 계열 (MIPS64 5Kf, 200 MHz) | 320 DMIPS | **127배** |
| BAE RAD750 (200 MHz) | ~400 DMIPS | 102배 |
| Gaisler GR740 (LEON4FT, 코어당) | 425 DMIPS / CoreMark 512 | 96배 / 68배 |
| NASA/Microchip HPSC (칩 전체 목표) | RAD750의 ~100배 | **≈ 1배 (같은 자릿수)** |

- SLIM의 SMU가 쓴 CPU 기종을 공개 명시한 문서는 확인하지 못했다. HR5000 계열은
  "JAXA 표준 64-bit rad-hard MPU 세대"의 대표값으로만 쓴다. SLIM의 크레이터
  탐지·매칭은 CPU가 아니라 **Microsemi RTG4 FPGA 전용 로직**이 수행했고(촬영→결과
  ≤5 s), 이는 아래 결론과 정합한다.
- 우리 1스레드 ≈ HPSC 칩 전체 목표 성능과 같은 자릿수 → **현재 τ 실측(n INT8
  195.7 ms 등)은 "차세대(HPSC급) 온보드" 프록시**로 해석한다. 레거시 CPU 기준으로는
  낙관적 하한이다.

## 2. 격차를 반영한 τ 환산 (파생 계산)

n INT8 CPU 실측 τ median 195.7 ms 기준:

| 가정 온보드 | 환산 τ | 프레임 주기(1 s) 대비 | 스윕 결과로 본 CEP |
|---|---|---|---|
| HPSC급 (≈×1) | ~0.2 s | 이하 — 드롭 없음 | 보상 110.8 m (평탄 구간) |
| GR740급 (×68~96) | 13~19 s | 초과 — 드롭 지배 | τ=5 s 외삽 밖: 보상 250 m↑, 미보상 2160 m↑ |
| HR5000/RAD750급 (×102~127) | 20~25 s | 초과 | 소프트웨어 YOLO TRN 성립 불가 |

결론: **레거시 CPU에서는 소프트웨어 탐지 TRN 자체가 불가**(SLIM이 FPGA를 쓴 이유),
**차세대 CPU에서는 현재 실측 그대로 성립**. 양자화·경량화는 이 임계 아래로 τ를
유지하는 마진 수단이다.

## 3. 재현 방안 비교

| 방안 | 방법 | 평가 |
|---|---|---|
| **A. τ 스케일링 (채택)** | 벤치 비율(§1)로 τ를 환산해 스윕 축에 매핑. 파이프라인이 이미 τ를 입력 파라미터로 받음 | 추가 구현 0. 1차 근사(메모리·SIMD 차이는 미반영)임을 명시. τ 격자 상한 5 s 밖은 외삽하지 말고 "성립 불가"로만 서술 |
| B. Windows Job Object CPU rate 제한 | JOBOBJECT_CPU_RATE_CONTROL로 추론 프로세스 사용률을 ~1%로 제한 후 τ 재실측 | 실행 검증용 스팟체크로 유효. 단 듀티사이클 방식이라 지연 분포가 버스트로 왜곡되고 캐시·메모리 속도는 그대로 — median만 참고. 필요 시 10월 |
| C. ISA 에뮬레이션 (QEMU MIPS/SPARC) | 대상 ISA로 추론 스택 실행 | ORT/YOLO 이식 비용이 과도, INT8 커널 미지원 — 기각 |
| D. 전원 계획 클럭 제한 | 최대 프로세서 상태 % 축소 | 최대 3~6배 감속뿐 — 두 자릿수 격차 재현 불가, 기각 |

채택안 A의 서술 규칙: 슬라이드·문서에서 "HR5000급 CPU면 τ ≈ 20 s级 → 프레임 주기의
20배, TRN 불가"처럼 **벤치 비율 기반 1차 환산임을 병기**하고, results에 없는 환산
수치는 표에 계산식과 함께만 제시한다.

출처: HR5000 — wdic.org/w/SCI/HR5000, NEC SpaceWire WG 2008; RAD750 — BAE/Wikipedia;
GR740 — Gaisler Product Brief 2024-12; HPSC — Microchip white paper 2024; SLIM RTG4 —
Acta Astronautica Vol.226 (2025).
