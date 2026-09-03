# STATUS

## P8b · 씬-카탈로그 정합 진단·보정 — 실런 바이어스 원인 규명 (2026-09-03 야간)

- 배경: P8 실런 MC의 CEP 297.8 m가 계통 바이어스(290 m) 지배임이 확인됨 → 원인 진단.
- 진단(`scripts/calibrate_registration.py`): 회랑 참값 pose 101곳 × 태양각 2종(135°/315°)
  개루프 렌더→탐지→연관(참값 pose)→PnP로 오차 **벡터** e(t) 실측. 결과: 전체 중앙값
  바이어스는 [15, 23, 19] m로 작지만 **East 빈 중앙값이 트랙을 따라 +60→−80→+120 m로
  굽이침**(figs/p8_reg_diag.png). 태양각 2종 일치 → 조명 아닌 씬-카탈로그 정합.
  메커니즘: b(r)의 along-track 변화율을 EKF가 속도로 흡수 → 밴드 이탈 후 250 s
  IMU 단독 coast에서 증폭 (1.2 m/s × 250 s ≈ 290 m, 자릿수 일치).
- 보정: `--fit east_bins`로 East 10 km 구간 8-bin 바이어스 중앙값 테이블
  (results/registration_correction.json). `UnityMeasurementModel`이
  measurement.registration_correction 키(기본 없음=꺼짐)로 z에서 b(r̂_East) 차감 —
  계약 §2.4의 z 정의·필터 불변, 기존 결과 재현 불변. 실제 임무의 사전(pre-flight)
  맵타이 보정에 해당(참값 pose 사용은 오프라인 보정이므로 정당, limitations 명시).
- 재검증(실런 MC n=200, wallclock τ, 보정 on): **CEP 297.8 → 133.2 m [120.7, 139.5]**,
  잔여 바이어스(중앙값) 101.5 m, 통계 MC 110.8 m [98.0, 132.5]와 CI 인접(약간 겹침).
  seed 14는 이번에도 초기 연관 실패로 추락(27.9 km, 유한값) — mc.crash_error_m(5000 m,
  config 신설)로 발산/추락 분류 일관화(CEP·타원 제외, n_diverged 집계).
- 테스트: `pytest -q` 69 passed — 보정 룩업·클램프, 추락 분류 테스트 추가.
- 생성: results/p8_reg_diag.json, registration_correction.json, p8_unity_mc_corr.json,
  p8_summary.json(corr 키 추가), figs/p8_reg_diag.png, p8_landing_dispersion.png(3원),
  results_summary.md 39행(정합보정 5행 추가).
- 다음: 잔여 바이어스(~100 m, 10 km bin 조도 한계) — 5 km bin 또는 2차 보정은 10월.
  seed 14류 초기 연관 실패는 lost-in-space 매칭(10월)의 몫. 스토리보드에 3원 산포도 반영 권장.

## P8 · Unity 실런 몬테카를로 — 프레임별 실측 τ(wallclock) n=200 (2026-09-03)

- 실행 조건: `scripts/run_mc_unity.py` — measurement=unity, tau=wallclock(프레임별
  탐지+연관+PnP 실측), detector=runs/export/crater_int8_ort.onnx(ORT CPU 1스레드),
  seed 0~199, 직렬 1워커(Unity 서버 1개), frames_dir 없음(PNG 미저장), 런당 ~35 s·총 ~2 h.
  시드별 체크포인트(jsonl, .gitignore) 재개 지원.
- 완료 200/200, 통신 실패 시드 0. **발산 1런(seed 14)**: 첫 프레임부터 연관 실패
  (매칭 26·인라이어 0~6, PnP 오차 km급) → 게이트 대량 기각(수락 11%) 중 초기 3개 통과
  → 필터 발산. 최근접 연관의 lost-in-space 한계(TODO oct) 실증 사례. 집계에서
  n_diverged로 분리 보고(CEP는 착륙 199런 기준).
- 핵심 수치: docs/results_summary.md "실런 MC" 행(= results/p8_unity_mc.json,
  p8_summary.json)만 인용. 요지: 실런 CEP는 통계 MC 대비 배수(ratio_unity_over_stat)이며
  갭의 지배 성분은 분산이 아니라 계통 바이어스(unity_bias_norm_m ≈ CEP,
  R95/CEP 1.39 vs 통계 2.32). 프레임 단위 PnP 산포(고도 bin 로버스트 σ)는
  데이터셋 iid보다 오히려 작음 — figs/p8_err_vs_altitude_inloop.png.
- 테스트 결과: `pytest -q` 68 passed — tests/test_p8.py 3건 신규(체크포인트 재개,
  meas 중복 제거, 발산 런 분리 집계).
- 생성 파일: results/p8_unity_mc.json, p8_summary.json, figs/p8_landing_dispersion.png,
  p8_err_vs_altitude_inloop.png, p8_tau_inloop_hist.png, scripts/run_mc_unity.py,
  analyze_p8.py, sim/measurement.py에 tau_det_s·h_true_m 기록 추가,
  results_summary.md 매니페스트 6행 추가(34행).
- 수동 작업 필요: 없음.
- 다음 단계: 바이어스 원인 분해(합성 씬-카탈로그 정합·시간상관·IMU bias, 10월),
  슬라이드에 실런 MC 산포도 반영 여부 판단(스토리보드 슬라이드 9/13 후보).

## P7c · 고전(PCA 템플릿) 탐지 베이스라인 비교 — mAP·CEP 양축 (2026-09-03)

- 완료:
  - `perception/classic.py` — PCA 외형 부분공간 템플릿 + Fisher 판별(SLIM 계열 비DL 근사).
    학습 스트리밍(메모리 상한)·임계 튜닝은 prefix 누적 F1 최대. `.npz` 28 KB.
  - `perception/metrics.py` — 계열 무관 단일 지표 구현(AP/mAP50-95/TP·FP·FN).
    `perception/detect.py`가 `.npz`를 `pca` 백엔드로 디스패치 → 연관·PnP·EKF 경로 그대로 사용.
  - 스크립트 3종: `scripts/train_classic.py`(학습 + 고도 사전정보 사본 저장),
    `scripts/eval_classic.py`(같은 val·같은 지표로 mAP·P/R·τ), `scripts/compare_classic_cep.py`
    (측정통계 × τ 2×2 MC). `calibrate_measurement.py --model` 단일 모델 보정 추가.
- 테스트 결과: `pytest -q` 65 passed (52.8 s) — tests/test_classic.py 8건 신규
  (AP 정의, greedy 매칭, 학습 결정론, 저장·복원 동일성, 스케일 제한, 백엔드 디스패치).
- 생성 파일: results/p7c_classic_train.json, p7c_det_compare.json, p7c_cep_compare.json,
  measurement_model_classic.json, figs/p7c_det_compare.png, p7c_cep_compare.png,
  p7c_pnp_error_hist.png, figs/slides/slide_13a·13b, docs/classic_baseline.md,
  runs/classic/pca_crater{,_prior}.npz(비커밋).
- 핵심 수치(val 112프레임 / MC n=200·직렬·보상 on):
  - 탐지: 고전(고도 사전정보) mAP50-95 **0.093**·recall 0.256·τ **218.8 ms**
    vs YOLO11n INT8 **0.983**·0.999·**151.6 ms**. 사전정보 없는 전수 스케일은
    mAP50 0.106·τ 498.4 ms — 미세 스케일 탐색이 비용·오검출의 대부분.
  - 측정: σ수평 460.4 m vs 87.9 m, 오검출률 0.151 vs 0.101, PnP 유효율 0.921 vs 0.980.
  - **CEP 664.4 m [605, 753] vs 126.6 m [105, 143] — 5.2배.** τ를 서로 바꿔도 CEP 불변
    (둘 다 프레임 주기 1 s 미만 → 보상이 흡수). 계열 격차는 지연이 아니라 측정 품질로 들어온다.
  - ultralytics val 교차확인: FP32 mAP50-95 0.994(ultralytics) vs 0.997(본 구현) — 보간 차 수준.
- 특이사항: ① 학습셋 888장 전체 적재 시 4.3 GB 스와핑 → 프레임 스트리밍 + reservoir로 교체
  ② 임계 튜닝 격자 탐색이 점수 범위 확대로 폭주 → prefix 누적 방식으로 교체
  ③ eval의 fp32 기본 경로가 runs/detect/runs/train/... 임(p5_det.json과 동일).
- 수동 작업 필요: 없음.
- 다음 단계: 슬라이드 부록에 13a·13b 배치(본편은 3분 제약상 제외 권장), 10월 본실험에서
  고전 방식도 MC 500회·고정소수점 구현 비용 추정으로 확장.

## 문서 갱신(9/5분 앞당김) + P8 다이어그램 (2026-09-02)

- 발표 스토리보드 docs/storyline.md (15장 흐름·자산 매핑·금지 수치 목록) — 커밋 45eff17.
- SLIM 비교표를 p7b_baseline(calibrated 110.8 m) 기준으로 재생성 + assumed 가드.
  team_status.md 전면 갱신(구 수치 정정 공지 포함).
- PROMPTS 문서갱신 세션 4항목 실행: limitations §3 확장(지터 모델 범위·직렬/병렬·밴드
  하한·지향 미정량·구동기 이상화), P8 신규성 절 교체(기여 4 — 기여③은
  p7b_tau_serial로 확인 후 결과 문장으로 확장), docs/symbols.md 생성, qa_facts 출처
  지시 추가 — 커밋 941aee7.
- scripts/make_diagrams.py: 슬라이드 5(파이프라인)·6(직렬 모델+지연 보상) 다이어그램
  → figs/slides/slide_05_pipeline.png, slide_06_serial_model.png.
- **시연 영상(산출물 ⑤) 1차본**: frames/p6 오버레이 99장 → figs/slides/p6_demo.mp4
  (H.264 yuv420p, 4 fps, 24.75 s, 4.6 MB, 자막·로고 없음). 인코더는 venv의
  imageio-ffmpeg 동봉 ffmpeg. 재현: `ffmpeg -framerate 4 -i frames/p6/%05d.png
  -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart figs/slides/p6_demo.mp4`.
  Unity Game 뷰 실캡처(궤적·텔레메트리 뷰) 추가 여부는 사용자 판단 대기.
- **시연 영상 세트 완성(9/2 저녁)**: Unity 캡처를 카메라별 RenderTexture 3화면 동시
  저장으로 개선(Game 뷰 무관, 재녹화 시 이전 프레임 자동 삭제) → 사용자 Play 1회
  (475프레임×3) → 16 fps 인코딩 **29.69 s** (30 s 규격 내): display2_landing.mp4(추적),
  display3_detection_synced.mp4(오버레이 동기), display4_telemetry.mp4(그래프),
  display4_telemetry.png(최종 정지 그래프). 추적·텔레메트리 시각 동기 확인(T 237 s ↔
  ALT 6.35 km). 슬라이드 9는 landing+detection 중 택1 또는 병렬 배치.
- **P8 자산 세트 완료(9/2)**: make_slide_assets.py에 results_summary(숫자↔출처↔git 자동
  추출 21행)·슬라이드 그림 복사 추가 → docs/results_summary.md, figs/slides/slide_*.png.
  docs/qa_facts.md 작성(계획서 대조: 산출물 ①~⑤ 매핑, 선행연구 표, 신규성 4, 자주 나올
  질문, 밴드 22 km·MC 500 10월). docs/symbols.md와 함께 Q&A 문서 완비.
- 남은 것: 슬라이드 본문 제작(팀 작업, 9/10 동결) — 자산·수치·스토리보드·Q&A 전부 준비됨.
  선택: P7c(카메라 지향, 조건부), 10월 본실험(IMU bias·시간상관·n=500·기하 매칭).

## P7b+ · 보상 성립 조건 실험 — 타임스탬프 지터·카메라 레이트 (2026-09-01 저녁)

- 동기: 사용자 지적 "결국 다 보상하면 그만" → 보상 만능은 이상화(정확한 t_c·바이어스
  없는 IMU·τ<주기)의 산물임을 실험으로 정량화. 채택 2종 구현·실행 완료(n=200/조건).
- **① 타임스탬프 지터 스윕** (τ=196 ms, 직렬, 보상 on) — p7b_jitter_sweep.json:
  ≤20 ms 무해(110.8→115.9, CI 겹침 — IMU 스텝 10 ms 반올림 흡수), 100 ms부터 유의
  (136.5), **200 ms(≈τ, "도착시각=촬영시각" 구현 실수에 해당)면 194.9 m로 미보상
  (159.9)보다 나쁨**, 500 ms 349.5 m·게이트 수락 0.94→0.60 붕괴.
  → "지연 보상의 진짜 요구사항은 계산력이 아니라 촬영 시각 태깅"(교차점 ~140 ms).
- **② 카메라 레이트 스윕** (1/2/5 Hz × CPU 실측 τ 4점, 보상 on) — p7b_rate_sweep.json:
  | rate | n INT8(196 ms) | n FP32(219 ms) | s INT8(376 ms) | s FP32(537 ms) |
  |---|---|---|---|---|
  | 1 Hz | 110.8 | 110.8 | 110.8 | 110.8 |
  | 2 Hz | 82.5 | 82.5 | 82.5 | **115.5 (drop 50%)** |
  | 5 Hz | **48.7 (drop 0)** | 65.7 (drop 50%) | 65.7 (drop 50%) | 88.4 (drop 67%) |
  → 레이트↑ = 측정↑ = CEP↓이므로 고레이트가 이득인데, 고레이트일수록 τ가 주기 임계에
  물림. **5 Hz에서 양자화 단독(n FP32→INT8, τ 23 ms 차)이 임계(200 ms)를 사이에 두고
  65.7→48.7 m CI 분리** — 보상 켠 채로 양자화→τ→CEP 인과 직접 증명(P7b 목표 달성).
- 배관: run_closed_loop t_c_jitter(0이면 rng 미소비, 골든 유지), config
  tau.t_c_jitter_*·camera.rate_sweep_hz(본값 rate_hz 불변). pytest 57 passed.
- 잔여 아이디어(미실행): 상수 오프셋 지터 변형, IMU 바이어스 주입(10월 TODO),
  AR(1) 시간상관 측정 오차, 부분 보상 비교.

## P7b 실행 · calibrated 첫 수치·구 P7 오라벨 정정·CPU 벤치 (2026-09-01 오후)

- **[정정] 구 P7 "calibrated 81.4 m"는 assumed였다**: aggregate_mc.py가 mode가 아니라
  보정 파일 존재로 라벨을 달아(구 라인 80) 잘못 표기. 현재 코드로 assumed 조건 재현 시
  81.44394950773159 **비트 일치**로 확증. → aggregate는 개별 파일 기록 우선(없으면
  None=경고), run_mc가 assumed_measurement_stats를 직접 기록하도록 수정.
  **구 p7_mc.json 수치는 슬라이드 사용 금지**, 아래 p7b_*가 calibrated 공식 수치.
- **τ 직렬 스윕 (calibrated, n=200/조건, 격자4+실측5)** — results/p7b_tau_serial{,_compoff}.json:
  | τ | 보상 CEP | 미보상 CEP |
  |---|---|---|
  | 50 ms~1 s (실측 4점 포함) | **110.8 m 평탄** | 114→160(n INT8)→165(n FP32)→242(s INT8)→323(s FP32)→556(1 s) |
  | 2 s (드롭 50%) | 177.8 | 1044.9 |
  | 5 s (드롭 80%) | 250.4 | 2160.5 |
  → **임계·마진 구조**: 보상 시 τ<프레임 주기(1 s)는 전부 흡수, 넘으면 드롭으로 악화.
  미보상 시 양자화 효과는 s에서 유의(323→242, CI 분리), n에서 비유의(165→160, τ 이득
  23 ms뿐). 경량화(s→n INT8) 유의(242→160).
- **fp 스윕 (산출물 ④)** — p7b_fp_sweep.json: fp 0→0.3에서 CEP 112.7→150.8, 기각률
  0.009→0.319. 보정 fp(0.101) 근방 126.6 m. baseline(fp 0.05) 110.8 = 스윕 0.05점과 일치.
- **오프셋**: baseline(381.4 m) 110.8 vs fp2000 118.9 — CI 겹침. s 통계 MC(p7b_mc_s,
  s INT8 τ) 111.3 = n과 동급 (측정 품질 통제 확인).
- **CPU 1스레드 벤치** — p7b_cpu_bench.json: CoreMark 34,876 / 40,654 DMIPS =
  **RAD750 102배, JAXA HR5000(320 DMIPS, MIPS64 200 MHz) 127배, GR740 코어당 68~96배,
  HPSC 칩 목표(~100×RAD750)와 같은 자릿수** → 현재 τ 실측 = 차세대(HPSC급) 프록시.
  SLIM SMU의 CPU 기종 공개 문서는 미확인(HR5000은 JAXA 세대 대표값), SLIM 영상처리는
  RTG4 FPGA(≤5 s). 도구: zig cc(venv ziglang) -O2, CoreMark/Dhrystone(스크래치패드 클론).
- **온보드 재현 방안 검토** — docs/onboard_cpu_emulation.md: τ 스케일링 채택(HR5000급
  τ≈20~25 s → TRN 불가 = SLIM이 FPGA 쓴 이유와 정합; HPSC급 ≈ 실측 그대로).
  Job Object 사용률 제한은 10월 스팟체크 후보, QEMU·클럭 제한 기각.
- 다음: P7c(조건부) 또는 문서 갱신 — 슬라이드 수치를 p7b_*로 교체 필수. (선택) 카메라
  레이트 축 스윕(주기 0.5/0.2 s면 s가 드롭 영역 진입 — 보상 하 양자화 인과 직접 증명).

## P7b 코드 세션 · 직렬 처리 모델·fp 스윕·s 보정·SLIM 표 (2026-09-01)

- **measurement.mode → calibrated 전환**(config 기준값). 기존 테스트는 assumed 고정
  (deepcopy)으로 분리. run_mc 산포도 제목 assumed/calibrated 동적 표기.
- **직렬 처리량 모델**(tau.serial, P7b 핵심): 직전 프레임 처리(busy_until) 중이면 촬영
  생략(n_dropped), τ·측정 rng는 실제 촬영 시에만 소비. serial=false 경로는 수정 전
  커밋의 골든 값과 비트 동일 — tests/test_p7b.py 5개로 고정. 1 Hz·τ=2.5 s급에서
  드롭 비율 ~1/3 확인. "무한 병렬 촬영" 가정 제거로 '보상 시 CEP 평탄' 주장 방어.
- **run별 extras**: n_meas/n_dropped/게이트 수락률/ΔV(Σ|a_T|·dt)/fp_offset_used_m —
  run_mc 3-튜플 반환 + extras_summary(). run_closed_loop.py 출력에도 추가.
- **fp_offset 우선순위**: CLI --fp-offset > calibrated 파일 fp_offset_med_m(381.4 m) >
  config 2000 m. baseline vs fp2000 비교 배관 완료.
- **스크립트**: sweep_tau.py 조건 리스트 모드(직렬 격자 4 + 실측 τ 5, CEP·측정수 이중축),
  sweep_fp.py 신규(연구계획서 산출물 ④ — fp_sweep 5점, calibrated 0.101 세로선),
  run_mc.py --tau-file/--serial/--fp-offset/--measurement-file 추가.
- **SLIM 비교표 생성**: results/p7_slim_comparison.json — 측정 성공률 0.98 vs 14/14,
  σ수평 ≈85 m(GSD 25~34 m/px → 수 px 수준), CEP 81.4 m vs ~10 m. plausibility 수위 한정,
  착지 55 m 이탈(추진 고장)은 비교 제외. 출처: Acta Astronautica Vol.226 (2025).
- **s 보정 완료**: results/measurement_model_s.json — INT8 s σ [82.2, 88.2, 30.8] m,
  fp_rate 0.082, offset 381.7 m, valid 0.97 — **n([87.9, 81.8, 32.9]·0.101)과 동급**.
  → n vs s의 CEP 차이는 측정 품질이 아니라 τ가 원인이라는 주장의 직접 근거.
- **ΔV truth 기준**: 1109.5 m/s(참값 유도, 착륙 0.0013 m) → results/p7b_deltav.json.
  팀원 로켓 방정식 입력용 — 우리 쪽 해석·주장에는 쓰지 않는다.
- 테스트: `pytest -q` **53 passed**.
- **수동 작업 필요(tmux, 사용자)**: ① τ 직렬 스윕 ② baseline ③ baseline fp2000
  ④ fp 스윕 ⑤ p7b_mc_s — 명령 세트는 세션 로그 참조.
- 다음: 사용자 MC 완료 후 결과 검토·그림 확인 → P7c(조건부) → 문서 갱신(9/5) → P8.

## trn_band 확정·P6/P7 최종·일반화 test (2026-08-29 낮)

- **h_min ablation → 22000 m 사용자 확정**: ① σ-고도 분석(944프레임): h≥22 km σ수평
  85~186 m·매칭 27+, 19~22 km 전이(σ 최대 1937 m, 매칭 5), h<19 km 붕괴
  (p5_sigma_vs_altitude) ② iid 통계 스윕은 17 km 최적(49.5 m)이라 답하지만
  ③ unity 실런 4시드는 h22 중앙값 257 m vs h19 483 m — iid 가정이 시간상관 편향을
  못 담아 밴드 설계에서 틀린 답을 냄 = 폐루프 평가 필요성의 실증(p5_hmin_tradeoff,
  p6_h{19000,22000}_s*).
- **게이트 연쇄 기각 디버그 사슬**(재현 자료 results/p6_diag*.json): 지속 z-편향 구간에서
  첫 기각 후 영구 잠금(착륙 3.6~6 km) → τ 상수 판별 실험으로 필터 취약성 확정 →
  P 팽창(연속 5회 기각 시 4배) + 팽창 예산(3회/비행) 도입, 게이트 판정식 §2.4 불변.
- **P6 공식(h22)**: 착륙 197.8 m, 연착륙 0.65 m/s, 수락 75/99, 사슬 τ 실측 133 ms.
- **P7 최종(h22, calibrated σ [87.9,81.8,32.9]·fp 0.101)**: 보상 CEP **81.4 m 전 τ 평탄**,
  미보상 89.5(50 ms)→2379 m(5 s). 실측 τ 지점: n INT8 CPU(196 ms) 144.2 m vs
  s INT8 CPU(376 ms) 238.1 m — mAP 동급인데 CEP 1.65배 차 = "mAP 아닌 CEP로 평가" 실증.
  통계 81 m vs 실런 198 m 갭 = 시간상관 편향(iid 미포함), TODO(oct) bias 상태.
- **일반화 test(미학습 고지대 val 99프레임)**: n mAP50 0.995→0.354(R 0.265),
  s 0.995→0.244 — s가 더 과적합. 프레임: 한계가 아닌 TRN 설계 특성(사전 측량 착륙지
  특화)이며 "역시나" 한 줄로 보고(사용자 확정). 정밀도 0.66~0.72 유지, overlay 정합 검증.
- 전 커밋 push 완료(origin/main). 다음: P8 슬라이드 자산.

## 밤샘 자율 세션 · P5~P7 (2026-08-29 새벽)

- **P5 완료**: 데이터셋 1000프레임(888/112 궤적 분리, 평균 36.5 라벨/프레임) →
  yolo11n 100ep 학습(mAP50-95 0.994/INT8 0.980, R≈1.0) → INT8 양자화(실렌더 캘리브레이션,
  TRT 엔진 박스 수 fp32와 일치) → 측정 보정 results/measurement_model.json.
- **비교용 yolo11s도 학습**: mAP50-95 0.995/INT8 0.986 — n 대비 +0.001~0.006, "아는 지도"
  도메인에선 n 포화. 결론은 τ 벤치(아침) 후 CEP 축으로 확정.
- **측정 보정 방법론 확정(P6 디버그의 산물)**: R = 0 기준 로버스트 σ(3σ 클리핑 RMS).
  근거: EKF에 bias 상태가 없어 구간·태양각 따라 이동하는 계통 편향(±50~120 m,
  z ~50 m)이 혁신에 남는데, 중앙값 기준 σ(z 26 m)로는 χ² 게이트가 연쇄 기각
  (수락 21/125, 착륙 3584 m). 0 기준 σ [97.6, 90.2, 33.6] m로 수락 89/143,
  착륙 138.9 m. 대형 실패는 fp_rate_est 0.127·offset ~470 m로 분리 보고(게이트 담당).
  저고도(<19 km) 측정 품질 급락(σ_y ~959 m)은 게이트가 걸러냄 — trn_band 하한 상향 검토
  또는 고도 의존 R은 TODO(oct), 사용자 판단 대기.
- **P6 완주 (Plan A 성립)**: Unity 실렌더→TRT INT8→연관→PnP→EKF(지연 보상, τ=wallclock)
  → 착륙 오차 **138.9 m**, 연착륙 0.43 m/s, 전체 사슬 τ median **165 ms**,
  frames/p6 오버레이 147장. results/p6_closed_loop.json (+진단 p6_diag.json, gate_log 포함).
- **P7 calibrated 12조건(n=200)**: 보상 시 CEP **41.9 m 전 τ 평탄**, 미보상 τ 비례
  52.5 m(0.05 s)→5558 m(5 s). 실측 σ·fp_rate 기반 — assumed 아님.
- 밤중 이슈 해결 로그: 학습 pin-memory 크래시(워커 8→2), 학습+MC 병렬 RAM 고갈(직렬화),
  eval_det ONNX CUDA 바인딩(cpu 강제), bench_tau 자체 양자화 sanity 실패(export 산출물은
  정상 — bench와 export 양자화 설정 통일 필요, 아침 항목).
- **아침 남은 일**: ① Unity 끄고 클린 τ 벤치(n·s, 4백엔드) — 밤 측정치는 Unity 점유로
  오염(TRT FP32 66 ms 등) ② 실측 τ 지점 MC 6조건 ③ aggregate_mc 집계·그림
  ④ (선택) 고지대 held-out 일반화 test ⑤ n vs s CEP 비교표.

## P4 완료 · Unity 렌더 서버·투영 정합 검증 통과 (2026-08-28)

- Unity 6.5(6000.5.10f1), BIRP, 프로젝트 루트 = unity/ (Hub가 만든 하위 폴더에서
  Packages/ProjectSettings 이동으로 전환). 파생 에셋(LunarTerrainData 34.6 MB 등)은 gitignore.
- 세션 중 잡은 버그: ① SceneBuilder의 Unity 가짜 null `??` 패턴 → Light 예외 (명시적 == null로 교체)
  ② 텍스처를 메모리 Texture2D로만 로드 → 도메인 리로드 시 체크무늬 (에셋 임포트로 영구화, 4096 설정)
  ③ 원거리 base map 저해상도 + 주변광 과다로 뿌연 렌더 (RenderServer ConfigureQuality:
  basemapDistance 500 km, 주변광·안개 제거, 그림자 60 km, 배경 검정)
- **정합 검증 통과**: check_projection을 Hough → WAC 텍스처 템플릿 매칭(NCC, 고역 통과,
  크레이터별 카탈로그 고도 평면으로 시차 보정)으로 교체. 밴드 중간 고도 공식 포즈
  평균 오차 1.5 px, 추가 3개 포즈 0.9~2.2 px — 기준 5 px 통과 (반전·오프셋·스케일 오류 없음,
  전역 위상상관 (1,−1) px). → results/p4_projection_check.json, figs/p4_projection_check.png
- RenderServer previewOnScreen: Play 중 Game 뷰에 마지막 렌더 프레임 표시(검수용).
- 다음: P5 — Unity 서버 Play 상태에서 make_dataset → train(사용자 GPU) → export_int8 →
  eval_det → calibrate_measurement. 탐지 overlay 검수는 P5 figs/p5_overlay_*.png와
  P6 --frames-dir 프레임으로 가능.

## 계획 추가 · 미학습 지역 일반화 test (2026-08-28 사용자 지시)

- 배경: 학습·검증·폐루프가 전부 SLIM 박스 한 곳 → "특정 지점 과적합" 질문에 대비.
  train/val은 SLIM 유지가 맞음(카탈로그 매칭 TRN은 사전 측량 착륙지가 배포 도메인이고,
  P5 측정 보정 σ가 SLIM 폐루프에 들어가므로 같은 site여야 함). 단 val mAP는
  "아는 지도에서의 성능"임을 슬라이드·Q&A에 명시한다.
- **추가 항목(P5 학습 완료 후, 크리티컬 패스에 여유 있을 때)**: 남부 고지대 박스를
  순수 held-out test로 사용 — 학습에 전혀 안 쓴 지역, 지질 유형도 다름(고지대 vs 바다 인접).
  1. Unity 지형을 data/processed/highlands/(P3 site 비교 산출물, 카탈로그 642개)로 재빌드
  2. 밴드 내 포즈 ~200프레임 렌더 + 투영 자동 라벨
  3. scripts/eval_det.py로 mAP·FP/frame 측정 → results/p5_det_highlands.json
  4. 보고 형식: "SLIM(val) mAP X vs 미학습 고지대 mAP Y, 갭 Z" — 일반화 근거 한 줄
- 시간 없으면 10월 항목으로 이월하되 qa_facts.md에 계획 자체를 명시.

## P3 보완 · 궤적-박스-밴드 정합 (2026-08-27)

- 감사에서 발견: 시나리오 r0(동쪽 −150 km)가 박스 220 km(±110) 밖 → TRN 밴드 시작 시점에도
  기체 x=−118.9 km로 박스 밖, 밴드 구간의 28%에서 카메라 풋프린트가 지형 데이터 밖.
  통계 모드(P1·P7)에선 무해했지만 Unity 렌더·데이터셋에는 치명적이었음.
- 사용자 승인으로 box_km 220→**340**×60 확장, 재크롭·재분석: catalog 566개(+303),
  밴드 제안 16472~30000 m → trn_band 16500~30000 m 확정.
- 3자 정합 재검증: 전체 궤적에서 풋프린트 100% 박스 내 포함(밴드 구간 포함). pytest 47 passed.
- 구 P1/P7 수치(CEP 31~34 m)는 구 trn_band(15–45 km)·가정 통계 기준 → 슬라이드 사용 금지,
  P7 재실행 시 갱신(results meta config_hash로 구분).

## P3 완료 · site 확정 (2026-08-27)

- 사용자 확정: site = SLIM 착륙점(LROC 측정 13.3160S, 25.2510E) = L 원점.
  config.yaml site/trn_band 반영(trn_band 16500–27900 m = 제안 16472–27954의 안쪽 반올림).
- 최종 산출(data/processed/): dem_L.npz(dz −3792~+3673 m), texture_L.png, heightmap.raw,
  catalog_L.csv 263개. results/p3_altitude_band.json, figs/p3_registration.png, p3_altitude_band.png.
- P7 프롬프트에 SLIM 공개 비행결과(Acta Astronautica Vol.226) 정합성 비교 명시 —
  L 원점이 SLIM 착륙점과 동일하므로 비교표 성립.
- 다음: P4 Unity Editor 수동 작업(unity/README.md, 9/3 정오 시한)이 크리티컬 패스.

## P3 실행 · site 후보 비교 (2026-08-27)

- 원본 데이터 확보(data/raw, 총 3.7 GB): SLDEM2015 512ppd FLOAT 타일 2장(0–30S, 30–60S / 0–45E),
  WAC E300S0450 GeoTIFF(474 MB, EXTRAS/BROWSE 경로 — DATA/BDR 밑 .TIF는 404),
  Robbins PDS4 번들(astropedia 다운 서버 사망 → CKAN 미러). scripts/download_raw.sh로 재현.
- crop.py 수정 2건: ① SLDEM(PDS)·WAC(GeoTIFF)는 등장방형 투영 미터 좌표 →
  CRS 인식해 경위도 변환 후 보간 ② rasterio merge()가 PDS nodata(-3.4e38)를 못 다뤄
  전부 0을 반환 → 단일 파일은 직접 read. SLDEM 고도는 km 단위(--dem-scale 1000).
- 후보 비교(scripts/p3_frontend.sh, config_site_{slim,highlands}.yaml):
  | | SLIM(−13.3, 25.2) | 남부고지대(−42, 14) |
  |---|---|---|
  | 박스 내 D≥1km | 260개 | 642개 |
  | n≥8 통과 밴드 | 19.7–28.0 km | 12.9–27.6 km |
  | 시야 내 최대 | 41개 | 70개 |
  둘 다 n_min 8 통과. 정합 그림(figs/p3_registration_*.png) 양호.
- 산출물: data/processed/{slim,highlands}/(dem_L.npz, texture_L.png, heightmap.raw, catalog_L.csv),
  results/p3_altitude_band_{slim,highlands}.json, figs/p3_{registration,altitude_band}_*.png
- 대기: 사용자 site 확정(1순위 SLIM 통과 확인됨) + config.yaml site/trn_band 반영(사용자 항목),
  이후 data/processed/ 최종본 생성 → P4 Unity 수동 절차.

## P2 추가 · TRT FP16 벤치 (2026-08-27)

- perception/bench_tau.py에 trt_fp16 백엔드와 `--backends` 부분 실행 옵션 추가.
  TRT 11은 BuilderFlag.FP16이 제거되어(strongly typed) ultralytics `half=True`로
  FP16 ONNX(yolo11n_half.onnx)를 내보내 엔진을 빌드하는 경로로 구현.
- 결과(동일 세션 재측정): TRT FP32 median 20.01 ms / p95 21.76 ms,
  **TRT FP16 median 22.18 ms / p95 26.18 ms — FP32보다 오히려 느림.**
  sanity IoU(FP16 vs FP32) 0.997, 72/72 박스 매칭 — 정확도 손실은 없음.
- 해석: τ_det에서 GPU 행렬연산 비중이 작고 전처리(CPU letterbox)+NMS+복사·동기화가
  지배적. FP32 엔진은 이미 TF32 텐서코어를 쓰므로 FP16 이득이 없고, FP16 입출력
  캐스트 오버헤드만 추가됨. → 슬라이드에는 "GPU에서는 정밀도 축소가 τ를 더 줄이지
  못한다(병목은 전·후처리)"의 근거로 사용 가능. 온보드 CPU/NPU 환경(ORT)에서는
  INT8이 유효(207→179 ms)하다는 기존 결론 유지.
- 산출물: results/tau_trt_fp16.json, figs/p2_tau_hist.png(5패널), logs/p2_fp16_bench.log

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
