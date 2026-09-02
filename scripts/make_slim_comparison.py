"""SLIM 정합성 비교표 (P7 프롬프트 명세): results/p7_slim_comparison.json.

우리 시뮬 수치(results/*.json)를 SLIM 공개 비행 결과와 나란히 표로 만든다.
외부 기준값 출처: "Vision-based navigation and obstacle detection flight results in
SLIM lunar landing", Acta Astronautica Vol.226 (2025).
주장 수위는 validation이 아니라 plausibility(자리수·경향 일관)로 제한한다 —
센서 구성(지도 매칭+레이더+LRF vs YOLO+PnP)과 자유도(6-DOF vs 3-DOF)가 다르고,
우리 TRN 밴드 하한(h_min) 아래는 IMU 단독 전파라 저고도 비교는 구조적으로 다르다.
SLIM 실제 착지점 이탈 ~55 m는 엔진 노즐 탈락(추진 고장) 영향이므로 비교 기준으로 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mc import result_meta  # noqa: E402

SLIM_SOURCE = ("Vision-based navigation and obstacle detection flight results in SLIM "
               "lunar landing, Acta Astronautica Vol.226 (2025)")
# 공개 논문에 보고된 수치 (외부 기준값 — 우리 산출물이 아님)
SLIM = {
    "vbn_success": "14/14 (CST1/2, VLD1/2 전부 성공)",
    "nav_err_horiz_m_at_500m": "< 1 m (고도 500 m 시점)",
    "landing_precision_m_at_50m": "~10 m (고도 50 m 시점 평가, 목표 100 m)",
    "excluded": "착지점 이탈 ~55 m — 엔진 노즐 탈락(추진 고장) 영향이라 비교 기준 제외",
}


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # CLI 규약 통일용
    ap.add_argument("--out", default="results/p7_slim_comparison.json")
    ap.add_argument("--baseline", default="results/p7b_baseline.json",
                    help="대표 조건 MC (calibrated, τ=n INT8 CPU, comp on, serial)")
    ap.add_argument("--p6", default="results/p6_closed_loop.json")
    ap.add_argument("--measurement", default="results/measurement_model.json")
    ap.add_argument("--tau-file", default="results/tau_ort_cpu_int8.json",
                    help="대표 조건 τ (n INT8 CPU median)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    site = cfg["site"]
    if abs(float(site["lat0_deg"]) + 13.316) > 0.01 or abs(float(site["lon0_deg"]) - 25.251) > 0.01:
        raise SystemExit("site가 SLIM 착륙점이 아니다 — 비교표가 성립하지 않는다.")

    meas = _load(args.measurement)
    p6 = _load(args.p6)
    tau_med = float(_load(args.tau_file)["median_s"])

    # 대표 조건: p7b_baseline (calibrated 확정 수치 — 구 p7_mc는 assumed 오라벨로 사용 금지)
    base = _load(args.baseline)
    if base.get("assumed_measurement_stats"):
        raise SystemExit(f"{args.baseline}이 assumed 통계다 — SLIM 표에 쓸 수 없다.")
    rep = {"cep_m": base["cep_m"], "tau_s": float(base["params"]["tau"]),
           "n_runs": int(base["params"]["n_runs"]), "file": args.baseline}
    if abs(rep["tau_s"] - tau_med) > 0.01:
        raise SystemExit(f"baseline τ({rep['tau_s']})가 {args.tau_file} median({tau_med})과 다르다.")

    # 카메라 GSD(nadir): f[px] = H/(2·tan(θ_v/2)), GSD(h) = h/f — 픽셀 정규화 비교용
    f_px = float(cfg["camera"]["H"]) / (2.0 * math.tan(math.radians(float(cfg["camera"]["fov_v_deg"])) / 2.0))
    h_min = float(cfg["trn_band"]["h_min_m"])
    h_max = float(cfg["trn_band"]["h_max_m"])
    gsd_min, gsd_max = h_min / f_px, h_max / f_px
    sigma = meas["sigma_xyz_m"]
    sigma_h = (sigma[0] ** 2 + sigma[1] ** 2) ** 0.5 / math.sqrt(2.0)  # 수평 축 평균 σ
    sigma_px = [sigma_h / gsd_max, sigma_h / gsd_min]  # 밴드 상단(멀다)~하단

    rows = [
        {
            "quantity": "영상 항법 측정 성공률",
            "slim": SLIM["vbn_success"],
            "ours": (f"보정 valid {meas['valid_ratio']:.2f} ({meas['n_valid']}/{meas['n_frames']}), "
                     f"P6 실런 게이트 수락 {p6['n_measurements'] - p6['n_gate_rejected']}"
                     f"/{p6['n_measurements']}"),
            "note": "우리는 D≥1 km 카탈로그·자동 연관 기준",
        },
        {
            "quantity": "항법 수평 오차",
            "slim": SLIM["nav_err_horiz_m_at_500m"],
            "ours": (f"σ 수평 ≈ {sigma_h:.0f} m (고도 {h_min / 1e3:.0f}–{h_max / 1e3:.0f} km, "
                     f"GSD {gsd_min:.0f}–{gsd_max:.0f} m/px)"),
            "note": (f"픽셀 정규화 σ ≈ {sigma_px[0]:.1f}–{sigma_px[1]:.1f} px — 관측 해상도 대비 "
                     "수 픽셀 수준이라는 점에서 경향 일관 (고도·해상도가 달라 절대값 직접 비교 불가)"),
        },
        {
            "quantity": "착륙 정밀도",
            "slim": SLIM["landing_precision_m_at_50m"],
            "ours": (f"CEP {rep['cep_m']:.1f} m (τ={rep['tau_s'] * 1e3:.0f} ms 보상, "
                     f"n={rep['n_runs']}, calibrated 통계)"),
            "note": (f"우리는 h<{h_min / 1e3:.0f} km에서 IMU 단독 전파(측정 없음) — SLIM은 "
                     "저고도 지형 촬영을 착륙 직전까지 계속하므로 직접 비교가 아니라 자리수 비교"),
        },
    ]

    out = {
        "meta": result_meta(args.config),
        "claim_level": ("plausibility — 실제 임무 공개 비행 결과와 자리수·경향이 일관함을 확인. "
                        "validation 아님(센서 구성·자유도·측정 고도 구간이 다름)"),
        "slim_source": SLIM_SOURCE,
        "slim_excluded": SLIM["excluded"],
        "site": {"lat0_deg": site["lat0_deg"], "lon0_deg": site["lon0_deg"],
                 "note": "L 원점 = SLIM 착륙점(LROC 측정 13.3160S, 25.2510E)"},
        "our_sources": {
            "measurement": args.measurement,
            "p6": args.p6,
            "baseline": args.baseline,
            "tau": args.tau_file,
            "representative_condition": rep["file"],
        },
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        print(f"- {r['quantity']}: SLIM {r['slim']} | ours {r['ours']}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
