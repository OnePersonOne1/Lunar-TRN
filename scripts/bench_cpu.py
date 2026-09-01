"""CPU 1스레드 벤치 CLI: CoreMark(+Dhrystone) → 온보드 프로세서 공표치와 비교 json.

목적: τ 벤치를 돌린 CPU(1스레드)의 연산 수준을 우주 온보드 프로세서와 같은 지표로
비교해 "온보드 프록시" 가정의 근거를 만든다 (results/p7b_cpu_bench.json).
- CoreMark: https://github.com/eembc/coremark (posix 포트, 단일 스레드)
- Dhrystone 2.1: https://github.com/sifive/benchmark-dhrystone (빌드 실패 시 생략)
- 컴파일러: zig cc(clang, venv의 ziglang 패키지) -O2 — 점수와 함께 기록한다.
- 실행 중 프로세스 친화도를 코어 1개로 고정한다. MC 등 부하와 동시 실행 금지.
외부 기준값(RAD750/GR740/HPSC)은 공개 문서 수치이며 출처 문자열을 함께 기록한다.
"""
from __future__ import annotations

import argparse
import ctypes
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mc import result_meta  # noqa: E402

# 공개 문서에 보고된 수치 (외부 기준값 — 우리 산출물이 아님)
REFERENCES = [
    {
        "name": "BAE RAD750 (200 MHz)",
        "role": "레거시 rad-hard CPU (Curiosity, Perseverance, JWST)",
        "dmips": 400.0,
        "coremark": None,
        "note": "~266 MIPS @110-200 MHz, Dhrystone 2.1 ~400 DMIPS @200 MHz",
        "source": "BAE Systems / Wikipedia RAD750; militaryaerospace.com 16723785",
    },
    {
        "name": "Gaisler GR740 (quad LEON4FT, 250 MHz)",
        "role": "ESA 차세대 rad-hard SoC (NGMP)",
        "dmips": 425.0,          # 코어당 1.7 DMIPS/MHz × 250 MHz (1스레드 비교용)
        "coremark": 511.69,      # 코어당 CoreMark 1.0
        "note": "칩 전체 1800 DMIPS(쿼드). 여기 값은 코어당(1스레드 비교용)",
        "source": "Gaisler GR740 Product Brief 2024-12 / GR740-VALT-0010 benchmarking note",
    },
    {
        "name": "JAXA HR5000 계열 (MIPS64 5Kf, 200 MHz)",
        "role": "JAXA 표준 64-bit rad-hard MPU (H-IIA/B GCC 등). SLIM SMU의 CPU 기종을 "
                "공개 명시한 문서는 미확인 — JAXA 세대 대표값으로만 사용",
        "dmips": 320.0,
        "coremark": None,
        "note": "320 DMIPS @200 MHz, ~4 W. HR5000S는 FD-SOI 공정 파생(상세 미공개). "
                "SLIM의 영상처리는 CPU가 아닌 RTG4 FPGA 담당(별도 행 참조)",
        "source": "wdic.org/w/SCI/HR5000; NEC SpaceWire WG 2008 ASIC dev; "
                  "JAXA H-IIB overview",
    },
    {
        "name": "NASA/Microchip HPSC (PIC64, RISC-V 8코어)",
        "role": "차세대 우주 컴퓨터 (EM칩 2025 Q3)",
        "dmips": None,
        "coremark": None,
        "note": "RAD750 대비 ~100배(NASA 목표), JPL 초기 시험 ~500배 보도 — 공식 단일지표 미공표",
        "source": "Microchip 'Dawn of the HPSC Era' white paper 2024; hothardware.com NASA RISC-V",
    },
    {
        "name": "JAXA SLIM 영상처리 (Microsemi RTG4 FPGA)",
        "role": "실제 달 착륙 TRN의 크레이터 탐지·매칭 하드웨어",
        "dmips": None,
        "coremark": None,
        "note": "CPU가 아닌 방사선 내성 FPGA 전용 로직 — 촬영→결과 5초 이내",
        "source": "Acta Astronautica Vol.226 (2025) SLIM VBN flight results; JAXA cosmos 해설",
    },
]

DHRY_DMIPS_DIV = 1757.0  # DMIPS = Dhrystones/s ÷ 1757 (VAX 11/780 기준)


def parse_coremark(text: str) -> dict:
    """CoreMark 실행 출력에서 점수·반복수·시간을 추출한다."""
    out: dict = {}
    m = re.search(r"Iterations/Sec\s*:\s*([0-9.]+)", text)
    if m:
        out["iterations_per_sec"] = float(m.group(1))
    m = re.search(r"Iterations\s*:\s*([0-9]+)", text)
    if m:
        out["iterations"] = int(m.group(1))
    m = re.search(r"Total time \(secs\)\s*:\s*([0-9.]+)", text)
    if m:
        out["total_time_s"] = float(m.group(1))
    m = re.search(r"Correct operation validated", text)
    out["validated"] = bool(m)
    return out


def parse_dhrystone(text: str) -> dict:
    """Dhrystone 실행 출력에서 Dhrystones/s를 추출한다 (포트별 표기 차이 허용)."""
    m = re.search(r"Dhrystones per Second\s*:\s*([0-9.]+)", text)
    if not m:
        m = re.search(r"DMIPS\D+([0-9.]+)", text)
        if m:
            return {"dmips": float(m.group(1))}
        return {}
    dps = float(m.group(1))
    return {"dhrystones_per_sec": dps, "dmips": dps / DHRY_DMIPS_DIV}


def _pin_one_core(proc: subprocess.Popen) -> None:
    """프로세스 친화도를 CPU 0 하나로 고정 (Windows)."""
    ctypes.windll.kernel32.SetProcessAffinityMask(int(proc._handle), 0x1)


def _run_pinned(exe: Path, timeout_s: float = 300.0) -> str:
    proc = subprocess.Popen([str(exe)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    _pin_one_core(proc)
    out, _ = proc.communicate(timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"{exe.name} 실행 실패 (exit {proc.returncode}):\n{out[-2000:]}")
    return out


def _zig_cc(py: str, args: list[str]) -> None:
    r = subprocess.run([py, "-m", "ziglang", "cc", *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"zig cc 실패:\n{r.stderr[-2000:]}")


def build_coremark(py: str, src: Path, build: Path) -> Path:
    exe = build / "coremark.exe"
    srcs = ["core_list_join.c", "core_main.c", "core_matrix.c", "core_state.c",
            "core_util.c", "posix/core_portme.c"]
    _zig_cc(py, ["-O2", f"-I{src}", f"-I{src / 'posix'}",
                 "-DPERFORMANCE_RUN=1", "-DITERATIONS=0", '-DFLAGS_STR="-O2"',
                 *[str(src / s) for s in srcs], "-o", str(exe)])
    return exe


def build_dhrystone(py: str, src: Path, build: Path) -> Path:
    exe = build / "dhrystone.exe"
    srcs = [p for p in ("dhry_1.c", "dhry_2.c", "dhry_stubs.c") if (src / p).exists()]
    _zig_cc(py, ["-O2", "-DTIME", "-DDHRY_ITERS=500000000", "-std=gnu89",
                 "-Wno-implicit-int", "-Wno-implicit-function-declaration", f"-I{src}",
                 *[str(src / s) for s in srcs], "-o", str(exe)])
    return exe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)  # CLI 규약 통일용 (벤치는 미사용)
    ap.add_argument("--out", default="results/p7b_cpu_bench.json")
    ap.add_argument("--src-coremark", required=True)
    ap.add_argument("--src-dhrystone", default=None)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--repeats", type=int, default=3, help="반복 중 최고 점수 기록")
    args = ap.parse_args()

    import json
    import platform

    py = sys.executable
    build = Path(args.build_dir)
    build.mkdir(parents=True, exist_ok=True)

    cm_exe = build_coremark(py, Path(args.src_coremark), build)
    cm_runs = []
    for i in range(args.repeats):
        r = parse_coremark(_run_pinned(cm_exe))
        if not r.get("validated"):
            raise RuntimeError(f"CoreMark 검증 실패 (run {i})")
        cm_runs.append(r)
        print(f"coremark run {i}: {r['iterations_per_sec']:.1f} iter/s", flush=True)
    cm_best = max(r["iterations_per_sec"] for r in cm_runs)

    dh: dict = {"note": "미실행"}
    if args.src_dhrystone:
        try:
            dh_exe = build_dhrystone(py, Path(args.src_dhrystone), build)
            best = {}
            for i in range(args.repeats):
                r = parse_dhrystone(_run_pinned(dh_exe))
                if r.get("dmips", 0) > best.get("dmips", 0):
                    best = r
                print(f"dhrystone run {i}: {r.get('dmips', float('nan')):.1f} DMIPS",
                      flush=True)
            dh = best if best else {"note": "출력 파싱 실패"}
        except RuntimeError as e:
            dh = {"note": f"빌드/실행 실패 — 생략: {str(e)[:300]}"}

    ratios = {}
    for ref in REFERENCES:
        if ref["coremark"]:
            ratios[ref["name"]] = {"coremark_x": cm_best / ref["coremark"]}
        if ref["dmips"] and dh.get("dmips"):
            ratios.setdefault(ref["name"], {})["dmips_x"] = dh["dmips"] / ref["dmips"]

    out = {
        "meta": result_meta(args.config),
        "cpu": platform.processor(),
        "conditions": "1 thread (affinity 1 core), zig cc(clang) -O2, "
                      f"repeats={args.repeats} 중 최고",
        "coremark": {"iterations_per_sec": cm_best, "runs": cm_runs,
                     "source_repo": "github.com/eembc/coremark"},
        "dhrystone": {**dh, "source_repo": "github.com/sifive/benchmark-dhrystone"},
        "references": REFERENCES,
        "speedup_vs_reference": ratios,
        "claim_note": "τ 벤치(1스레드 ORT CPU)를 돌린 CPU의 연산 수준 근거. "
                      "레거시 rad-hard 대비 수백 배 빠르므로 τ 실측치는 낙관적 하한 — "
                      "차세대(HPSC급) 온보드의 프록시로 해석한다.",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CoreMark {cm_best:.1f} iter/s, DMIPS {dh.get('dmips', 'N/A')}")
    for name, r in ratios.items():
        print(f"  vs {name}: " + ", ".join(f"{k} {v:,.0f}x" for k, v in r.items()))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
