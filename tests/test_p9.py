"""P9 매트릭스 스윕의 순수 로직 테스트 (MC 실행 없이 파생 모델·스케일만)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweep_matrix import derive_meas_file, load_compute_classes  # noqa: E402


def test_derive_meas_file_top_level(tmp_path: Path) -> None:
    src = tmp_path / "m.json"
    src.write_text(json.dumps({
        "sigma_xyz_m": [80.0, 81.0, 30.0], "fp_rate_est": 0.1,
        "fp_offset_med_m": 380.0,
        "by_precision": {"fp32": {"sigma_xyz_m": [84.0, 85.0, 31.0],
                                  "fp_rate_est": 0.11, "fp_offset_med_m": 370.0}},
    }), encoding="utf-8")
    out = tmp_path / "d.json"
    m = derive_meas_file(src, None, out)
    assert m["sigma_xyz_m"] == [80.0, 81.0, 30.0] and m["fp_rate_est"] == 0.1
    m32 = derive_meas_file(src, "fp32", tmp_path / "d32.json")
    assert m32["sigma_xyz_m"] == [84.0, 85.0, 31.0] and m32["fp_offset_med_m"] == 370.0
    assert json.loads(out.read_text(encoding="utf-8"))["sigma_xyz_m"] == [80.0, 81.0, 30.0]


def test_load_compute_classes(tmp_path: Path) -> None:
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"speedup_vs_reference": {
        "Gaisler GR740 (quad LEON4FT, 250 MHz)": {"dmips_x": 95.7},
        "JAXA HR5000 계열 (MIPS64 5Kf, 200 MHz)": {"dmips_x": 127.0},
    }}), encoding="utf-8")
    classes = load_compute_classes(bench)
    by = {c["key"]: c for c in classes}
    assert by["next_gen"]["tau_scale"] == 1.0
    assert by["current_gen"]["tau_scale"] == pytest.approx(95.7)
    assert by["legacy"]["tau_scale"] == pytest.approx(127.0)
