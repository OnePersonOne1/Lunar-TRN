"""TauSampler empirical 모드가 P2 벤치마크 출력 스키마(results/tau_*.json)를 읽는지 검증."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.measurement import TauSampler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _schema_payload(samples: list[float]) -> dict:
    arr = np.asarray(samples)
    stats = {
        "median_s": float(np.median(arr)) if arr.size else None,
        "p95_s": float(np.percentile(arr, 95)) if arr.size else None,
        "mean_s": float(arr.mean()) if arr.size else None,
        "std_s": float(arr.std()) if arr.size else None,
    }
    return {"samples_s": samples, **stats, "meta": {}}


def test_tau_sampler_empirical_reads_schema(cfg: dict, tmp_path: Path) -> None:
    samples = [0.11, 0.12, 0.13, 0.35]
    f = tmp_path / "tau_test.json"
    f.write_text(json.dumps(_schema_payload(samples)), encoding="utf-8")

    cfg2 = copy.deepcopy(cfg)
    cfg2["tau"]["mode"] = "empirical"
    cfg2["tau"]["empirical_file"] = str(f)
    ts = TauSampler(cfg2, np.random.default_rng(0))
    assert ts.max_tau == 0.35
    for _ in range(50):
        assert ts.sample() in samples


def test_tau_sampler_empirical_rejects_empty(cfg: dict, tmp_path: Path) -> None:
    f = tmp_path / "tau_empty.json"
    f.write_text(json.dumps(_schema_payload([])), encoding="utf-8")
    cfg2 = copy.deepcopy(cfg)
    cfg2["tau"]["mode"] = "empirical"
    cfg2["tau"]["empirical_file"] = str(f)
    with pytest.raises(ValueError):
        TauSampler(cfg2, np.random.default_rng(0))


def test_tau_sampler_reads_real_bench_output_if_present(cfg: dict) -> None:
    """P2 실행 후 생성되는 실제 파일(config tau.empirical_file)이 있으면 로드 검증."""
    path = ROOT / cfg["tau"]["empirical_file"]
    if not path.exists():
        pytest.skip("P2 벤치마크 출력이 아직 없음")
    cfg2 = copy.deepcopy(cfg)
    cfg2["tau"]["mode"] = "empirical"
    cfg2["tau"]["empirical_file"] = str(path)
    ts = TauSampler(cfg2, np.random.default_rng(0))
    s = ts.sample()
    assert 0.0 < s < 60.0
