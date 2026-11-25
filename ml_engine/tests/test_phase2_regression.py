"""Phase 2 regression tests for Nexus Sentinel ML engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datasets.generate_phase2 import generate_phase2_dataset
from ml_engine.predict import (
    FALLBACK_MODEL_PATH,
    PHASE2_MODEL_PATH,
    predict_assessment,
    predict_assessment_detailed,
)
from ml_engine.schema import SCHEMA_VERSION
from ml_engine.train import load_csv_dataset
from ml_engine.train_phase2 import (
    EVALUATION_REPORT_PATH,
    PHASE2_METRICS_PATH,
    PHASE2_TRAIN_CSV,
    PHASE2_VAL_CSV,
    count_exact_row_overlap,
)


def test_phase1_fallback_artifact_preserved_and_working() -> None:
    assert FALLBACK_MODEL_PATH.is_file()
    obs = {
        "file_mod_rate": 10.0,
        "entropy_delta": 0.2,
        "rename_ratio": 0.05,
        "unique_ext_count": 3,
        "burst_io_score": 0.15,
        "shadow_copy_access_flag": 0,
    }
    res = predict_assessment(obs, model_path=FALLBACK_MODEL_PATH)
    assert set(res.keys()) == {"risk_score", "label", "indicators", "simulated"}
    assert res["label"] == "normal"
    assert res["simulated"] is True


def test_phase2_datasets_have_no_exact_row_leakage() -> None:
    assert PHASE2_TRAIN_CSV.is_file()
    assert PHASE2_VAL_CSV.is_file()

    x_train, _, train_scenarios = load_csv_dataset(PHASE2_TRAIN_CSV)
    x_val, _, val_scenarios = load_csv_dataset(PHASE2_VAL_CSV)

    assert len(x_train) == 1200
    assert len(x_val) == 600
    assert set(train_scenarios) == {"normal", "backup", "compression", "suspicious"}
    assert set(val_scenarios) == {"normal", "backup", "compression", "suspicious"}
    assert count_exact_row_overlap(x_train, x_val) == 0


def test_phase2_generator_deterministic(tmp_path: Path) -> None:
    p1 = generate_phase2_dataset(
        tmp_path / "p2_a.csv",
        {"normal": 5, "backup": 5, "compression": 5, "suspicious": 5},
        seed=123,
    )
    p2 = generate_phase2_dataset(
        tmp_path / "p2_b.csv",
        {"normal": 5, "backup": 5, "compression": 5, "suspicious": 5},
        seed=123,
    )
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


def test_explanations_distinguish_model_derived_and_heuristic() -> None:
    assert PHASE2_MODEL_PATH.is_file()
    compression_obs = {
        "file_mod_rate": 95.0,
        "entropy_delta": 2.3,
        "rename_ratio": 0.10,
        "unique_ext_count": 3,
        "burst_io_score": 0.72,
        "shadow_copy_access_flag": 0,
    }
    detailed = predict_assessment_detailed(compression_obs, model_path=PHASE2_MODEL_PATH)
    assert set(detailed["explanations"].keys()) == {"model_derived", "heuristic"}
    assert all(s.startswith("model: ") for s in detailed["explanations"]["model_derived"])
    assert all(s.startswith("heuristic: ") for s in detailed["explanations"]["heuristic"])
    assert len(detailed["explanations"]["model_derived"]) >= 1
    assert len(detailed["explanations"]["heuristic"]) >= 1

    # API_CONTRACT.md compatibility check
    contract_res = predict_assessment(compression_obs, model_path=PHASE2_MODEL_PATH)
    assert set(contract_res.keys()) == {"risk_score", "label", "indicators", "simulated"}
    assert contract_res["risk_score"] == detailed["risk_score"]
    assert contract_res["label"] == detailed["label"]


def test_risk_score_consistency_and_invalid_inputs() -> None:
    low_risk = {
        "file_mod_rate": 8.0,
        "entropy_delta": 0.2,
        "rename_ratio": 0.04,
        "unique_ext_count": 2,
        "burst_io_score": 0.15,
        "shadow_copy_access_flag": 0,
    }
    high_risk = {
        "file_mod_rate": 190.0,
        "entropy_delta": 3.2,
        "rename_ratio": 0.80,
        "unique_ext_count": 22,
        "burst_io_score": 0.88,
        "shadow_copy_access_flag": 1,
    }
    low_res = predict_assessment(low_risk, model_path=PHASE2_MODEL_PATH)
    high_res = predict_assessment(high_risk, model_path=PHASE2_MODEL_PATH)

    assert isinstance(low_res["risk_score"], int) and 0 <= low_res["risk_score"] < 50
    assert low_res["label"] == "normal"
    assert isinstance(high_res["risk_score"], int) and 50 <= high_res["risk_score"] <= 100
    assert high_res["label"] == "suspicious"
    assert high_res["risk_score"] > low_res["risk_score"]

    invalid_res = predict_assessment({"file_mod_rate": "bad"}, model_path=PHASE2_MODEL_PATH)
    assert invalid_res["risk_score"] is None
    assert invalid_res["label"] == "unknown"
    assert invalid_res["simulated"] is True


def test_phase2_metrics_and_report_generated() -> None:
    assert PHASE2_METRICS_PATH.is_file()
    assert EVALUATION_REPORT_PATH.is_file()

    data = json.loads(PHASE2_METRICS_PATH.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["exact_row_overlap"] == 0

    p2 = data["phase2_model"]
    assert p2["validation_samples"] == 600
    assert p2["recall"] >= 0.80
    assert p2["precision"] >= 0.80
    assert p2["false_positive_rate"] <= 0.15

