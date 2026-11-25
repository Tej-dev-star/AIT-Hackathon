"""Reproducible tests for Nexus Sentinel ML engine (synthetic data only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datasets.generate_synthetic import generate_dataset
from ml_engine.predict import predict_assessment
from ml_engine.schema import FEATURE_NAMES, SCHEMA_VERSION, validate_observation
from ml_engine.train import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, load_csv_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def test_schema_validation_accepts_valid_observation() -> None:
    obs = {
        "file_mod_rate": 10.5,
        "entropy_delta": 0.2,
        "rename_ratio": 0.05,
        "unique_ext_count": 3,
        "burst_io_score": 0.15,
        "shadow_copy_access_flag": 0,
    }
    ok, reasons, vec = validate_observation(obs)
    assert ok is True
    assert reasons == []
    assert vec is not None
    assert len(vec) == len(FEATURE_NAMES)


@pytest.mark.parametrize(
    "bad_obs",
    [
        {},
        {"file_mod_rate": -1.0, "entropy_delta": 0.2, "rename_ratio": 0.1, "unique_ext_count": 2, "burst_io_score": 0.1, "shadow_copy_access_flag": 0},
        {"file_mod_rate": float("nan"), "entropy_delta": 0.2, "rename_ratio": 0.1, "unique_ext_count": 2, "burst_io_score": 0.1, "shadow_copy_access_flag": 0},
        {"file_mod_rate": 10.0, "entropy_delta": 0.2, "rename_ratio": 0.1, "unique_ext_count": 2, "burst_io_score": 0.1, "shadow_copy_access_flag": 2},
    ],
)
def test_predict_returns_unknown_for_invalid_input(bad_obs: dict) -> None:
    result = predict_assessment(bad_obs)
    assert result["label"] == "unknown"
    assert result["risk_score"] is None
    assert result["simulated"] is True
    assert len(result["indicators"]) > 0


def test_synthetic_generator_is_deterministic(tmp_path: Path) -> None:
    p1 = generate_dataset(tmp_path / "a.csv", {"normal": 10, "backup": 10, "suspicious": 10}, seed=7)
    p2 = generate_dataset(tmp_path / "b.csv", {"normal": 10, "backup": 10, "suspicious": 10}, seed=7)
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


def test_independent_datasets_exist_and_contain_all_scenarios() -> None:
    train_csv = ROOT_DIR / "datasets" / "train_synthetic.csv"
    test_csv = ROOT_DIR / "datasets" / "test_synthetic.csv"
    assert train_csv.is_file()
    assert test_csv.is_file()

    x_train, y_train, train_scenarios = load_csv_dataset(train_csv)
    x_test, y_test, test_scenarios = load_csv_dataset(test_csv)
    assert len(x_train) == 900
    assert len(x_test) == 450
    assert set(train_scenarios) == {"normal", "backup", "suspicious"}
    assert set(test_scenarios) == {"normal", "backup", "suspicious"}
    assert set(y_train.tolist()) == {0, 1}
    assert set(y_test.tolist()) == {0, 1}


def test_trained_artifact_and_predictions_match_contract() -> None:
    assert DEFAULT_MODEL_PATH.is_file()
    assert DEFAULT_METRICS_PATH.is_file()

    metrics = json.loads(DEFAULT_METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics["schema_version"] == SCHEMA_VERSION
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["false_positive_rate"] <= 1.0

    normal_obs = {
        "file_mod_rate": 6.0,
        "entropy_delta": 0.15,
        "rename_ratio": 0.03,
        "unique_ext_count": 2,
        "burst_io_score": 0.12,
        "shadow_copy_access_flag": 0,
    }
    backup_obs = {
        "file_mod_rate": 115.0,
        "entropy_delta": 0.35,
        "rename_ratio": 0.08,
        "unique_ext_count": 4,
        "burst_io_score": 0.72,
        "shadow_copy_access_flag": 0,
    }
    suspicious_obs = {
        "file_mod_rate": 180.0,
        "entropy_delta": 3.4,
        "rename_ratio": 0.78,
        "unique_ext_count": 20,
        "burst_io_score": 0.85,
        "shadow_copy_access_flag": 1,
    }

    for obs, expected_label in (
        (normal_obs, "normal"),
        (backup_obs, "normal"),
        (suspicious_obs, "suspicious"),
    ):
        res = predict_assessment(obs)
        assert set(res.keys()) == {"risk_score", "label", "indicators", "simulated"}
        assert res["simulated"] is True
        assert isinstance(res["risk_score"], int)
        assert 0 <= res["risk_score"] <= 100
        assert isinstance(res["indicators"], list) and len(res["indicators"]) >= 1
        assert res["label"] == expected_label

