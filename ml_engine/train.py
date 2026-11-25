"""Train and evaluate the Nexus Sentinel Phase 1 Random Forest baseline."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from ml_engine.schema import FEATURE_NAMES, SCHEMA_VERSION

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_CSV = ROOT_DIR / "datasets" / "train_synthetic.csv"
DEFAULT_TEST_CSV = ROOT_DIR / "datasets" / "test_synthetic.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "rf_baseline_v1.joblib"
DEFAULT_METRICS_PATH = ARTIFACTS_DIR / "metrics_v1.json"


def load_csv_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load feature matrix X, binary labels y, and scenario names from CSV."""
    features: list[list[float]] = []
    labels: list[int] = []
    scenarios: list[str] = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            features.append([float(row[name]) for name in FEATURE_NAMES])
            labels.append(int(row["label"]))
            scenarios.append(row["scenario"])

    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64), scenarios


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scenarios: list[str],
) -> dict[str, Any]:
    """Compute precision, recall, F1, false-positive rate, and scenario stats."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    scenario_fp: dict[str, dict[str, int | float]] = {}
    scenarios_arr = np.asarray(scenarios)
    for name in ("normal", "backup"):
        mask = scenarios_arr == name
        total = int(np.sum(mask))
        false_pos = int(np.sum((y_pred[mask] == 1)))
        scenario_fp[name] = {
            "total": total,
            "false_positives": false_pos,
            "false_positive_rate": round(false_pos / total, 6) if total > 0 else 0.0,
        }

    susp_mask = scenarios_arr == "suspicious"
    susp_total = int(np.sum(susp_mask))
    susp_tp = int(np.sum(y_pred[susp_mask] == 1))

    return {
        "test_samples": int(len(y_true)),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "false_positive_rate": round(fpr, 6),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "scenario_breakdown": {
            **scenario_fp,
            "suspicious": {
                "total": susp_total,
                "true_positives": susp_tp,
                "detection_rate": round(susp_tp / susp_total, 6) if susp_total > 0 else 0.0,
            },
        },
    }


def train_and_evaluate(
    train_csv: Path = DEFAULT_TRAIN_CSV,
    test_csv: Path = DEFAULT_TEST_CSV,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    """Train CPU RandomForestClassifier and evaluate on independent test data."""
    x_train, y_train, _ = load_csv_dataset(train_csv)
    x_test, y_test, test_scenarios = load_csv_dataset(test_csv)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    clf.fit(x_train, y_train)

    start_ns = time.perf_counter_ns()
    y_pred = clf.predict(x_test)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    avg_latency_ms = round(elapsed_ms / max(len(x_test), 1), 6)

    metrics = evaluate_predictions(y_test, y_pred, test_scenarios)
    metrics["avg_inference_latency_ms"] = avg_latency_ms
    metrics["train_samples"] = int(len(x_train))
    metrics["schema_version"] = SCHEMA_VERSION
    metrics["model_type"] = "RandomForestClassifier"
    metrics["sklearn_version"] = sklearn.__version__
    metrics["numpy_version"] = np.__version__
    metrics["joblib_version"] = joblib.__version__
    metrics["limitations"] = (
        "Trained and evaluated exclusively on synthetic tabular records. "
        "risk_score is a model-derived 0-100 score, not a calibrated real-world probability. "
        "Does not provide real-world ransomware protection."
    )

    artifact = {
        "model": clf,
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "model_version": "rf-baseline-v1",
        "metrics": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


if __name__ == "__main__":
    results = train_and_evaluate()
    print(json.dumps(results, indent=2))

