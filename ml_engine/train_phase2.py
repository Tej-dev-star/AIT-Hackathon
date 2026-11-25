"""Train and evaluate Phase 2 Random Forest on independent synthetic scenarios."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from ml_engine.predict import FALLBACK_MODEL_PATH, predict_assessment
from ml_engine.schema import FEATURE_NAMES, SCHEMA_VERSION
from ml_engine.train import load_csv_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
PHASE2_TRAIN_CSV = ROOT_DIR / "datasets" / "train_phase2.csv"
PHASE2_VAL_CSV = ROOT_DIR / "datasets" / "val_phase2.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
PHASE2_MODEL_PATH = ARTIFACTS_DIR / "rf_phase2.joblib"
PHASE2_METRICS_PATH = ARTIFACTS_DIR / "metrics_phase2.json"
EVALUATION_REPORT_PATH = Path(__file__).resolve().parent / "EVALUATION_REPORT.md"


def count_exact_row_overlap(x_train: np.ndarray, x_val: np.ndarray) -> int:
    """Count exact feature-vector matches between train and validation sets."""
    train_set = {tuple(row.tolist()) for row in x_train}
    return sum(1 for row in x_val if tuple(row.tolist()) in train_set)


def evaluate_model_on_dataset(
    model: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
    scenarios: list[str],
) -> dict[str, Any]:
    """Evaluate a classifier on validation records including compression & backup."""
    start_ns = time.perf_counter_ns()
    probs = model.predict_proba(x_val)[:, 1]
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    avg_batch_latency_ms = round(elapsed_ms / max(len(x_val), 1), 6)

    y_pred = (np.rint(probs * 100.0) >= 50).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    scenarios_arr = np.asarray(scenarios)
    scenario_breakdown: dict[str, dict[str, int | float]] = {}
    for name in ("normal", "backup", "compression"):
        mask = scenarios_arr == name
        total = int(np.sum(mask))
        false_pos = int(np.sum(y_pred[mask] == 1))
        scenario_breakdown[name] = {
            "total": total,
            "false_positives": false_pos,
            "false_positive_rate": round(false_pos / total, 6) if total > 0 else 0.0,
        }

    susp_mask = scenarios_arr == "suspicious"
    susp_total = int(np.sum(susp_mask))
    susp_tp = int(np.sum(y_pred[susp_mask] == 1))
    scenario_breakdown["suspicious"] = {
        "total": susp_total,
        "true_positives": susp_tp,
        "detection_rate": round(susp_tp / susp_total, 6) if susp_total > 0 else 0.0,
    }

    return {
        "validation_samples": int(len(y_val)),
        "precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_val, y_pred, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_val, y_pred, zero_division=0)), 6),
        "false_positive_rate": round(fpr, 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "scenario_breakdown": scenario_breakdown,
        "avg_batch_inference_latency_ms": avg_batch_latency_ms,
    }


def _measure_single_call_latency_ms(x_val: np.ndarray, model_path: Path) -> float:
    """Measure average single-observation predict_assessment() latency."""
    sample_obs = {
        name: float(x_val[0, idx]) if name != "shadow_copy_access_flag" else int(x_val[0, idx])
        for idx, name in enumerate(FEATURE_NAMES)
    }
    predict_assessment(sample_obs, model_path=model_path)  # warm cache
    runs = 50
    start_ns = time.perf_counter_ns()
    for _ in range(runs):
        predict_assessment(sample_obs, model_path=model_path)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    return round(elapsed_ms / runs, 4)


def _render_evaluation_report(report_data: dict[str, Any]) -> str:
    """Render EVALUATION_REPORT.md from actual measured metrics."""
    p2 = report_data["phase2_model"]
    p1 = report_data["phase1_fallback_on_phase2_val"]
    cm2 = p2["confusion_matrix"]
    sb2 = p2["scenario_breakdown"]

    return f"""# Nexus Sentinel — Phase 2 Model Evaluation Report

All metrics below are measured on the independent Phase 2 synthetic validation dataset (`datasets/val_phase2.csv`, `{p2["validation_samples"]}` records across `normal`, `backup`, `compression`, and `suspicious` scenarios). Exact row overlap with `datasets/train_phase2.csv`: `{report_data["exact_row_overlap"]}`.

## 1. Phase 2 Model vs Phase 1 Fallback (on `val_phase2.csv`)

| Metric | Phase 2 (`rf_phase2.joblib`) | Phase 1 Fallback (`rf_baseline_v1.joblib`) |
| :--- | :---: | :---: |
| Precision | {p2["precision"]:.4f} | {p1["precision"]:.4f} |
| Recall | {p2["recall"]:.4f} | {p1["recall"]:.4f} |
| F1 Score | {p2["f1_score"]:.4f} | {p1["f1_score"]:.4f} |
| False-Positive Rate | {p2["false_positive_rate"]:.4f} | {p1["false_positive_rate"]:.4f} |
| Batch Inference Latency (ms/sample) | {p2["avg_batch_inference_latency_ms"]:.6f} | {p1["avg_batch_inference_latency_ms"]:.6f} |

Single-observation `predict_assessment()` latency (warm artifact cache): `{report_data["single_call_latency_ms"]:.4f} ms`.

## 2. Confusion Matrix (`rf_phase2.joblib` on `val_phase2.csv`)

| Actual \\ Predicted | Normal (`0`) | Suspicious (`1`) |
| :--- | :---: | :---: |
| **Normal/Benign (`0`)** | TN = {cm2["tn"]} | FP = {cm2["fp"]} |
| **Suspicious (`1`)** | FN = {cm2["fn"]} | TP = {cm2["tp"]} |

## 3. Per-Scenario Breakdown (`rf_phase2.joblib`)

| Scenario | Total | Errors | Rate |
| :--- | :---: | :---: | :--- |
| `normal` | {sb2["normal"]["total"]} | {sb2["normal"]["false_positives"]} FP | FPR = {sb2["normal"]["false_positive_rate"]:.4f} |
| `backup` | {sb2["backup"]["total"]} | {sb2["backup"]["false_positives"]} FP | FPR = {sb2["backup"]["false_positive_rate"]:.4f} |
| `compression` | {sb2["compression"]["total"]} | {sb2["compression"]["false_positives"]} FP | FPR = {sb2["compression"]["false_positive_rate"]:.4f} |
| `suspicious` | {sb2["suspicious"]["total"]} | {sb2["suspicious"]["total"] - sb2["suspicious"]["true_positives"]} FN | Detection = {sb2["suspicious"]["detection_rate"]:.4f} |

## 4. Inference Latency vs End-to-End Detection Latency
- **Model inference latency** (`{report_data["single_call_latency_ms"]:.4f} ms` per call) measures only the in-memory `predict_assessment()` computation on one tabular observation.
- **End-to-end detection latency** in a full system includes telemetry aggregation/windowing, backend HTTP request handling, SQLite persistence, and frontend polling intervals (seconds, not sub-millisecond).

## 5. Limitations
- Trained and evaluated exclusively on synthetic tabular records; not validated on real endpoint telemetry or real malware.
- `risk_score` (`0–100`) is a model-derived score, not a calibrated real-world probability.
- Does not provide real-world ransomware protection, process termination, or live filesystem monitoring.
"""


def train_and_evaluate_phase2(
    train_csv: Path = PHASE2_TRAIN_CSV,
    val_csv: Path = PHASE2_VAL_CSV,
    model_path: Path = PHASE2_MODEL_PATH,
    metrics_path: Path = PHASE2_METRICS_PATH,
    report_path: Path = EVALUATION_REPORT_PATH,
) -> dict[str, Any]:
    """Train Phase 2 Random Forest and evaluate both Phase 2 and Phase 1 models."""
    x_train, y_train, _ = load_csv_dataset(train_csv)
    x_val, y_val, val_scenarios = load_csv_dataset(val_csv)

    overlap = count_exact_row_overlap(x_train, x_val)

    benign_mask = y_train == 0
    benign_medians = tuple(float(v) for v in np.median(x_train[benign_mask], axis=0))

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=1,
    )
    clf.fit(x_train, y_train)

    phase2_metrics = evaluate_model_on_dataset(clf, x_val, y_val, val_scenarios)

    fallback_artifact = joblib.load(FALLBACK_MODEL_PATH)
    phase1_metrics = evaluate_model_on_dataset(
        fallback_artifact["model"], x_val, y_val, val_scenarios
    )

    artifact = {
        "model": clf,
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "model_version": "rf-phase2-v1",
        "benign_medians": list(benign_medians),
        "metrics": phase2_metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    single_latency_ms = _measure_single_call_latency_ms(x_val, model_path)

    report_data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "train_samples": int(len(x_train)),
        "exact_row_overlap": overlap,
        "single_call_latency_ms": single_latency_ms,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "phase2_model": phase2_metrics,
        "phase1_fallback_on_phase2_val": phase1_metrics,
        "limitations": (
            "Synthetic tabular records only. risk_score is a model-derived 0-100 score, "
            "not a calibrated real-world probability. Does not provide real-world protection."
        ),
    }

    metrics_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    report_path.write_text(_render_evaluation_report(report_data), encoding="utf-8")
    return report_data


if __name__ == "__main__":
    output = train_and_evaluate_phase2()
    print(json.dumps(output, indent=2))

