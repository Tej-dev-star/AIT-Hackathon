"""Inference interface for Nexus Sentinel (synthetic observations only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml_engine.schema import FEATURE_NAMES, SCHEMA_VERSION, validate_observation

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
FALLBACK_MODEL_PATH = ARTIFACTS_DIR / "rf_baseline_v1.joblib"
PHASE2_MODEL_PATH = ARTIFACTS_DIR / "rf_phase2.joblib"
DEFAULT_MODEL_PATH = PHASE2_MODEL_PATH

_DEFAULT_BENIGN_MEDIANS: tuple[float, ...] = (15.0, 0.45, 0.12, 4.0, 0.30, 0.0)
_ARTIFACT_CACHE: dict[Path, dict[str, Any]] = {}


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve explicit model path, Phase 2 artifact, or Phase 1 fallback."""
    if model_path is not None:
        return Path(model_path).resolve()
    if PHASE2_MODEL_PATH.is_file():
        return PHASE2_MODEL_PATH.resolve()
    return FALLBACK_MODEL_PATH.resolve()


def load_trusted_artifact(model_path: str | Path | None = None) -> dict[str, Any]:
    """Load a locally produced trusted joblib artifact."""
    resolved = resolve_model_path(model_path)
    if resolved in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[resolved]

    if not resolved.is_file():
        raise FileNotFoundError(f"Model artifact not found: {resolved}")

    artifact = joblib.load(resolved)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Invalid model artifact format")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Schema version mismatch: expected {SCHEMA_VERSION}, "
            f"got {artifact.get('schema_version')}"
        )
    if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Feature names mismatch in model artifact")

    _ARTIFACT_CACHE[resolved] = artifact
    return artifact


def _compute_model_explanations(
    model: Any,
    vector: list[float],
    benign_medians: tuple[float, ...],
    base_prob: float,
) -> list[str]:
    """Compute local model-derived feature contributions via batched perturbation."""
    num_features = len(FEATURE_NAMES)
    batch = np.tile(np.asarray(vector, dtype=np.float64), (num_features, 1))
    for idx in range(num_features):
        batch[idx, idx] = benign_medians[idx]

    probs_batch = model.predict_proba(batch)
    p_pert_all = probs_batch[:, 1] if probs_batch.shape[1] > 1 else probs_batch[:, 0]

    contributions: list[tuple[float, str, float]] = []
    for idx, name in enumerate(FEATURE_NAMES):
        delta = base_prob - float(p_pert_all[idx])
        if abs(delta) >= 0.03:
            contributions.append((abs(delta), name, delta))

    contributions.sort(reverse=True)
    if not contributions:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(importances) == len(FEATURE_NAMES):
            top_idx = int(np.argmax(importances))
            return [
                f"model: {FEATURE_NAMES[top_idx]} has highest global weight "
                f"({importances[top_idx]:.2f}); observation aligns with baseline"
            ]
        return ["model: observation aligns with benign baseline"]

    explanations: list[str] = []
    for _, name, delta in contributions[:3]:
        direction = "increases" if delta > 0 else "decreases"
        explanations.append(
            f"model: {name} {direction} suspicious score by {abs(delta) * 100:.1f} pts vs benign median"
        )
    return explanations


def _derive_heuristic_indicators(observation: dict[str, Any]) -> list[str]:
    """Generate rule-based heuristic threshold explanations."""
    heuristics: list[str] = []
    if float(observation["entropy_delta"]) >= 1.5:
        heuristics.append("heuristic: elevated entropy delta (>= 1.5)")
    if float(observation["rename_ratio"]) >= 0.45:
        heuristics.append("heuristic: elevated file rename ratio (>= 0.45)")
    if int(observation["unique_ext_count"]) >= 10:
        heuristics.append("heuristic: broad extension diversity (>= 10)")
    if float(observation["file_mod_rate"]) >= 80.0:
        heuristics.append("heuristic: high file modification rate (>= 80/min)")
    if float(observation["burst_io_score"]) >= 0.7:
        heuristics.append("heuristic: sustained I/O burst (>= 0.70)")
    if int(observation["shadow_copy_access_flag"]) == 1:
        heuristics.append("heuristic: shadow copy access pattern (flag == 1)")

    if not heuristics:
        heuristics.append("heuristic: all thresholds within normal baseline")
    return heuristics


def predict_assessment_detailed(
    observation: dict[str, Any],
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Predict risk assessment with separated model-derived and heuristic explanations."""
    is_valid, reasons, vector = validate_observation(observation)
    if not is_valid or vector is None:
        return {
            "risk_score": None,
            "label": "unknown",
            "indicators": [f"validation: {r}" for r in reasons],
            "simulated": True,
            "explanations": {
                "model_derived": [],
                "heuristic": [f"validation: {r}" for r in reasons],
            },
            "model_version": None,
        }

    artifact = load_trusted_artifact(model_path)
    model = artifact["model"]
    medians = tuple(artifact.get("benign_medians", _DEFAULT_BENIGN_MEDIANS))

    x = np.asarray([vector], dtype=np.float64)
    probs = model.predict_proba(x)[0]
    suspicious_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
    risk_score = max(0, min(100, int(round(suspicious_prob * 100.0))))
    label = "suspicious" if risk_score >= 50 else "normal"

    model_explanations = _compute_model_explanations(model, vector, medians, suspicious_prob)
    heuristic_explanations = _derive_heuristic_indicators(observation)

    return {
        "risk_score": risk_score,
        "label": label,
        "indicators": model_explanations + heuristic_explanations,
        "simulated": True,
        "explanations": {
            "model_derived": model_explanations,
            "heuristic": heuristic_explanations,
        },
        "model_version": artifact.get("model_version", "unknown"),
    }


def predict_assessment(
    observation: dict[str, Any],
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Predict risk assessment for a single synthetic tabular observation.

    Returns exact API_CONTRACT.md assessment fields:
    - risk_score: int in [0, 100] or None when label == 'unknown'
    - label: 'normal' | 'suspicious' | 'unknown'
    - indicators: list[str] (model-derived and heuristic explanations)
    - simulated: True
    """
    detailed = predict_assessment_detailed(observation, model_path=model_path)
    return {
        "risk_score": detailed["risk_score"],
        "label": detailed["label"],
        "indicators": detailed["indicators"],
        "simulated": detailed["simulated"],
    }
