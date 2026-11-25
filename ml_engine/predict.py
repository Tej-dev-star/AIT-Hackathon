"""Inference interface for Nexus Sentinel (synthetic observations only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml_engine.schema import FEATURE_NAMES, SCHEMA_VERSION, validate_observation

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "rf_baseline_v1.joblib"

_ARTIFACT_CACHE: dict[Path, dict[str, Any]] = {}


def load_trusted_artifact(model_path: str | Path | None = None) -> dict[str, Any]:
    """Load a locally produced trusted joblib artifact."""
    resolved = Path(model_path).resolve() if model_path is not None else DEFAULT_MODEL_PATH.resolve()
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


def _derive_indicators(observation: dict[str, Any], label: str) -> list[str]:
    """Generate human-readable explanation indicators from observation values."""
    indicators: list[str] = []
    if float(observation["entropy_delta"]) >= 1.5:
        indicators.append("high entropy delta")
    if float(observation["rename_ratio"]) >= 0.45:
        indicators.append("elevated file rename ratio")
    if int(observation["unique_ext_count"]) >= 10:
        indicators.append("broad extension diversity")
    if float(observation["file_mod_rate"]) >= 80.0:
        indicators.append("high file modification rate")
    if float(observation["burst_io_score"]) >= 0.7:
        indicators.append("sustained I/O burst")
    if int(observation["shadow_copy_access_flag"]) == 1:
        indicators.append("shadow copy access pattern")

    if not indicators:
        indicators.append("within normal baseline" if label == "normal" else "combined feature elevation")
    return indicators


def predict_assessment(
    observation: dict[str, Any],
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Predict risk assessment for a single synthetic tabular observation.

    Returns a dict compatible with API_CONTRACT.md assessment fields:
    - risk_score: int in [0, 100] or None when label == 'unknown'
    - label: 'normal' | 'suspicious' | 'unknown'
    - indicators: list[str]
    - simulated: True

    Backend supplies assessment_id, endpoint_id, and timestamp.
    """
    is_valid, reasons, vector = validate_observation(observation)
    if not is_valid or vector is None:
        return {
            "risk_score": None,
            "label": "unknown",
            "indicators": reasons,
            "simulated": True,
        }

    artifact = load_trusted_artifact(model_path)
    model = artifact["model"]

    x = np.asarray([vector], dtype=np.float64)
    probs = model.predict_proba(x)[0]
    suspicious_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
    risk_score = max(0, min(100, int(round(suspicious_prob * 100.0))))
    label = "suspicious" if risk_score >= 50 else "normal"

    return {
        "risk_score": risk_score,
        "label": label,
        "indicators": _derive_indicators(observation, label),
        "simulated": True,
    }

