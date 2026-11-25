"""Versioned synthetic feature schema and validation for Nexus Sentinel."""

from __future__ import annotations

import math
from typing import Any

SCHEMA_VERSION = "1.0.0"

FEATURE_NAMES: tuple[str, ...] = (
    "file_mod_rate",
    "entropy_delta",
    "rename_ratio",
    "unique_ext_count",
    "burst_io_score",
    "shadow_copy_access_flag",
)

SCENARIOS: tuple[str, ...] = ("normal", "backup", "suspicious")

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "file_mod_rate": (0.0, 10000.0),
    "entropy_delta": (0.0, 8.0),
    "rename_ratio": (0.0, 1.0),
    "unique_ext_count": (0.0, 1000.0),
    "burst_io_score": (0.0, 1.0),
    "shadow_copy_access_flag": (0.0, 1.0),
}


def validate_observation(
    observation: Any,
) -> tuple[bool, list[str], list[float] | None]:
    """Validate a synthetic observation dict against SCHEMA_VERSION.

    Returns (is_valid, reasons, feature_vector).
    When invalid, feature_vector is None so callers can return label='unknown'.
    """
    if not isinstance(observation, dict):
        return False, ["observation must be a JSON object"], None

    reasons: list[str] = []
    vector: list[float] = []

    for name in FEATURE_NAMES:
        if name not in observation:
            reasons.append(f"missing feature: {name}")
            continue
        raw = observation[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            reasons.append(f"invalid type for {name}: expected number")
            continue
        val = float(raw)
        if not math.isfinite(val):
            reasons.append(f"non-finite value for {name}")
            continue
        low, high = FEATURE_BOUNDS[name]
        if val < low or val > high:
            reasons.append(f"out-of-range value for {name}: {val} not in [{low}, {high}]")
            continue
        if name == "shadow_copy_access_flag" and val not in (0.0, 1.0):
            reasons.append("shadow_copy_access_flag must be 0 or 1")
            continue
        vector.append(val)

    if reasons:
        return False, reasons, None

    return True, [], vector

