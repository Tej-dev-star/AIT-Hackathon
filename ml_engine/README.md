# Nexus Sentinel — ML Engine (`ml_engine/`)

Defensive early-warning classification using **synthetic tabular observations only**.

> **Scope & Limitations:** Trained and evaluated exclusively on synthetic tabular records (`datasets/`). `risk_score` is a model-derived score in `[0, 100]` and is **not** a calibrated real-world probability. Does **not** monitor live filesystems, terminate processes, modify networks, or claim real-world ransomware protection.

## Inference Interface (for Backend)

```python
from ml_engine import predict_assessment

observation = {
    "file_mod_rate": 95.0,
    "entropy_delta": 2.3,
    "rename_ratio": 0.10,
    "unique_ext_count": 3,
    "burst_io_score": 0.72,
    "shadow_copy_access_flag": 0,
}

result = predict_assessment(observation)
```

### Return Format (`API_CONTRACT.md` compatible)
`predict_assessment` returns the assessment payload fields; the backend supplies `assessment_id`, `endpoint_id`, and `timestamp`:

```json
{
  "risk_score": 18,
  "label": "normal",
  "indicators": [
    "model: rename_ratio decreases suspicious score by 12.4 pts vs benign median",
    "heuristic: elevated entropy delta (>= 1.5)",
    "heuristic: high file modification rate (>= 80/min)"
  ],
  "simulated": true
}
```

For separated explanation objects (`explanations.model_derived` vs `explanations.heuristic`), use `predict_assessment_detailed(observation)`.

If `observation` is malformed, missing fields, out of bounds, or contains non-finite values, `predict_assessment` returns `label: "unknown"` with `risk_score: null` (never fabricates a score):

```json
{
  "risk_score": null,
  "label": "unknown",
  "indicators": ["validation: missing feature: entropy_delta"],
  "simulated": true
}
```

## Synthetic Feature Schema (`SCHEMA_VERSION = "1.0.0"`)

| Feature | Type | Valid Range | Description |
| :--- | :--- | :--- | :--- |
| `file_mod_rate` | `float` | `[0.0, 10000.0]` | Synthetic file modification rate per minute |
| `entropy_delta` | `float` | `[0.0, 8.0]` | Synthetic average file entropy change |
| `rename_ratio` | `float` | `[0.0, 1.0]` | Synthetic ratio of rename-to-write operations |
| `unique_ext_count` | `int/float` | `[0.0, 1000.0]` | Synthetic distinct file extensions touched |
| `burst_io_score` | `float` | `[0.0, 1.0]` | Synthetic I/O burstiness indicator |
| `shadow_copy_access_flag` | `int` | `0` or `1` | Synthetic shadow-copy-like access flag |

## Model Artifacts & Fallback
- **Default (Phase 2):** `ml_engine/artifacts/rf_phase2.joblib` (trained on `datasets/train_phase2.csv` with `normal`, `backup`, `compression`, and `suspicious` scenarios).
- **Fallback (Phase 1):** `ml_engine/artifacts/rf_baseline_v1.joblib` (preserved; automatically used if Phase 2 artifact is absent, or selectable via `model_path`).
- **Evaluation Report:** `ml_engine/EVALUATION_REPORT.md` and `ml_engine/artifacts/metrics_phase2.json`.
- **Verified Python environment:** Python `3.14.5`, `numpy 2.5.5`, `scikit-learn 1.9.1`, `scipy 1.18.1`, `joblib 1.6.0`, `pytest 9.2.0` (no `pandas` dependency).

## Latency Distinction
- **Inference latency:** CPU prediction time inside `predict_assessment()` (including local perturbation explanations; see `ml_engine/EVALUATION_REPORT.md` for measured batch vs single-call latency).
- **End-to-end detection latency:** Includes telemetry windowing, backend HTTP handling, SQLite persistence, and frontend polling intervals (typically seconds, much larger than model inference alone).

## Reproduce Training & Tests

```powershell
.\.venv\Scripts\python.exe -m datasets.generate_phase2
.\.venv\Scripts\python.exe -m ml_engine.train_phase2
.\.venv\Scripts\python.exe -m pytest ml_engine/tests -v
```
