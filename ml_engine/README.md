# Nexus Sentinel — ML Engine (`ml_engine/`)

Phase 1 baseline for defensive early-warning classification using **synthetic tabular observations only**.

> **Scope & Limitations:** Trained and evaluated exclusively on synthetic tabular records (`datasets/train_synthetic.csv`, `datasets/test_synthetic.csv`). `risk_score` is a model-derived score in `[0, 100]` and is **not** a calibrated real-world probability. Does **not** monitor live filesystems, terminate processes, modify networks, or claim real-world ransomware protection.

## Inference Interface (for Backend)

```python
from ml_engine import predict_assessment

observation = {
    "file_mod_rate": 115.0,
    "entropy_delta": 0.35,
    "rename_ratio": 0.08,
    "unique_ext_count": 4,
    "burst_io_score": 0.72,
    "shadow_copy_access_flag": 0,
}

result = predict_assessment(observation)
```

### Return Format (`API_CONTRACT.md` compatible)
`predict_assessment` returns the assessment payload fields; the backend supplies `assessment_id`, `endpoint_id`, and `timestamp`:

```json
{
  "risk_score": 12,
  "label": "normal",
  "indicators": ["high file modification rate", "sustained I/O burst"],
  "simulated": true
}
```

If `observation` is malformed, missing fields, out of bounds, or contains non-finite values, `predict_assessment` returns `label: "unknown"` with `risk_score: null` (never fabricates a score):

```json
{
  "risk_score": null,
  "label": "unknown",
  "indicators": ["missing feature: entropy_delta"],
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

## Model Artifact & Dependencies
- **Artifact:** `ml_engine/artifacts/rf_baseline_v1.joblib` (CPU `RandomForestClassifier`)
- **Metrics:** `ml_engine/artifacts/metrics_v1.json`
- **Verified Python environment:** Python `3.14.5`, `numpy 2.5.5`, `scikit-learn 1.9.1`, `scipy 1.18.1`, `joblib 1.6.0`, `pytest 9.2.0`

## Reproduce Training & Tests

```powershell
.\.venv\Scripts\python.exe -m datasets.generate_synthetic
.\.venv\Scripts\python.exe -m ml_engine.train
.\.venv\Scripts\python.exe -m pytest ml_engine/tests -v
```

