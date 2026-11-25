# Nexus Sentinel — Phase 2 Model Evaluation Report

All metrics below are measured on the independent Phase 2 synthetic validation dataset (`datasets/val_phase2.csv`, `600` records across `normal`, `backup`, `compression`, and `suspicious` scenarios). Exact row overlap with `datasets/train_phase2.csv`: `0`.

## 1. Phase 2 Model vs Phase 1 Fallback (on `val_phase2.csv`)

| Metric | Phase 2 (`rf_phase2.joblib`) | Phase 1 Fallback (`rf_baseline_v1.joblib`) |
| :--- | :---: | :---: |
| Precision | 0.9801 | 0.6899 |
| Recall | 0.9850 | 0.9900 |
| F1 Score | 0.9825 | 0.8131 |
| False-Positive Rate | 0.0100 | 0.2225 |
| Batch Inference Latency (ms/sample) | 0.070793 | 0.020935 |

Single-observation `predict_assessment()` latency (warm artifact cache): `40.5175 ms`.

## 2. Confusion Matrix (`rf_phase2.joblib` on `val_phase2.csv`)

| Actual \ Predicted | Normal (`0`) | Suspicious (`1`) |
| :--- | :---: | :---: |
| **Normal/Benign (`0`)** | TN = 396 | FP = 4 |
| **Suspicious (`1`)** | FN = 3 | TP = 197 |

## 3. Per-Scenario Breakdown (`rf_phase2.joblib`)

| Scenario | Total | Errors | Rate |
| :--- | :---: | :---: | :--- |
| `normal` | 200 | 0 FP | FPR = 0.0000 |
| `backup` | 125 | 3 FP | FPR = 0.0240 |
| `compression` | 75 | 1 FP | FPR = 0.0133 |
| `suspicious` | 200 | 3 FN | Detection = 0.9850 |

## 4. Inference Latency vs End-to-End Detection Latency
- **Model inference latency** (`40.5175 ms` per call) measures only the in-memory `predict_assessment()` computation on one tabular observation.
- **End-to-end detection latency** in a full system includes telemetry aggregation/windowing, backend HTTP request handling, SQLite persistence, and frontend polling intervals (seconds, not sub-millisecond).

## 5. Limitations
- Trained and evaluated exclusively on synthetic tabular records; not validated on real endpoint telemetry or real malware.
- `risk_score` (`0–100`) is a model-derived score, not a calibrated real-world probability.
- Does not provide real-world ransomware protection, process termination, or live filesystem monitoring.
