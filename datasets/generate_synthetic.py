"""Generate deterministic synthetic tabular datasets for Nexus Sentinel.

All records are purely synthetic numerical observations for defensive
classification demonstrations. No live system telemetry or malware is used.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ml_engine.schema import FEATURE_NAMES

DATASETS_DIR = Path(__file__).resolve().parent
CSV_COLUMNS = ("scenario", *FEATURE_NAMES, "label")


def _sample_scenario(
    rng: np.random.Generator,
    scenario: str,
    count: int,
) -> list[dict[str, float | int | str]]:
    """Sample synthetic tabular observations for one scenario."""
    rows: list[dict[str, float | int | str]] = []
    for _ in range(count):
        if scenario == "normal":
            file_mod_rate = float(np.clip(rng.normal(8.0, 5.0), 0.0, 60.0))
            entropy_delta = float(np.clip(rng.normal(0.25, 0.18), 0.0, 2.0))
            rename_ratio = float(np.clip(rng.beta(1.5, 12.0), 0.0, 0.5))
            unique_ext_count = int(np.clip(rng.poisson(3.0), 0, 20))
            burst_io_score = float(np.clip(rng.beta(2.0, 7.0), 0.0, 0.8))
            shadow_copy_flag = int(rng.random() < 0.01)
            label = 0
        elif scenario == "backup":
            # Legitimate high-I/O backup: high modification rate & burstiness,
            # but low entropy delta, low rename ratio, and few extensions.
            file_mod_rate = float(np.clip(rng.normal(110.0, 35.0), 25.0, 260.0))
            entropy_delta = float(np.clip(rng.normal(0.45, 0.25), 0.0, 2.2))
            rename_ratio = float(np.clip(rng.beta(1.8, 10.0), 0.0, 0.55))
            unique_ext_count = int(np.clip(rng.poisson(4.5), 1, 22))
            burst_io_score = float(np.clip(rng.beta(5.0, 2.5), 0.2, 0.98))
            shadow_copy_flag = int(rng.random() < 0.04)
            label = 0
        elif scenario == "suspicious":
            # Synthetic ransomware-like pattern with realistic boundary overlap.
            file_mod_rate = float(np.clip(rng.normal(145.0, 55.0), 20.0, 400.0))
            entropy_delta = float(np.clip(rng.normal(2.6, 0.85), 0.4, 6.0))
            rename_ratio = float(np.clip(rng.beta(5.5, 2.5), 0.1, 1.0))
            unique_ext_count = int(np.clip(rng.poisson(14.0), 2, 60))
            burst_io_score = float(np.clip(rng.beta(5.5, 2.0), 0.25, 1.0))
            shadow_copy_flag = int(rng.random() < 0.65)
            label = 1
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        rows.append(
            {
                "scenario": scenario,
                "file_mod_rate": round(file_mod_rate, 4),
                "entropy_delta": round(entropy_delta, 4),
                "rename_ratio": round(rename_ratio, 4),
                "unique_ext_count": unique_ext_count,
                "burst_io_score": round(burst_io_score, 4),
                "shadow_copy_access_flag": shadow_copy_flag,
                "label": label,
            }
        )
    return rows


def generate_dataset(
    output_path: Path,
    counts: dict[str, int],
    seed: int,
) -> Path:
    """Generate a shuffled synthetic CSV dataset at output_path."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for scenario, count in counts.items():
        rows.extend(_sample_scenario(rng, scenario, count))

    perm = rng.permutation(len(rows))
    shuffled = [rows[int(idx)] for idx in perm]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(shuffled)

    return output_path


def generate_default_datasets() -> tuple[Path, Path]:
    """Generate independent training and held-out test CSVs."""
    train_path = generate_dataset(
        DATASETS_DIR / "train_synthetic.csv",
        counts={"normal": 350, "backup": 250, "suspicious": 300},
        seed=42,
    )
    test_path = generate_dataset(
        DATASETS_DIR / "test_synthetic.csv",
        counts={"normal": 175, "backup": 125, "suspicious": 150},
        seed=2026,
    )
    return train_path, test_path


if __name__ == "__main__":
    train_p, test_p = generate_default_datasets()
    print(f"Generated train dataset: {train_p}")
    print(f"Generated test dataset:  {test_p}")

