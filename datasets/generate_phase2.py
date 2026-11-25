"""Generate Phase 2 synthetic tabular datasets for Nexus Sentinel.

Includes realistic boundary overlap and a legitimate 'compression' scenario
(high entropy delta + high I/O) alongside 'normal', 'backup', and 'suspicious'
observations. Purely synthetic numerical records; no live system access.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ml_engine.schema import FEATURE_NAMES

DATASETS_DIR = Path(__file__).resolve().parent
CSV_COLUMNS = ("scenario", *FEATURE_NAMES, "label")


def _sample_phase2_scenario(
    rng: np.random.Generator,
    scenario: str,
    count: int,
) -> list[dict[str, float | int | str]]:
    """Sample synthetic tabular observations with realistic class overlap."""
    rows: list[dict[str, float | int | str]] = []
    for _ in range(count):
        if scenario == "normal":
            file_mod_rate = float(np.clip(rng.normal(12.0, 8.0), 0.0, 90.0))
            entropy_delta = float(np.clip(rng.normal(0.35, 0.28), 0.0, 2.5))
            rename_ratio = float(np.clip(rng.beta(1.6, 10.0), 0.0, 0.65))
            unique_ext_count = int(np.clip(rng.poisson(3.5), 0, 25))
            burst_io_score = float(np.clip(rng.beta(2.2, 6.0), 0.0, 0.85))
            shadow_copy_flag = int(rng.random() < 0.02)
            label = 0
        elif scenario == "backup":
            # Legitimate high-I/O backup with occasional moderate entropy/renames.
            file_mod_rate = float(np.clip(rng.normal(120.0, 45.0), 20.0, 320.0))
            entropy_delta = float(np.clip(rng.normal(0.75, 0.45), 0.0, 3.0))
            rename_ratio = float(np.clip(rng.beta(2.2, 7.5), 0.0, 0.70))
            unique_ext_count = int(np.clip(rng.poisson(6.0), 1, 28))
            burst_io_score = float(np.clip(rng.beta(5.0, 2.5), 0.15, 0.99))
            shadow_copy_flag = int(rng.random() < 0.08)
            label = 0
        elif scenario == "compression":
            # Legitimate archiving/compression: high entropy delta & high I/O,
            # but lower rename ratio and fewer unique extensions than ransomware.
            file_mod_rate = float(np.clip(rng.normal(95.0, 40.0), 15.0, 280.0))
            entropy_delta = float(np.clip(rng.normal(2.2, 0.75), 0.5, 4.8))
            rename_ratio = float(np.clip(rng.beta(2.0, 8.0), 0.0, 0.60))
            unique_ext_count = int(np.clip(rng.poisson(4.0), 1, 18))
            burst_io_score = float(np.clip(rng.beta(4.8, 2.6), 0.2, 0.98))
            shadow_copy_flag = int(rng.random() < 0.03)
            label = 0
        elif scenario == "suspicious":
            # Mix of aggressive and stealthy/low-rate suspicious synthetic patterns.
            stealth = rng.random() < 0.25
            if stealth:
                file_mod_rate = float(np.clip(rng.normal(45.0, 22.0), 8.0, 140.0))
                entropy_delta = float(np.clip(rng.normal(1.65, 0.60), 0.4, 3.8))
                rename_ratio = float(np.clip(rng.beta(3.8, 3.5), 0.1, 0.95))
                unique_ext_count = int(np.clip(rng.poisson(9.0), 2, 35))
                burst_io_score = float(np.clip(rng.beta(3.2, 3.5), 0.1, 0.92))
                shadow_copy_flag = int(rng.random() < 0.25)
            else:
                file_mod_rate = float(np.clip(rng.normal(150.0, 60.0), 20.0, 420.0))
                entropy_delta = float(np.clip(rng.normal(2.55, 0.85), 0.5, 6.0))
                rename_ratio = float(np.clip(rng.beta(5.0, 2.5), 0.15, 1.0))
                unique_ext_count = int(np.clip(rng.poisson(14.5), 3, 60))
                burst_io_score = float(np.clip(rng.beta(5.2, 2.2), 0.2, 1.0))
                shadow_copy_flag = int(rng.random() < 0.60)
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


def generate_phase2_dataset(
    output_path: Path,
    counts: dict[str, int],
    seed: int,
) -> Path:
    """Generate a shuffled Phase 2 synthetic CSV dataset."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for scenario, count in counts.items():
        rows.extend(_sample_phase2_scenario(rng, scenario, count))

    perm = rng.permutation(len(rows))
    shuffled = [rows[int(idx)] for idx in perm]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(shuffled)

    return output_path


def generate_default_phase2_datasets() -> tuple[Path, Path]:
    """Generate independent Phase 2 training and validation CSVs."""
    train_path = generate_phase2_dataset(
        DATASETS_DIR / "train_phase2.csv",
        counts={"normal": 400, "backup": 250, "compression": 150, "suspicious": 400},
        seed=2027,
    )
    val_path = generate_phase2_dataset(
        DATASETS_DIR / "val_phase2.csv",
        counts={"normal": 200, "backup": 125, "compression": 75, "suspicious": 200},
        seed=9001,
    )
    return train_path, val_path


if __name__ == "__main__":
    train_p, val_p = generate_default_phase2_datasets()
    print(f"Generated Phase 2 train dataset: {train_p}")
    print(f"Generated Phase 2 val dataset:   {val_p}")

