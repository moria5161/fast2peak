"""Numerical macro-averages for the four reported benchmark methods."""

import csv
from pathlib import Path
import numpy as np

METHODS = {"input": "RAW", "savgol": "SavGol", "wavelet_coif4": "Wavelet", "f2p_rpe": "F2P"}
METRICS = ("pearson_mean", "rmse_mean", "snr_db_mean", "peak_f1_mean")


def save_macro_average(leaderboard: list[dict], output_dir: Path) -> list[dict]:
    conditions = {row["condition"] for row in leaderboard}
    rows = []
    for method, label in METHODS.items():
        selected = [row for row in leaderboard if row["method"] == method]
        if not selected:
            continue
        if len(selected) != len(conditions) or {row["condition"] for row in selected} != conditions:
            raise ValueError(f"Expected one row per noise condition for {method}")
        rows.append({
            "method": label,
            **{metric: float(np.mean([float(row[metric]) for row in selected])) for metric in METRICS},
            "aggregation": "unweighted_macro_average",
            "number_of_conditions": len(conditions),
        })
    fields = ["method", *METRICS, "aggregation", "number_of_conditions"]
    with (output_dir / "leaderboard_macro_average.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# PXRD macro-average results", "",
        f"Unweighted macro averages over {len(conditions)} noise conditions. Each condition has equal weight.", "",
        "| Method | Pearson ↑ | RMSE ↓ | SNR (dB) ↑ | Peak F1 ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['method']} | {row['pearson_mean']:.4f} | {row['rmse_mean']:.4f} | {row['snr_db_mean']:.2f} | {row['peak_f1_mean']:.4f} |")
    lines += ["", "Conditions: " + ", ".join(sorted(conditions)) + ".", "",
              "GT is a high-quality measured PXRD reference, not physically noise-free ground truth."]
    (output_dir / "leaderboard_macro_average.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
