"""Per-spectrum metrics and leaderboard export."""

import csv
from pathlib import Path
import numpy as np


def pearson_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)


def metric_summary(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    pearson = pearson_rows(target, prediction)
    mse = np.mean((target - prediction) ** 2, axis=1)
    rmse = np.sqrt(mse)
    signal_power = np.sum(target**2, axis=1)
    error_power = np.sum((target - prediction) ** 2, axis=1)
    snr = 10.0 * np.log10(signal_power / np.maximum(error_power, 1e-12))
    return {
        "pearson_mean": float(pearson.mean()),
        "pearson_median": float(np.median(pearson)),
        "pearson_q25": float(np.quantile(pearson, 0.25)),
        "pearson_q75": float(np.quantile(pearson, 0.75)),
        "rmse_mean": float(rmse.mean()),
        "snr_db_mean": float(snr.mean()),
    }


def export_leaderboard(summary: dict, output_dir: Path) -> None:
    fields = ["method", "pearson_mean", "pearson_median", "rmse_mean", "snr_db_mean"]
    methods = {"input": "RAW", "savgol": "SavGol", "paper_nn": "Autoencoder", "f2p_corrected": "F2P"}
    with (output_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, name in methods.items():
            metrics = summary["metrics"][key]
            writer.writerow({"method": name, **{field: metrics[field] for field in fields[1:]}})
