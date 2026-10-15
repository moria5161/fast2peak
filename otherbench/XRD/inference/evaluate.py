#!/usr/bin/env python3
"""Evaluate classical denoisers and zero-shot F2P RPE on the PXRD benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pywt
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, savgol_filter


from otherbench.XRD.models.baselines import wavelet_denoise, anscombe_wavelet_denoise
from otherbench.XRD.models.f2p import load_f2p
from otherbench.XRD.inference.f2p import infer_f2p
from otherbench.XRD.utils.metrics import metrics_for_sample, aggregate
from otherbench.XRD.utils.reporting import save_macro_average
from otherbench.XRD.utils.config import XRD_ROOT, DEFAULT_CONFIG, load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=XRD_ROOT / "datasets" / "benchmark" / "benchmark.h5")
    parser.add_argument("--output-dir", type=Path, default=XRD_ROOT / "outputs" / "evaluation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug-only sample limit")
    parser.add_argument("--skip-f2p", action="store_true", help="Debug classical baselines without loading F2P")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model, checkpoint_config, checkpoint = (None, None, None)
    if not args.skip_f2p:
        model, checkpoint_config, checkpoint = load_f2p(config, device)

    all_rows: list[dict] = []
    inference_times = {}
    with h5py.File(args.data.resolve(), "r") as source, h5py.File(output_dir / "denoised.h5", "w") as destination:
        sample_count = len(source["gt"]) if args.max_samples is None else min(args.max_samples, len(source["gt"]))
        target = source["gt"][:sample_count].astype(np.float32)
        axis = source["two_theta"][:].astype(np.float32)
        destination.create_dataset("gt", data=target, compression="gzip", shuffle=True)
        destination.create_dataset("two_theta", data=axis)
        output_groups = destination.create_group("conditions")
        for condition in config["noise_conditions"]:
            condition_name = condition["name"]
            noisy = source["noise"][condition_name][:sample_count].astype(np.float32)
            methods = {
                "input": noisy,
                "savgol": np.clip(
                    savgol_filter(
                        noisy,
                        window_length=int(config["baselines"]["savgol"]["window_length"]),
                        polyorder=int(config["baselines"]["savgol"]["polyorder"]),
                        axis=1,
                    ),
                    0.0,
                    None,
                ).astype(np.float32),
                "wavelet_coif4": wavelet_denoise(
                    noisy,
                    config["baselines"]["wavelet"]["family"],
                    int(config["baselines"]["wavelet"]["level"]),
                    config["baselines"]["wavelet"]["threshold_mode"],
                ),
            }
            # Match the final benchmark: Anscombe is used for pure Poisson only.
            if condition["kind"] == "poisson":
                methods["anscombe_coif4"] = anscombe_wavelet_denoise(
                    noisy, float(condition["peak_counts"]), config["baselines"]["wavelet"]
                )
            if model is not None:
                methods["f2p_rpe"], inference_times[condition_name] = infer_f2p(
                    model, noisy, int(config["f2p"]["batch_size"]), device
                )

            condition_group = output_groups.create_group(condition_name)
            condition_group.create_dataset("noisy", data=noisy, compression="gzip", shuffle=True)
            for method, prediction in methods.items():
                if method != "input":
                    condition_group.create_dataset(method, data=prediction, compression="gzip", shuffle=True)
                for sample_index in range(sample_count):
                    all_rows.append({
                        "condition": condition_name,
                        "method": method,
                        "sample_index": sample_index,
                        **metrics_for_sample(
                            target[sample_index], prediction[sample_index], noisy[sample_index], axis,
                            config["peak_metrics"],
                        ),
                    })

    metric_fields = list(all_rows[0].keys())
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = []
    for condition in config["noise_conditions"]:
        condition_name = condition["name"]
        methods = list(dict.fromkeys(row["method"] for row in all_rows if row["condition"] == condition_name))
        for method in methods:
            selected = [
                row for row in all_rows if row["condition"] == condition_name and row["method"] == method
            ]
            summary_rows.append({"condition": condition_name, "method": method, **aggregate(selected)})
    summary_fields = list(summary_rows[0].keys())
    with (output_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "samples": int(sample_count),
        "device": str(device),
        "f2p_checkpoint": config["f2p"]["checkpoint"] if model is not None else None,
        "f2p_checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "f2p_inference_seconds_by_condition": inference_times,
        "leaderboard": summary_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_macro_average(summary_rows, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
