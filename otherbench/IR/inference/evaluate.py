#!/usr/bin/env python3
"""Evaluate Fast2PeakRPE and baselines on the additive Gaussian IR test set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.signal import savgol_filter


from otherbench.IR.models.f2p import load_model
from otherbench.IR.utils.metrics import pearson_rows, metric_summary, export_leaderboard
from otherbench.IR.utils.config import IR_ROOT, REPO_ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "experiments" / "f2p" / "checkpoints" / "best_checkpoint.pth",
    )
    data_dir = IR_ROOT / "datasets" / "paired"
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument(
        "--paper-nn-dir", type=Path, default=IR_ROOT / "outputs" / "autoencoder"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=IR_ROOT / "outputs" / "evaluation"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--savgol-window", type=int, default=15)
    parser.add_argument("--savgol-order", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def infer(
    model: torch.nn.Module,
    noisy: np.ndarray,
    target_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    from utils.base import spectra_alignment, spectra_alignment_reverse, spectra_correction

    original_length = noisy.shape[1]
    processed, alignment_mode, pad_left = spectra_alignment(noisy, original_length, target_length)
    min_values = processed.min(axis=1, keepdims=True)
    max_values = processed.max(axis=1, keepdims=True)
    normalized = (processed - min_values) / (max_values - min_values + 1e-8)

    outputs: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start : start + batch_size]).unsqueeze(1)
            prediction = model(batch.to(device=device, dtype=torch.float32))
            outputs.append(prediction.squeeze(1).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - started

    denoised_normalized = np.concatenate(outputs, axis=0)
    denoised_processed = denoised_normalized * (max_values - min_values + 1e-8) + min_values
    denoised_raw = spectra_alignment_reverse(
        denoised_processed, alignment_mode, original_length, pad_left
    ).astype(np.float32)
    # This is the repository's standard inference-time affine amplitude correction.
    denoised_corrected = spectra_correction(noisy, denoised_raw).astype(np.float32)
    return denoised_raw, denoised_corrected, elapsed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint_path = args.checkpoint.resolve()
    data_dir = args.data_dir.resolve()
    paper_nn_dir = args.paper_nn_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    noisy = np.load(data_dir / "test_noisy.npy").astype(np.float32)
    target = np.load(data_dir / "test_target.npy").astype(np.float32)
    reference_id = np.load(data_dir / "test_reference_id.npy")
    if noisy.shape != target.shape or noisy.shape != (4000, 1024):
        raise ValueError(f"Expected paper test arrays of shape (4000, 1024), got {noisy.shape}/{target.shape}")

    model, config, checkpoint = load_model(checkpoint_path, device)
    f2p_raw, f2p_corrected, elapsed = infer(
        model, noisy, int(config["spectrum_length"]), args.batch_size, device
    )
    np.save(output_dir / "f2p_reconstructed_raw.npy", f2p_raw, allow_pickle=False)
    np.save(output_dir / "f2p_reconstructed_corrected.npy", f2p_corrected, allow_pickle=False)

    paper_nn = np.load(paper_nn_dir / "test_reconstructed.npy").astype(np.float32)
    savgol = savgol_filter(noisy, window_length=args.savgol_window, polyorder=args.savgol_order, axis=1).astype(np.float32)
    predictions = {
        "input": noisy,
        "paper_nn": paper_nn,
        "savgol": savgol,
        "f2p_raw": f2p_raw,
        "f2p_corrected": f2p_corrected,
    }
    correlations = {name: pearson_rows(target, values) for name, values in predictions.items()}

    fields = ["sample_index", "test_reference_id"] + [f"{name}_pearson" for name in predictions]
    with (output_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for index in range(len(target)):
            writer.writerow(
                [index, int(reference_id[index])]
                + [f"{correlations[name][index]:.8f}" for name in predictions]
            )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_score": checkpoint.get("score") if isinstance(checkpoint, dict) else None,
        "model_config": {key: config[key] for key in ("model_name", "spectrum_length", "patch_size", "embed_dim", "depth", "heads")},
        "device": str(device),
        "batch_size": args.batch_size,
        "samples": len(target),
        "inference_seconds": elapsed,
        "metrics": {name: metric_summary(target, values) for name, values in predictions.items()},
        "f2p_corrected_win_rate_vs_paper_nn": float(np.mean(correlations["f2p_corrected"] > correlations["paper_nn"])),
        "f2p_corrected_win_rate_vs_savgol": float(np.mean(correlations["f2p_corrected"] > correlations["savgol"])),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    export_leaderboard(summary, output_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
