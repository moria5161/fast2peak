#!/usr/bin/env python3
"""Train and evaluate a PyTorch equivalent of the section 3.1 autoencoder."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter
from torch import nn
from torch.utils.data import DataLoader

from otherbench.IR.models.autoencoder import DenseAutoencoder
from otherbench.IR.utils.data import NpyPairDataset
from otherbench.IR.utils.metrics import pearson_rows
from otherbench.IR.utils.config import IR_ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data_dir = IR_ROOT / "datasets" / "paired"
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--output-dir", type=Path, default=IR_ROOT / "outputs" / "autoencoder")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--val-reference-fraction", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--savgol-window", type=int, default=15)
    parser.add_argument("--savgol-order", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.val_reference_fraction is None:
        args.val_reference_fraction = 0.0 if args.val_fraction > 0.0 else 0.10
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_squared_error = 0.0
    total_values = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            total_squared_error += torch.sum((predictions - targets) ** 2).item()
            total_values += targets.numel()
    return total_squared_error / total_values


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    batches = []
    with torch.no_grad():
        for inputs, _ in loader:
            batches.append(model(inputs.to(device)).cpu().numpy())
    return np.concatenate(batches).astype(np.float32, copy=False)


def save_history(history: list[dict[str, float]], output: Path) -> None:
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = args.data_dir.resolve()

    train_noisy = np.load(data / "train_noisy.npy", mmap_mode="r")
    train_target = np.load(data / "train_target.npy", mmap_mode="r")
    test_noisy = np.load(data / "test_noisy.npy", mmap_mode="r")
    test_target = np.load(data / "test_target.npy", mmap_mode="r")
    test_reference_id = np.load(data / "test_reference_id.npy")
    if train_noisy.shape != (45000, 1024) or test_noisy.shape != (4000, 1024):
        print(f"Warning: non-paper smoke-test shapes: train={train_noisy.shape}, test={test_noisy.shape}")
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if not 0.0 <= args.val_reference_fraction < 1.0:
        raise ValueError("val_reference_fraction must be in [0, 1)")
    if args.val_fraction > 0.0 and args.val_reference_fraction > 0.0:
        raise ValueError("Use either val_fraction or val_reference_fraction, not both")

    validation_reference_ids = None
    if args.val_reference_fraction > 0.0:
        train_reference_id = np.load(data / "train_reference_id.npy", mmap_mode="r")
        reference_ids = np.unique(train_reference_id)
        split_rng = np.random.default_rng(args.seed + 1)
        shuffled_reference_ids = split_rng.permutation(reference_ids)
        val_reference_count = max(1, int(round(len(reference_ids) * args.val_reference_fraction)))
        validation_reference_ids = np.sort(shuffled_reference_ids[:val_reference_count])
        validation_mask = np.isin(train_reference_id, validation_reference_ids)
        train_indices = np.flatnonzero(~validation_mask)
        val_indices = np.flatnonzero(validation_mask)
        train_inputs = np.asarray(train_noisy[train_indices])
        train_targets = np.asarray(train_target[train_indices])
        val_inputs = np.asarray(train_noisy[val_indices])
        val_targets = np.asarray(train_target[val_indices])
        print(
            f"Reference-level split: {len(train_inputs)} train / {len(val_inputs)} validation; "
            f"validation references={validation_reference_ids.tolist()}"
        )
    elif args.val_fraction > 0.0:
        split_rng = np.random.default_rng(args.seed + 1)
        permutation = split_rng.permutation(len(train_noisy))
        val_size = max(1, int(round(len(train_noisy) * args.val_fraction)))
        val_indices = permutation[:val_size]
        train_indices = permutation[val_size:]
        train_inputs = np.asarray(train_noisy[train_indices])
        train_targets = np.asarray(train_target[train_indices])
        val_inputs = np.asarray(train_noisy[val_indices])
        val_targets = np.asarray(train_target[val_indices])
        print(f"Training split: {len(train_inputs)} train / {len(val_inputs)} validation")
    else:
        train_inputs, train_targets = train_noisy, train_target
        val_inputs = val_targets = None

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    print(f"Using device: {device}")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        NpyPairDataset(train_inputs, train_targets), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", generator=generator,
    )
    val_loader = None
    if val_inputs is not None:
        val_loader = DataLoader(
            NpyPairDataset(val_inputs, val_targets), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
    test_loader = DataLoader(
        NpyPairDataset(test_noisy, test_target), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    model = DenseAutoencoder().to(device)
    # Match Keras Adam defaults used by model.compile(optimizer='adam').
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-7
    )
    criterion = nn.MSELoss()
    history: list[dict[str, float]] = []
    best_val_mse = float("inf")
    best_epoch = None
    best_state = None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_squared_error = 0.0
        total_values = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_squared_error += torch.sum((predictions.detach() - targets) ** 2).item()
            total_values += targets.numel()
        train_mse = total_squared_error / total_values
        val_mse = evaluate_loss(model, val_loader, device) if val_loader is not None else None
        row = {"epoch": epoch, "train_mse": train_mse}
        if val_mse is not None:
            row["val_mse"] = val_mse
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)
        val_text = f", val_mse={val_mse:.8f}" if val_mse is not None else ""
        print(f"Epoch {epoch:02d}/{args.epochs}: train_mse={train_mse:.8f}{val_text}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best validation model from epoch {best_epoch}: val_mse={best_val_mse:.8f}")
    final_test_mse = evaluate_loss(model, test_loader, device)
    print(f"Final held-out test MSE (evaluated once): {final_test_mse:.8f}")
    reconstructed = predict(model, test_loader, device)
    np.save(output / "test_reconstructed.npy", reconstructed, allow_pickle=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": "Dense(1024,128,ReLU) -> Dense(128,1024,ReLU)",
            "args": vars(args),
            "torch_version": torch.__version__,
        },
        output / "model.pt",
    )
    save_history(history, output)

    target_array = np.asarray(test_target)
    noisy_array = np.asarray(test_noisy)
    savgol = savgol_filter(noisy_array, window_length=args.savgol_window, polyorder=args.savgol_order, axis=1)
    input_corr = pearson_rows(target_array, noisy_array)
    nn_corr = pearson_rows(target_array, reconstructed)
    sg_corr = pearson_rows(target_array, savgol)
    metrics = np.column_stack((np.arange(len(test_target)), test_reference_id, input_corr, nn_corr, sg_corr))
    np.savetxt(
        output / "test_metrics.csv", metrics, delimiter=",",
        header="sample_index,test_reference_id,input_pearson,nn_pearson,savgol_pearson", comments="",
        fmt=["%d", "%d", "%.8f", "%.8f", "%.8f"],
    )
    summary = {
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "elapsed_seconds": time.time() - started,
        "samples": len(test_target),
        "final_train_mse": history[best_epoch - 1]["train_mse"] if best_epoch is not None else history[-1]["train_mse"],
        "final_test_mse": final_test_mse,
        "input_pearson_mean": float(input_corr.mean()),
        "nn_pearson_mean": float(nn_corr.mean()),
        "nn_pearson_median": float(np.median(nn_corr)),
        "savgol_pearson_mean": float(sg_corr.mean()),
        "savgol_pearson_median": float(np.median(sg_corr)),
    }
    if best_epoch is not None:
        summary["best_validation_epoch"] = best_epoch
        summary["best_validation_mse"] = best_val_mse
        summary["validation_split"] = "reference_level" if validation_reference_ids is not None else "sample_level"
        summary["validation_reference_ids"] = (
            validation_reference_ids.tolist() if validation_reference_ids is not None else None
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
