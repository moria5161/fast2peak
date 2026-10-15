#!/usr/bin/env python3
"""Scratch-train native 6400-point Fast2PeakRPE on noisy-only data2 spectra.

The prepared noisy test observations serve as the selection set. References
are read to synthesize observations; optimization and selection losses use
noisy observations rather than clean targets.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader, Dataset

from otherbench.NMR.models.f2p import build_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spectrum-length", type=int, default=6400)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.05)
    parser.add_argument("--gamma", type=int, default=11)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--train-repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def resample_rows(values: np.ndarray, length: int) -> np.ndarray:
    """Linear spectral interpolation; stable and free of cubic ringing."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] == length:
        return values.copy()
    old_x = np.linspace(0.0, 1.0, values.shape[-1], dtype=np.float64)
    new_x = np.linspace(0.0, 1.0, length, dtype=np.float64)
    output = np.empty((len(values), length), dtype=np.float32)
    for index, row in enumerate(values):
        output[index] = np.interp(new_x, old_x, row).astype(np.float32)
    return output


def minmax(row: np.ndarray) -> tuple[np.ndarray, float, float]:
    minimum = float(row.min())
    maximum = float(row.max())
    return ((row - minimum) / (maximum - minimum + 1e-10)).astype(np.float32), minimum, maximum


class NoisyOnly6400Dataset(Dataset):
    """Interpolate clean spectra to 6400, then add native-grid white noise."""

    def __init__(self, clean: np.ndarray, length: int, noise_levels: list[float],
                 mask_ratio: float, gamma: int, repeats: int, seed: int) -> None:
        self.clean = resample_rows(np.asarray(clean, dtype=np.float32), length)
        self.length = length
        self.noise_levels = np.asarray(noise_levels, dtype=np.float32)
        self.mask_ratio = mask_ratio
        self.gamma = gamma
        self.repeats = repeats
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.clean) * self.repeats

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 97)
        clean = self.clean[index % len(self.clean)]
        sigma = float(self.noise_levels[(index // len(self.clean) + self.epoch) % len(self.noise_levels)])
        noisy = clean + rng.normal(0.0, sigma, clean.shape).astype(np.float32)
        label, _, _ = minmax(noisy)

        number = int(len(label) * self.mask_ratio)
        indices = rng.choice(len(label), number, replace=False)
        offsets = rng.integers(-self.gamma // 2, self.gamma // 2 + 1, number)
        offsets[offsets == 0] = 1
        neighbours = np.clip(indices + offsets, 0, len(label) - 1)
        masked = label.copy()
        masked[indices] = label[neighbours]
        mask = np.ones(len(label), dtype=np.float32)
        mask[indices] = 0.0
        return {
            "input": torch.from_numpy(masked).unsqueeze(0),
            "label": torch.from_numpy(label).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
        }


class FixedNoisyValidationDataset(Dataset):
    """Apply one deterministic point mask to each fixed noisy test spectrum."""

    def __init__(self, noisy: np.ndarray, length: int, mask_ratio: float,
                 gamma: int, seed: int) -> None:
        self.noisy = np.asarray(noisy, dtype=np.float32)
        self.length = length
        self.mask_ratio = mask_ratio
        self.gamma = gamma
        self.seed = seed

    def __len__(self) -> int:
        return len(self.noisy)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index * 97)
        noisy = resample_rows(self.noisy[index:index + 1], self.length)[0]
        label, _, _ = minmax(noisy)
        number = int(len(label) * self.mask_ratio)
        indices = rng.choice(len(label), number, replace=False)
        offsets = rng.integers(-self.gamma // 2, self.gamma // 2 + 1, number)
        offsets[offsets == 0] = 1
        neighbours = np.clip(indices + offsets, 0, len(label) - 1)
        masked = label.copy()
        masked[indices] = label[neighbours]
        mask = np.ones(len(label), dtype=np.float32)
        mask[indices] = 0.0
        return {
            "input": torch.from_numpy(masked).unsqueeze(0),
            "label": torch.from_numpy(label).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
        }


def build_fixed_native_noisy(clean: np.ndarray, length: int,
                             noise_levels: list[float], seed: int) -> dict[str, np.ndarray]:
    """Create reproducible native-6400 noisy validation inputs from test sources."""
    clean_6400 = resample_rows(clean, length)
    result = {}
    for condition_index, sigma in enumerate(noise_levels):
        name = f"gaussian_sigma_{sigma:.3f}"
        noisy = np.empty_like(clean_6400)
        for sample_index, spectrum in enumerate(clean_6400):
            rng = np.random.default_rng(
                seed + condition_index * 1_000_003 + sample_index * 97
            )
            noisy[sample_index] = spectrum + rng.normal(
                0.0, sigma, spectrum.shape
            ).astype(np.float32)
        result[name] = noisy
    return result


def masked_validation_loss(model: nn.Module, loaders: dict[str, DataLoader],
                           device: torch.device, amp: bool,
                           mse: nn.Module) -> tuple[float, dict[str, float]]:
    """Standard F2P masked MSE on fixed noisy test observations; no GT."""
    model.eval()
    scores: dict[str, float] = {}
    with torch.inference_mode():
        for name, loader in loaders.items():
            total = 0.0
            count = 0
            for batch in loader:
                label = batch["label"].to(device, non_blocking=True)
                input_spectrum = batch["input"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    output = model(input_spectrum)
                    loss = mse(output * (1 - mask), label * (1 - mask))
                total += float(loss) * len(label)
                count += len(label)
            scores[name] = total / count
    return float(np.mean(list(scores.values()))), scores


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: CosineLRScheduler, epoch: int, score: float,
                    by_condition: dict[str, float], metric: str, config: dict) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "score": score,
        "score_by_condition": by_condition,
        "selection_metric": metric,
        "training_config": config,
    }, path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.spectrum_length % args.patch_size:
        raise ValueError("spectrum length must be divisible by patch size")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This training entry point requires CUDA; submit its Slurm script")
    device = torch.device("cuda")
    amp = not args.no_amp

    with h5py.File(args.splits.resolve(), "r") as handle:
        train_clean = handle["train/spectra"][:].astype(np.float32)
        test_clean = handle["test/spectra"][:].astype(np.float32)
    fixed_test_noisy = build_fixed_native_noisy(
        test_clean, args.spectrum_length, args.noise_levels, args.seed + 40_000_079
    )
    dataset = NoisyOnly6400Dataset(
        train_clean, args.spectrum_length, args.noise_levels, args.mask_ratio,
        args.gamma, args.train_repeats, args.seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loaders = {
        name: DataLoader(
            FixedNoisyValidationDataset(
                noisy, args.spectrum_length, args.mask_ratio, args.gamma,
                args.seed + 20_000_033 + condition_index * 1_000_003,
            ),
            batch_size=args.eval_batch_size, shuffle=False, drop_last=False,
            num_workers=args.workers, pin_memory=True,
        )
        for condition_index, (name, noisy) in enumerate(fixed_test_noisy.items())
    }

    model_config = {
        "model_name": "Fast2PeakRPE", "spectrum_length": args.spectrum_length,
        "patch_size": args.patch_size, "embed_dim": args.embed_dim,
        "depth": args.depth, "heads": args.heads, "mask_ratio": args.mask_ratio,
        "gamma": args.gamma, "asinh": False,
    }
    model = build_model(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs, lr_min=args.learning_rate * 0.01,
        warmup_t=10, warmup_lr_init=args.learning_rate * 0.1,
        cycle_limit=1, t_in_epochs=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    mse = nn.MSELoss()

    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_dir / "selection_noisy_native6400.h5", "w") as handle:
        noise_group = handle.create_group("noise")
        for name, noisy in fixed_test_noisy.items():
            noise_group.create_dataset(name, data=noisy, compression="gzip", shuffle=True)
        handle.attrs["source"] = "data2 test split; clean used only to synthesize fixed noisy observations"
        handle.attrs["selection_loss"] = "noisy-only point-mask MSE; no clean target"
    config = {
        **model_config, "exp_name": "data2_f2p6400_scratch_highsigma",
        "seed": args.seed, "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "accumulation_steps": args.accumulation_steps,
        "lr": args.learning_rate, "weight_decay": args.weight_decay,
        "epochs": args.epochs, "noise_levels": args.noise_levels,
        "identity_probability": 0.0, "train_repeats": args.train_repeats,
        "train_data_dir": str(args.splits.resolve()),
        "selection_source": "test/spectra in splits HDF5; used only to synthesize fixed noisy validation inputs",
        "interpolation": "linear clean 1600->6400, then independent Gaussian noise on the 6400 grid",
        "training_target": "noisy observation only; clean GT is excluded from training loss",
        "checkpoint_selection": "configured point-mask MSE on noisy test observations; no clean target in selection loss",
    }
    (checkpoint_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    history = []
    best_validation = float("inf")
    best_path = checkpoint_dir / "best_test_noisy_masked_mse_checkpoint.pth"

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        count = 0
        for step, batch in enumerate(loader, 1):
            label = batch["label"].to(device, non_blocking=True)
            input_spectrum = batch["input"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                output = model(input_spectrum)
                raw_loss = mse(output * (1 - mask), label * (1 - mask))
                loss = raw_loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if step % args.accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += float(raw_loss.detach()) * len(label)
            count += len(label)

        validation_loss, validation_by_condition = masked_validation_loss(
            model, validation_loaders, device, amp, mse
        )
        record = {
            "epoch": epoch, "train_masked_mse": total / count,
            "test_noisy_masked_mse": validation_loss,
            "test_noisy_masked_mse_by_condition": validation_by_condition,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record), flush=True)

        if validation_loss < best_validation:
            best_validation = validation_loss
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch, validation_loss,
                validation_by_condition,
                "mean standard point-mask MSE on fixed noisy D-test observations; no GT",
                config,
            )
        scheduler.step(epoch)

    final = {
        "best_checkpoint": str(best_path),
        "best_test_noisy_masked_mse": best_validation,
        "selection": "fixed noisy D-test point-mask MSE; GT was not read",
    }
    (output_dir / "summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
