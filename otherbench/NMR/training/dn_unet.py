#!/usr/bin/env python3
"""Two-stage supervised DN-Unet fine-tuning on fixed data2 splits."""

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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


from otherbench.NMR.utils.config import NMR_ROOT
from otherbench.NMR.models.dn_unet import DNUNet

DEFAULT_PRETRAINED = NMR_ROOT / "outputs" / "pretrained" / "dn_unet.pth"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--noise-levels", type=float, nargs="+", required=True)
    parser.add_argument("--identity-probability", type=float, default=0.10)
    parser.add_argument("--train-repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--refit-only-epochs", type=int, default=0)
    parser.add_argument("--stage-a-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def resample_rows(values: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] == target_length:
        return values.copy()
    source_grid = np.linspace(0.0, 1.0, values.shape[-1])
    target_grid = np.linspace(0.0, 1.0, target_length)
    output = np.empty((len(values), target_length), dtype=np.float32)
    for index, row in enumerate(values):
        output[index] = np.interp(target_grid, source_grid, row)
    return output


class PairedData2Dataset(Dataset):
    """Synthesize noise on the 1600-point grid, then adapt pairs to 8192."""

    def __init__(
        self,
        clean: np.ndarray,
        model_length: int,
        noise_levels: list[float],
        identity_probability: float,
        repeats: int,
        seed: int,
    ) -> None:
        self.clean = np.asarray(clean, dtype=np.float32)
        self.clean_model = resample_rows(self.clean, model_length)
        self.model_length = model_length
        self.noise_levels = np.asarray(noise_levels, dtype=np.float32)
        self.identity_probability = identity_probability
        self.repeats = repeats
        self.seed = seed
        self.epoch = 0
        self.source_grid = np.linspace(0.0, 1.0, self.clean.shape[1])
        self.model_grid = np.linspace(0.0, 1.0, model_length)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.clean) * self.repeats

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 97)
        spectrum_index = index % len(self.clean)
        clean_source = self.clean[spectrum_index]
        sigma = 0.0 if rng.random() < self.identity_probability else float(
            rng.choice(self.noise_levels)
        )
        noisy_source = clean_source + rng.normal(
            0.0, sigma, clean_source.shape
        ).astype(np.float32)
        # Match evaluate_nmr.infer_author_network exactly: scale on the 1600-point
        # noisy input, then linearly interpolate to DN-Unet's native 8192 grid.
        scale = max(float(noisy_source.max()), 1e-8)
        noisy_model = np.interp(
            self.model_grid, self.source_grid, noisy_source
        ).astype(np.float32) / scale
        clean_model = self.clean_model[spectrum_index] / scale
        return (
            torch.from_numpy(noisy_model).unsqueeze(0),
            torch.from_numpy(clean_model).unsqueeze(0),
        )


def normalized_rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error_norm = torch.sqrt(torch.sum((prediction - target) ** 2, dim=(1, 2)))
    target_norm = torch.sqrt(torch.sum(target**2, dim=(1, 2))).clamp_min(1e-12)
    return torch.mean(error_norm / target_norm)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DNUNet()
    state = checkpoint.get("model_state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device), checkpoint


def validation_rmse(
    model: nn.Module,
    validation: dict[str, tuple[np.ndarray, np.ndarray]],
    model_length: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[float, dict[str, float]]:
    model.eval()
    scores: dict[str, float] = {}
    with torch.inference_mode():
        for condition, (noisy, target) in validation.items():
            scale = noisy.max(axis=1, keepdims=True).astype(np.float32)
            scale = np.where(scale > 0, scale, 1.0)
            normalized = (noisy / scale).astype(np.float32)
            predictions = []
            for start in range(0, len(noisy), batch_size):
                batch = torch.from_numpy(normalized[start : start + batch_size]).unsqueeze(1)
                batch = F.interpolate(
                    batch.to(device), size=model_length, mode="linear", align_corners=True
                )
                with torch.autocast(device_type=device.type, enabled=amp):
                    prediction = model(batch)
                prediction = F.interpolate(
                    prediction, size=noisy.shape[1], mode="linear", align_corners=True
                )
                predictions.append(prediction.squeeze(1).float().cpu().numpy())
            prediction_source = np.concatenate(predictions) * scale
            if not np.isfinite(prediction_source).all():
                raise FloatingPointError(
                    f"DN-Unet produced non-finite validation output for {condition}"
                )
            per_spectrum = np.sqrt(np.mean((prediction_source - target) ** 2, axis=1))
            scores[condition] = float(per_spectrum.mean())
    return float(np.mean(list(scores.values()))), scores


def set_encoder_frozen(model: nn.Module, frozen: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not frozen)


def freeze_batch_norm_statistics(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_stage(
    model: nn.Module,
    dataset: PairedData2Dataset,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    amp: bool,
    validation: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    stage_name: str,
    checkpoint_dir: Path,
) -> tuple[list[dict], int, float]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history: list[dict] = []
    best_epoch = 0
    best_score = float("inf")
    if validation is not None:
        best_score, by_condition = validation_rmse(
            model, validation, dataset.model_length, args.batch_size, device, amp
        )
        torch.save(
            {
                "model_name": "dn_unet",
                "model_state_dict": model.state_dict(),
                "epoch": 0,
                "score": best_score,
                "score_by_condition": by_condition,
            },
            checkpoint_dir / "stage_a_best_val_checkpoint.pth",
        )
        print(json.dumps({"stage": stage_name, "epoch": 0, "val_gt_rmse": best_score,
                          "val_gt_rmse_by_condition": by_condition}), flush=True)

    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        set_encoder_frozen(model, epoch <= args.freeze_encoder_epochs)
        model.train()
        freeze_batch_norm_statistics(model)
        total = 0.0
        count = 0
        started = time.time()
        for noisy, clean in loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = normalized_rmse(model(noisy), clean)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite DN-Unet training loss at epoch {epoch}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(noisy)
            count += len(noisy)
        record = {
            "stage": stage_name,
            "epoch": epoch,
            "train_author_relative_l2": total / max(count, 1),
            "encoder_frozen": epoch <= args.freeze_encoder_epochs,
            "seconds": time.time() - started,
        }
        if validation is not None:
            score, by_condition = validation_rmse(
                model, validation, dataset.model_length, args.batch_size, device, amp
            )
            record["val_gt_rmse"] = score
            record["val_gt_rmse_by_condition"] = by_condition
            if score < best_score - 1e-10:
                best_epoch = epoch
                best_score = score
                torch.save(
                    {
                        "model_name": "dn_unet",
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "score": score,
                        "score_by_condition": by_condition,
                        "selection_metric": "mean full-spectrum GT RMSE on fixed validation",
                    },
                    checkpoint_dir / "stage_a_best_val_checkpoint.pth",
                )
        history.append(record)
        print(json.dumps(record), flush=True)
    return history, best_epoch, best_score


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp = device.type == "cuda" and not args.no_amp
    model_length = 8192

    with h5py.File(args.splits.resolve(), "r") as source:
        train_clean = source["train/spectra"][:].astype(np.float32)
        val_clean = source["val/spectra"][:].astype(np.float32)
    with h5py.File(args.validation_data.resolve(), "r") as source:
        target = source["gt"][:].astype(np.float32)
        validation = {
            name: (source["noise"][name][:].astype(np.float32), target)
            for name in source["noise"]
        }
    if len(target) != len(val_clean):
        raise ValueError("validation benchmark and split are not aligned")

    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pretrained = args.pretrained.resolve()
    model, source_checkpoint = load_model(pretrained, device)
    if args.refit_only_epochs > 0:
        if args.stage_a_checkpoint is None:
            raise ValueError("--stage-a-checkpoint is required with --refit-only-epochs")
        selection_path = args.stage_a_checkpoint.resolve()
        selection = torch.load(selection_path, map_location="cpu", weights_only=False)
        selected_epoch = int(selection["epoch"])
        if selected_epoch != args.refit_only_epochs:
            raise ValueError(
                "refit epoch count does not match stage-A checkpoint: "
                f"{args.refit_only_epochs} != {selected_epoch}"
            )
        selected_score = float(selection["score"])
        stage_a_history = []
        print(json.dumps({
            "stage": "selection_reused",
            "stage_a_checkpoint": str(selection_path),
            "selected_epoch": selected_epoch,
            "selected_validation_rmse": selected_score,
            "stage_a_weights_used_for_refit": False,
        }), flush=True)
    else:
        stage_a_dataset = PairedData2Dataset(
            train_clean, model_length, args.noise_levels, args.identity_probability,
            args.train_repeats, args.seed
        )
        stage_a_history, selected_epoch, selected_score = train_stage(
            model, stage_a_dataset, args.epochs, args, device, amp, validation,
            "train_selection", checkpoint_dir
        )
        selected_epoch = max(selected_epoch, 1)
    print(json.dumps({"selected_epoch": selected_epoch,
                      "selected_validation_rmse": selected_score,
                      "stage_b_spectra": int(len(train_clean) + len(val_clean))}), flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    final_model, _ = load_model(pretrained, device)
    combined = np.concatenate((train_clean, val_clean), axis=0)
    stage_b_dataset = PairedData2Dataset(
        combined, model_length, args.noise_levels, args.identity_probability,
        args.train_repeats, args.seed + 70_000_019
    )
    stage_b_history, _, _ = train_stage(
        final_model, stage_b_dataset, selected_epoch, args, device, amp, None,
        "train_plus_validation_refit", checkpoint_dir
    )

    config = {
        **vars(args),
        "splits": str(args.splits.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "output_dir": str(output_dir),
        "pretrained": str(pretrained),
        "model_native_length": model_length,
        "source_length": int(train_clean.shape[1]),
        "training_loss": "mean per-sample relative L2 (DN-Unet author loss)",
        "selection_metric": "mean full-spectrum GT RMSE on fixed validation",
        "noise_synthesis": "Gaussian on source 1600 grid before linear adaptation to 8192",
        "normalization": "positive maximum of noisy source, shared with clean target",
        "batch_norm_statistics": "frozen",
        "selected_epoch": selected_epoch,
        "selected_validation_rmse": selected_score,
        "final_fit_scope": "train+validation, 851 spectra",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    final_path = output_dir / "final_train_plus_val_checkpoint.pth"
    torch.save(
        {
            "model_name": "dn_unet",
            "model_state_dict": final_model.state_dict(),
            "epoch": selected_epoch,
            "selection_score": selected_score,
            "training_config": config,
            "source_checkpoint_epoch": source_checkpoint.get("epoch"),
        },
        final_path,
    )
    (output_dir / "history.json").write_text(
        json.dumps({"stage_a": stage_a_history, "stage_b": stage_b_history}, indent=2)
        + "\n", encoding="utf-8"
    )
    print(json.dumps({"final_checkpoint": str(final_path),
                      "selected_epoch": selected_epoch,
                      "selected_validation_rmse": selected_score}, indent=2))


if __name__ == "__main__":
    main()
