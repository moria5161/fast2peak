#!/usr/bin/env python3
"""QC real PXRD patterns, crop to 1600 points, and synthesize benchmark noise."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import yaml


from otherbench.XRD.utils.config import XRD_ROOT, DEFAULT_CONFIG, load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, default=XRD_ROOT / "datasets" / "raw")
    parser.add_argument("--output-dir", type=Path, default=XRD_ROOT / "datasets" / "benchmark")
    return parser.parse_args(argv)


def minmax_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    values -= values.min(axis=1, keepdims=True)
    maxima = values.max(axis=1, keepdims=True)
    np.divide(values, maxima, out=values, where=maxima > 0)
    return values


def qc_metrics(intensity: np.ndarray) -> dict[str, float]:
    q05, q99 = np.quantile(intensity, (0.05, 0.99))
    dynamic_range = float(q99 - q05)
    differences = np.diff(intensity)
    mad = np.median(np.abs(differences - np.median(differences)))
    noise_sigma = float(mad / 0.67448975 / np.sqrt(2.0))
    robust_snr = dynamic_range / max(noise_sigma, 1e-12)
    scaled = (intensity - q05) / max(dynamic_range, 1e-12)
    high_frequency_ratio = float(np.std(np.diff(scaled)) / max(np.std(scaled), 1e-12))
    return {
        "dynamic_range": dynamic_range,
        "noise_sigma_mad": noise_sigma,
        "robust_snr": robust_snr,
        "high_frequency_ratio": high_frequency_ratio,
    }


def add_noise(clean: np.ndarray, condition: dict, rng: np.random.Generator) -> np.ndarray:
    kind = condition["kind"]
    if kind == "gaussian":
        noisy = clean + rng.normal(0.0, float(condition["sigma"]), clean.shape)
    elif kind == "poisson":
        peak_counts = float(condition["peak_counts"])
        noisy = rng.poisson(np.clip(clean, 0, None) * peak_counts) / peak_counts
    elif kind == "mixed":
        peak_counts = float(condition["peak_counts"])
        noisy = rng.poisson(np.clip(clean, 0, None) * peak_counts) / peak_counts
        noisy += rng.normal(0.0, float(condition["sigma"]), clean.shape)
    else:
        raise ValueError(f"Unsupported noise kind: {kind}")
    # Keep the physical normalized-intensity/count scale in the benchmark file.
    # F2P performs its own per-spectrum min-max normalization at inference.
    return np.clip(noisy, 0.0, None).astype(np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_dir.rglob("*.xy"))
    if len(paths) != 815:
        raise ValueError(f"Expected 815 .xy files, found {len(paths)}")

    start = int(config["crop"]["start_index"])
    length = int(config["crop"]["length"])
    stop = start + length
    qc_config = config["quality_control"]
    rows, accepted, accepted_paths = [], [], []
    two_theta = None
    for path in paths:
        values = np.loadtxt(path, dtype=np.float64)
        if values.shape != (2406, 2):
            raise ValueError(f"Unexpected shape {values.shape}: {path}")
        axis, intensity = values[start:stop, 0], values[start:stop, 1]
        if two_theta is None:
            two_theta = axis
        elif not np.allclose(axis, two_theta, atol=1e-9):
            raise ValueError(f"Inconsistent 2theta axis: {path}")
        metrics = qc_metrics(intensity)
        reasons = []
        if not np.isfinite(intensity).all():
            reasons.append("nonfinite")
        if metrics["robust_snr"] < float(qc_config["min_robust_snr"]):
            reasons.append("low_robust_snr")
        if metrics["high_frequency_ratio"] > float(qc_config["max_high_frequency_ratio"]):
            reasons.append("high_frequency_noise")
        passed = not reasons
        rows.append({
            "relative_path": str(path.relative_to(input_dir)),
            "passed_qc": passed,
            "rejection_reasons": ";".join(reasons),
            **metrics,
        })
        if passed:
            accepted.append(intensity)
            accepted_paths.append(str(path.relative_to(input_dir)))

    clean = minmax_rows(np.asarray(accepted))
    np.save(output_dir / "clean_gt.npy", clean, allow_pickle=False)
    np.save(output_dir / "two_theta.npy", two_theta.astype(np.float32), allow_pickle=False)
    with (output_dir / "qc_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with h5py.File(output_dir / "benchmark.h5", "w") as handle:
        handle.create_dataset("gt", data=clean, compression="gzip", shuffle=True)
        handle.create_dataset("two_theta", data=two_theta.astype(np.float32))
        string_type = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("source_path", data=np.asarray(accepted_paths, dtype=object), dtype=string_type)
        noise_group = handle.create_group("noise")
        for condition_index, condition in enumerate(config["noise_conditions"]):
            rng = np.random.default_rng(int(config["seed"]) + condition_index)
            noisy = add_noise(clean, condition, rng)
            dataset = noise_group.create_dataset(
                condition["name"], data=noisy, compression="gzip", shuffle=True
            )
            for key, value in condition.items():
                dataset.attrs[key] = value

    summary = {
        "source_patterns": len(paths),
        "accepted_patterns": len(accepted),
        "rejected_patterns": len(paths) - len(accepted),
        "crop": {
            "start_index": start,
            "length": length,
            "two_theta_min": float(two_theta[0]),
            "two_theta_max": float(two_theta[-1]),
            "reason": "highest aggregate-signal contiguous 1600-point window; preserves 85.5% of normalized full-range signal",
        },
        "quality_control": qc_config,
        "gt_shape": list(clean.shape),
        "noise_conditions": config["noise_conditions"],
        "gt_caveat": "High-quality measured PXRD is used as reference GT; it is not physically noise-free ground truth.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksum_paths = [
        output_dir / name for name in
        ("benchmark.h5", "clean_gt.npy", "two_theta.npy", "qc_manifest.csv", "summary.json")
    ]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in checksum_paths) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
