#!/usr/bin/env python3
"""Prepare additive Gaussian IR pairs using the frozen benchmark references."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


from otherbench.IR.utils.config import IR_ROOT
from otherbench.IR.utils.upstream.functions import reduceSpecsToNWavenumbers
from otherbench.IR.utils.upstream.import_data import load_specCSVs_from_directory

DATA_DIR = IR_ROOT / "datasets" / "paired"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-dir", type=Path, default=IR_ROOT / "datasets")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--reselect-references", action="store_true",
        help="Run the original order-dependent PCA/KMeans selection instead of reusing frozen references.",
    )
    parser.add_argument("--train-noise-seed", type=int, default=0)
    # Figure2.py calculates len(trainSpectra) + len(testSpectra): 1024 + 1024.
    parser.add_argument("--test-noise-seed", type=int, default=2048)
    parser.add_argument("--train-variations", type=int, default=500)
    parser.add_argument("--test-variations", type=int, default=200)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=0.10)
    return parser.parse_args(argv)


def minmax_rows(values: np.ndarray) -> np.ndarray:
    """Equivalent to Reconstruction.prepareSpecSet(..., addDimension=False)."""
    values = values.astype(np.float32, copy=True)
    values -= values.min(axis=1, keepdims=True)
    maxima = values.max(axis=1, keepdims=True)
    np.divide(values, maxima, out=values, where=maxima != 0)
    return values


def make_pairs(
    references: np.ndarray, variations: int, sigma_min: float, sigma_max: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Tile each reference and normalize the clean target before adding noise.
    clean_columns = np.tile(references.T, (1, variations))
    rng = np.random.RandomState(seed)
    clean_rows = minmax_rows(clean_columns.T)
    # Zero-mean additive Gaussian noise: no random baseline or signal mixing.
    sigma_rows = rng.uniform(sigma_min, sigma_max, size=(len(clean_rows), 1)).astype(np.float32)
    noisy_rows = clean_rows + sigma_rows * rng.normal(size=clean_rows.shape).astype(np.float32)
    reference_id = np.tile(np.arange(len(references), dtype=np.int32), variations)
    return noisy_rows.astype(np.float32), clean_rows, reference_id, sigma_rows[:, 0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spectra_dir = args.spectra_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.reselect_references:
        if not any(path.suffix.lower() == ".csv" for path in spectra_dir.iterdir()):
            raise FileNotFoundError(
                "Original CSVs are archived separately. For reference reselection, "
                "provide --spectra-dir pointing to the original CSV directory."
            )
        # Preserve the original selection for provenance. This mode depends on
        # filesystem enumeration and sklearn versions; it is not the default.
        selected_names, selected_with_axis = load_specCSVs_from_directory(str(spectra_dir), maxSpectra=110)
        selected_with_axis = reduceSpecsToNWavenumbers(selected_with_axis, 1024)
        wavenumbers = selected_with_axis[:, 0].copy()
        selected_references = selected_with_axis[:, 1:].T
        source_files = [path for path in spectra_dir.iterdir() if path.suffix.lower() == ".csv"]
        selected_order = {name: index for index, name in enumerate(selected_names)}
        split_rows = []
        for source_index, path in enumerate(source_files):
            stem = path.name.lower().rsplit(".csv", 1)[0]
            selected_index = selected_order.get(stem)
            if selected_index is None:
                split, split_reference_id = "unused", ""
            elif selected_index < 90:
                split, split_reference_id = "train", selected_index
            else:
                split, split_reference_id = "test", selected_index - 90
            split_rows.append({
                "source_os_listdir_index": source_index,
                "filename": path.name,
                "split": split,
                "split_reference_id": split_reference_id,
                "selected_order": "" if selected_index is None else selected_index,
            })
        source_count = len(source_files)
    else:
        reference_dir = args.reference_dir.resolve()
        frozen_manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
        selected_names = frozen_manifest["selection"]["train_names"] + frozen_manifest["selection"]["test_names"]
        selected_references = np.load(reference_dir / "selected_clean.npy", allow_pickle=False)
        wavenumbers = np.load(reference_dir / "wavenumbers.npy", allow_pickle=False)
        with (reference_dir / "split.csv").open(newline="", encoding="utf-8") as handle:
            split_rows = list(csv.DictReader(handle))
        source_count = frozen_manifest["source_spectra"]
    if selected_references.shape != (110, 1024):
        raise ValueError(f"Expected 110 x 1024 selected references, got {selected_references.shape}")

    train_names, test_names = selected_names[:90], selected_names[90:]
    train_noisy, train_target, train_reference_id, train_sigma = make_pairs(
        selected_references[:90], args.train_variations, args.sigma_min, args.sigma_max, args.train_noise_seed
    )
    test_noisy, test_target, test_reference_id, test_sigma = make_pairs(
        selected_references[90:], args.test_variations, args.sigma_min, args.sigma_max, args.test_noise_seed
    )

    arrays = {
        "wavenumbers.npy": wavenumbers.astype(np.float32),
        "selected_clean.npy": minmax_rows(selected_references),
        "train_noisy.npy": train_noisy,
        "train_target.npy": train_target,
        "train_reference_id.npy": train_reference_id,
        "train_sigma.npy": train_sigma,
        "test_noisy.npy": test_noisy,
        "test_target.npy": test_target,
        "test_reference_id.npy": test_reference_id,
        "test_sigma.npy": test_sigma,
    }
    for name, array in arrays.items():
        np.save(output / name, array, allow_pickle=False)

    with (output / "split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=split_rows[0].keys())
        writer.writeheader()
        writer.writerows(split_rows)

    manifest = {
        "protocol": "IR additive Gaussian benchmark; reference selection adapted from section 3.1 / Figure 2",
        "data_flow": (
            "author importData.py/functions.py; original order-dependent selection"
            if args.reselect_references else "frozen selected_clean.npy, wavenumbers.npy and split.csv; fixed reference order"
        ),
        "source_directory": str(spectra_dir),
        "source_spectra": source_count,
        "selection": {
            "method": "author getNMostDifferentSpectra: StandardScaler + PCA(2) + k_means(110)",
            "author_random_state": 42,
            "selected_references": len(selected_names),
            "train_references": len(train_names),
            "test_references": len(test_names),
            "train_names": train_names,
            "test_names": test_names,
        },
        "wavenumber_interval_cm-1": [float(wavenumbers[0]), float(wavenumbers[-1])],
        "n_wavenumbers": len(wavenumbers),
        "noise": {
            "formula": "clean = minmax(clean); noisy = clean + sigma*Normal(0,1); sigma sampled independently per spectrum; no baseline mixing or post-noise min-max",
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "train_seed": args.train_noise_seed,
            "test_seed": args.test_noise_seed,
        },
        "train": {"variations_per_reference": args.train_variations, "samples": len(train_noisy)},
        "test": {"variations_per_reference": args.test_variations, "samples": len(test_noisy)},
        "arrays": {name: {"shape": list(array.shape), "dtype": str(array.dtype)} for name, array in arrays.items()},
        "software": {"numpy": np.__version__},
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    checksums = ["sha256  file"]
    for path in sorted(output.iterdir()):
        if path.name != "checksums.sha256" and path.is_file():
            checksums.append(f"{sha256(path)}  {path.name}")
    (output / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
