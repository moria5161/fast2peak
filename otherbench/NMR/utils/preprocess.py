"""Rebuild the fixed native-6400 test data and supervised validation data."""

import argparse
import json
from pathlib import Path
import h5py
import numpy as np

from otherbench.NMR.utils.data import resample_rows, build_fixed_native_noisy


def write_data(path, target, ppm, ids, indices, noises):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.create_dataset("gt", data=target, compression="gzip", shuffle=True)
        h.create_dataset("ppm", data=ppm)
        h.create_dataset("sample_id", data=ids)
        h.create_dataset("source_index", data=indices)
        group = h.create_group("noise")
        for name, values in noises.items():
            group.create_dataset(name, data=values, compression="gzip", shuffle=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    args = parser.parse_args(argv)
    with h5py.File(args.splits) as h:
        source = h["test/spectra"][:].astype(np.float32)
        ppm0 = h["ppm"][:].astype(np.float32)
        target = resample_rows(source, 6400)
        ppm = np.linspace(float(ppm0[0]), float(ppm0[-1]), 6400, dtype=np.float32)
        noises = build_fixed_native_noisy(source, 6400, args.noise_levels, args.seed + 40_000_079)
        write_data(args.output, target, ppm, h["test/sample_id"][:], h["test/source_index"][:], noises)
        if args.validation_output:
            clean = h["val/spectra"][:].astype(np.float32)
            clean = clean / clean.max(axis=1, keepdims=True)
            val_noises = {}
            for index, sigma in enumerate(args.noise_levels):
                rng = np.random.default_rng(args.seed + index)
                val_noises[f"gaussian_sigma_{sigma:.3f}"] = (
                    clean + rng.normal(0.0, sigma, clean.shape)
                ).astype(np.float32)
            write_data(args.validation_output, clean, ppm0, h["val/sample_id"][:], h["val/source_index"][:], val_noises)
    print(json.dumps({"benchmark": str(args.output), "shape": list(target.shape),
                      "noise_seed": args.seed + 40_000_079}, indent=2))
