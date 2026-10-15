#!/usr/bin/env python3
"""Create a fixed molecule-level data2 train/validation/test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=701)
    parser.add_argument("--val-count", type=int, default=150)
    parser.add_argument("--test-count", type=int, default=150)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_ids(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_path = args.source_h5.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as source:
        spectra = source["gt"][:].astype(np.float32)
        ppm = source["ppm"][:].astype(np.float32)
        sample_ids = decode_ids(source["sample_id"][:])
    counts = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }
    if sum(counts.values()) != len(spectra):
        raise ValueError(f"split counts {counts} do not sum to {len(spectra)}")
    if len(sample_ids) != len(spectra) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be unique and aligned with spectra")

    indices = np.arange(len(spectra), dtype=np.int64)
    np.random.default_rng(args.seed).shuffle(indices)
    split_indices: dict[str, np.ndarray] = {}
    offset = 0
    for name, count in counts.items():
        split_indices[name] = indices[offset : offset + count]
        offset += count

    destination_path = output_dir / "splits.h5"
    with h5py.File(destination_path, "w") as destination:
        destination.attrs["seed"] = args.seed
        destination.attrs["split_unit"] = "unique paired molecule/spectrum record"
        destination.attrs["source_sha256"] = sha256(source_path)
        destination.create_dataset("ppm", data=ppm)
        string_type = h5py.string_dtype(encoding="utf-8")
        for name, selected in split_indices.items():
            group = destination.create_group(name)
            group.create_dataset("source_index", data=selected)
            group.create_dataset(
                "sample_id",
                data=np.asarray([sample_ids[index] for index in selected], dtype=object),
                dtype=string_type,
            )
            group.create_dataset(
                "spectra",
                data=spectra[selected],
                compression="gzip",
                shuffle=True,
            )

    summary = {
        "seed": args.seed,
        "source_h5": str(source_path),
        "source_sha256": sha256(source_path),
        "split_unit": "unique paired molecule/spectrum record",
        "counts": counts,
        "indices": {name: values.tolist() for name, values in split_indices.items()},
        "sample_ids": {
            name: [sample_ids[index] for index in values]
            for name, values in split_indices.items()
        },
    }
    (output_dir / "splits.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"seed": args.seed, "counts": counts}, indent=2))
    print(destination_path)


if __name__ == "__main__":
    main()
