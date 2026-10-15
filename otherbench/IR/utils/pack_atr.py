#!/usr/bin/env python3
"""Pack all ATR CSVs into a column-oriented array using the upstream alignment rule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


from otherbench.IR.utils.upstream.import_data import get_numbers_from_line


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing the archived original CSVs")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    input_dir = args.input_dir.resolve()
    output = args.output.resolve() if args.output else input_dir / "atr_spectra.npy"
    files = sorted(
        (path for path in input_dir.iterdir() if path.suffix.lower() == ".csv"),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not files:
        raise ValueError(f"No CSV spectra found in {input_dir}")

    spectra = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            data = np.asarray(
                [get_numbers_from_line(line) for line in handle if line.strip()],
                dtype=np.float64,
            )
        if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all():
            raise ValueError(f"Invalid two-column spectrum: {path}")
        spectra.append(data)

    axis = spectra[0][:, 0]
    if not np.all(np.diff(axis) > 0):
        raise ValueError("The reference wavenumber axis must be strictly increasing")
    packed = np.empty((len(axis), len(files) + 1), dtype=np.float64)
    packed[:, 0] = axis
    columns = []
    for column, (path, data) in enumerate(zip(files, spectra), start=1):
        identical_axis = np.array_equal(axis, data[:, 0])
        if identical_axis:
            packed[:, column] = data[:, 1]
        else:
            # Match importData.load_specCSVs_from_directory: nearest original
            # point, with np.argmin's first-match behavior for exact ties.
            indices = np.abs(data[:, 0, None] - axis[None, :]).argmin(axis=0)
            packed[:, column] = data[indices, 1]
        out_of_range = (axis < data[:, 0].min()) | (axis > data[:, 0].max())
        columns.append({
            "column_index": column,
            "material": path.stem,
            "source_csv": path.name,
            "original_points": len(data),
            "aligned": not identical_axis,
            "endpoint_extended_points": int(out_of_range.sum()),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, packed, allow_pickle=False)
    print(json.dumps({
        "output": str(output),
        "shape": list(packed.shape),
        "dtype": str(packed.dtype),
        "reference_axis_csv": files[0].name,
        "wavenumber_range_cm-1": [float(axis[0]), float(axis[-1])],
        "columns": columns,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
