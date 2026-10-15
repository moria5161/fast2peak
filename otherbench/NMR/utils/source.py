"""Prepare the original Maybridge matrix without generating any figures."""

import argparse
import json
import re
from pathlib import Path
import h5py
import numpy as np

OBSERVATION_CENTER_PPM = 6.175
OBSERVATION_WIDTH_PPM = 20.693
SOURCE_POINTS = 1024


def load_compound_codes(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return re.findall(r">\s*<compound_code>\s*\r?\n([^\r\n]+)", text)


def published_bin_centers(points: int) -> np.ndarray:
    """Reconstruct descending bin centers from the published window and center.

    The supplementary TXT does not include an x axis.  Using bin centers keeps
    the represented observation window exactly 20.693 ppm without pretending
    that the first and last samples are the acquisition-window boundaries.
    """
    bin_width = OBSERVATION_WIDTH_PPM / points
    high = OBSERVATION_CENTER_PPM + OBSERVATION_WIDTH_PPM / 2.0 - bin_width / 2.0
    low = OBSERVATION_CENTER_PPM - OBSERVATION_WIDTH_PPM / 2.0 + bin_width / 2.0
    return np.linspace(high, low, points, dtype=np.float64)


def resample_rows(values: np.ndarray, output_length: int) -> tuple[np.ndarray, np.ndarray]:
    source_position = np.linspace(0.0, 1.0, values.shape[1], dtype=np.float64)
    target_position = np.linspace(0.0, 1.0, output_length, dtype=np.float64)
    output = np.empty((len(values), output_length), dtype=np.float32)
    for index, spectrum in enumerate(values):
        output[index] = np.interp(target_position, source_position, spectrum)
    source_ppm = published_bin_centers(values.shape[1])
    ppm = np.interp(target_position, source_position, source_ppm).astype(np.float32)
    return output, ppm



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-txt", type=Path, required=True)
    parser.add_argument("--molecule-sdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    source = np.loadtxt(args.spectra_txt, delimiter=",", dtype=np.float32)
    codes = load_compound_codes(args.molecule_sdf)
    if source.ndim != 2 or source.shape[1] != 1024 or len(codes) != len(source):
        raise ValueError("Expected a paired [N,1024] spectral matrix and matching SDF")
    if not np.isfinite(source).all() or len(set(codes)) != len(codes):
        raise ValueError("Nonfinite spectra or duplicate compound codes")
    maxima = source.max(axis=1, keepdims=True)
    valid = maxima[:, 0] > 0
    normalized = source[valid] / maxima[valid]
    clean, ppm = resample_rows(normalized, 1600)
    ids = np.asarray([c for c, keep in zip(codes, valid) if keep], dtype=object)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(output / "clean.h5", "w") as h:
        h.create_dataset("gt", data=clean, compression="gzip", shuffle=True)
        h.create_dataset("ppm", data=ppm)
        h.create_dataset("sample_id", data=ids, dtype=h5py.string_dtype("utf-8"))
        h.attrs["source_article_doi"] = "10.1186/1758-2946-6-9"
    summary = {"source_shape": list(source.shape), "clean_shape": list(clean.shape),
               "source_article_doi": "10.1186/1758-2946-6-9",
               "normalization": "positive maximum before 1024->1600 linear interpolation",
               "ppm_axis": "descending bin centers, 20.693 ppm window centered at 6.175 ppm"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
