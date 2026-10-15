# Experimental PXRD data and provenance

The local data are associated with [Mapping the crystallization landscape of rare earth MOFs](https://pubs.rsc.org/en/content/articlehtml/2026/sc/d5sc09992g), published on the Royal Society of Chemistry website. The benchmark uses the single-metal and binary-metal high-throughput experimental PXRD collections.

## Raw data

Raw files are stored under `../datasets/raw/` (Git-ignored). All 815 XY files have two headerless columns: 2θ (degrees) and measured intensity. Each contains 2,406 points, spanning 2.010–38.085° with a 0.015° step. Original relative filenames remain traceable through [qc_manifest.csv](qc_manifest.csv) and the prepared HDF5 `source_path` field.

| Collection | Raw patterns | Retained | Rejected |
|---|---:|---:|---:|
| `pxrd_single-metal-syntheses` | 723 | 656 | 67 |
| `pxrd_binary-metal-syntheses` | 92 | 59 | 33 |
| Total | 815 | 715 | 100 |

## Reference construction

Each raw pattern is cropped to its first 1,600 points, covering 2.010–25.995°. QC requires robust SNR ≥ 30 and high-frequency ratio ≤ 0.60. Robust SNR is `(Q99 - Q05) / sigma`, where sigma is estimated from the MAD of adjacent differences; the high-frequency ratio is `std(diff(y)) / std(y)`. Retained intensities are min-max normalized per pattern.

GT is a high-quality measured experimental reference, not physically noise-free ground truth. No smoothing algorithm is used to generate the reference. The retained 715 references have shape `(715, 1600)`.

## Synthetic noise

Seven conditions are generated from the normalized references:

| Condition | Construction |
|---|---|
| Gaussian σ = 0.02, 0.05, 0.10 | `GT + Normal(0, sigma)` |
| Poisson peak counts = 100, 50, 10 | `Poisson(GT * peak_counts) / peak_counts` |
| Mixed peak counts = 500, σ = 0.03 | Poisson sampling at peak counts 500, then additive Gaussian noise |

Unlike the IR noise protocol, this existing XRD protocol clips negative generated intensities to zero. Each condition uses `default_rng(42 + condition_index)`. Condition order and parameters are fixed in `configs/benchmark.yaml`; noisy spectra remain on the generated intensity scale, with per-spectrum normalization performed inside F2P inference.

## Prepared data

`../datasets/benchmark/` contains `benchmark.h5`, `clean_gt.npy`, `two_theta.npy`, `qc_manifest.csv`, `summary.json` and the historical checksum record. The HDF5 contains `gt`, `two_theta`, `source_path` and a `noise/` group with seven datasets. Public provenance copies are [data_summary.json](data_summary.json) and [qc_manifest.csv](qc_manifest.csv).

The historical checksum record includes a former QC image that has been removed during cleanup; numerical artifacts remain unchanged. Regenerated checksum files contain only numerical/data metadata artifacts.
