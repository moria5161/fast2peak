# Final XRD benchmark workflow

## Preparation

From the repository root, activate `fast2peak` and run `python otherbench/XRD/main.py prepare`. Raw XY files in `datasets/raw/` are cropped, quality-filtered and normalized, then corrupted with the seven Gaussian/Poisson/mixed noise conditions. Preparation writes only arrays, HDF5, CSV and JSON/checksum metadata to `datasets/benchmark/`.

Use `prepare --output-dir PATH` for a separate regeneration. The default data preparation expects the complete 815-pattern collection. See [data_README.md](data_README.md) for data source, QC, reference semantics and noise formulas.

## Algorithms and evaluation

Run `python otherbench/XRD/main.py evaluate --device cuda`. The configuration fixes SavGol window 15/order 3; Coiflet-4 wavelet decomposition level 4, symmetric boundaries, MAD-based universal soft threshold; and peak matching prominence 0.05, minimum spacing 3 points, tolerance 0.06°.

F2P uses the shared pretrained relative-attention architecture and checkpoint via `models/f2p.py`, without XRD training or fine-tuning. Supply the checkpoint and its adjacent `config.yaml`. The loader handles the shared architecture rename from `Fast2PeakRPE` to `F2P` and loads weights strictly, retaining historic model metadata. Input length is already 1,600 points; inference min-max normalizes each input, restores its scale and applies the standard noisy-input-based amplitude correction.

The preserved final benchmark includes Anscombe + Coiflet-4 as an additional comparator for pure Poisson conditions only. The final manuscript four-method macro-average includes RAW, SavGol, Wavelet and F2P, excluding Anscombe. The unfinished mixed-Anscombe experiment was archived separately.

Metrics include Pearson, RMSE, MAE, SNR/ΔSNR, spectral angle, Peak F1, matched peak position/intensity errors and integrated-intensity relative error. All historical metric implementations and matching rules are retained.

## Outputs

| Directory/file | Purpose |
|---|---|
| `outputs/evaluation/denoised.h5` | Reference, axis and all per-condition method outputs |
| `outputs/evaluation/metrics.csv` | Per-spectrum detailed metrics |
| `outputs/evaluation/leaderboard.csv` | Per-condition aggregate metrics |
| `outputs/evaluation/summary.json` | Evaluation metadata and aggregate metrics |
| `outputs/evaluation/leaderboard_macro_average.csv` | Four-method, equal-condition macro-average |
| `docs/table_s6.csv`, `docs/table_s6.md` | Final manuscript table supplied by the author |
| `docs/leaderboard*.csv`, `docs/evaluation_summary.json` | Public computed metric snapshots |

Run `python otherbench/XRD/main.py export` after evaluation to refresh computed metric/QC snapshots. It leaves the supplied Table S6 files untouched. The supplied SavGol Pearson 0.9375 differs from the calculated macro-average 0.93744151; both are retained explicitly. No plots, notebooks or images are generated.

For a small diagnostic, use `evaluate --max-samples 2 --skip-f2p --device cpu --output-dir PATH` to run classical baselines, or omit `--skip-f2p` to include pretrained inference. Use a separate output directory for diagnostics.

## Local Slurm

Submit from the repository root after activating the environment and creating logs:

```bash
conda activate fast2peak
mkdir -p otherbench/XRD/outputs/logs
sbatch otherbench/XRD/scripts/run_benchmark.sh
# Or evaluate already prepared data:
sbatch otherbench/XRD/scripts/evaluate.sh
```

Scripts are local and Git-ignored. They inherit the active Python environment; `PYTHON` and `CONFIG` may override interpreter/configuration. Pass cluster-specific node/partition options to `sbatch`. There is no current training stage.
