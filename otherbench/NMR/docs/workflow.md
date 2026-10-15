# Reproducing the final Maybridge benchmark

Run from the repository root with the `fast2peak` environment activated. YAML paths are relative to `otherbench/NMR`, while explicit CLI paths follow the current working directory.

## Data

The retained frozen split and paired data can be used directly. To reconstruct the initial data and split from the TXT/SDF:

```bash
python otherbench/NMR/main.py source
python otherbench/NMR/main.py split
python otherbench/NMR/main.py prepare
```

`prepare` rebuilds both the 6,400-point test benchmark and the separate 1,600-point supervised validation benchmark. Pass `--output PATH --validation-output PATH` to write a separate diagnostic copy. The existing source files, frozen split and test references are documented in [data_README.md](data_README.md).

## Training

The three retained training modules implement the protocols used for the selected final models:

```bash
python otherbench/NMR/main.py train-f2p
python otherbench/NMR/main.py train-dn-unet
python otherbench/NMR/main.py train-ld-net
```

Use GPU compute nodes for training. Defaults are in `configs/benchmark.yaml`; explicit options can override them. These commands write to `outputs/retrained/` and do not overwrite the fixed Table S4 checkpoints in `outputs/training/`. To evaluate a newly trained model, point the appropriate model checkpoint field in a separate YAML at its result.

F2P has a noisy-test-observation-based checkpoint selection protocol. DN-Unet and LD-Net use supervised independent-validation selection followed by train+validation refits. See [model_provenance.md](model_provenance.md) for the distinction, selected epochs, losses and original author resources.

## Evaluation and export

```bash
python otherbench/NMR/main.py evaluate --device cuda
python otherbench/NMR/main.py export
```

The final benchmark uses three Gaussian conditions with 150 test references each. WT/Cadzow and both fine-tuned author networks are scored on the same 6,400-point input pairs as the selected F2P. Peak metrics use prominence 0.03, spacing 8 points and matching tolerance 0.03 ppm. Additional spectral, derivative, peak-width and integrated-intensity metrics are retained unchanged.

For a small diagnostic, use `evaluate --max-samples 1 --device cpu --output-dir PATH`. Add `--skip-f2p --skip-author-networks` for classical methods only. Full-model CPU and CUDA-autocast results can differ slightly; the preserved numerical Table S4 records the original inference results.

| Location | Artifacts |
|---|---|
| `outputs/training/f2p/` | Selected epoch-396 checkpoint, original configuration/history, fixed noisy selection inputs |
| `outputs/training/dn_unet/` | Selected Stage-A checkpoint, final epoch-49 refit, configuration/history |
| `outputs/training/ld_net/` | Selected Stage-A checkpoint, final epoch-99 refit, configuration/history |
| `outputs/pretrained/` | Original converted DN-Unet and pickled LD-Net initialization weights |
| `outputs/evaluation/` | Unified `denoised.h5`, `metrics.csv`, `leaderboard.csv`, `overall_leaderboard.csv`, `summary.json` |
| `docs/` | Fixed manuscript Table S4, computed full-precision metrics, data/model provenance |

`export` refreshes the computed metric copies in `docs/` without overwriting `table_s4.csv` or `table_s4.md`. Table RMSE is `1000 * rmse_mean`; values in machine-computed metric files remain on their original scale.

## Historical final-result assembly

The original `evaluation_best_f2p` CSV was not updated after the final mask-ratio search. The six-method results were therefore assembled from:

- RAW, WT, Cadzow, LD-Net and DN-Unet: `outputs_data2_native6400/evaluation_all`.
- F2P: `evaluation_f2p6400_depth4_epochs400_wd0_mask003_vs_mask005_native6400`, method `F2P-depth4-wd0-mask0.03`.

Those full original experiments remain in the external cleanup backup. `outputs/evaluation/summary.json` records the exact source artifact hashes. The `main.py assemble` command can reconstruct the final numerical and HDF5 results from explicit source inputs, checking that their GT, axes and noisy observations agree. It excludes all earlier F2P rows, retains the original per-spectrum metrics, and recomputes the correct per-condition and overall aggregates.

## Local Slurm

From the repository root:

```bash
conda activate fast2peak
mkdir -p otherbench/NMR/outputs/logs
sbatch otherbench/NMR/scripts/evaluate.sh
# Optional retraining:
sbatch otherbench/NMR/scripts/train.sh f2p
sbatch otherbench/NMR/scripts/train.sh dn-unet
sbatch otherbench/NMR/scripts/train.sh ld-net
```

Helpers inherit the active environment. Override `PYTHON` or `CONFIG` if needed and pass cluster-specific node/partition settings to `sbatch`. `datasets/`, `scripts/` and `outputs/` are intentionally Git-ignored; data and weight assets must be supplied separately for another checkout. No plotting code or images are retained.
