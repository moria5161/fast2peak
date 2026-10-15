# Final IR workflow

## Data

`datasets/atr_spectra.npy` stores 174 ATR references: `(1863, 175)`, float64, with wavenumber in column 0 and one intensity spectrum per remaining column. Column labels, alignment details and original provenance are in [data_README.md](data_README.md). The original CSVs are archived outside the repository.

`datasets/paired/` contains the fixed 110-reference selection and the synthetic pairs. Its public historical metadata is copied to [dataset_manifest.json](dataset_manifest.json) and [reference_split.csv](reference_split.csv); historic absolute paths in those metadata records document the original run.

| File | Content |
|---|---|
| `selected_clean.npy` | 110 frozen normalized references, `(110, 1024)` |
| `wavenumbers.npy` | Frozen 1,024-point axis |
| `train_noisy.npy`, `train_target.npy` | 45,000 candidate training/validation pairs |
| `test_noisy.npy`, `test_target.npy` | 4,000 held-out pairs |
| `*_reference_id.npy` | Local reference IDs in each split |
| `*_sigma.npy` | Per-spectrum noise sigma |
| `manifest.json`, `split.csv`, `checksums.sha256` | Generation record, source mapping and checksums |

Run `python otherbench/IR/main.py prepare` from the repository root. Each reference is normalized before adding zero-mean Gaussian noise with independently sampled sigma from Uniform[0.05, 0.10]. No baseline, clipping or post-noise normalization is applied. The first 90 frozen reference identities contribute 500 realizations each; the last 20 identities contribute 200 each. Seeds are 0 and 2048.

The default preparation uses frozen references rather than repeating order-dependent PCA/KMeans selection. Optional `prepare --reselect-references --spectra-dir PATH` requires externally supplied original CSVs. Repacking those external CSVs is optional: `python otherbench/IR/main.py pack --input-dir PATH --output otherbench/IR/datasets/atr_spectra.npy`.

## Algorithms and training

The dense autoencoder in `models/autoencoder.py` has architecture `1024 -> Dense(128, ReLU) -> Dense(1024, ReLU)`, with the original Glorot initialization. Training uses Adam, MSE, batch size 32, learning rate 0.001 and seed 42 for 100 epochs. Reference-level validation reserves IDs `[1, 3, 6, 48, 61, 70, 79, 83, 86]`, yielding 40,500 training and 4,500 validation pairs. The saved run selected epoch 7 by minimum validation MSE. The final test is evaluated after restoring the selected model. Numerical retraining differences can occur across hardware/software versions.

SavGol uses window 15/order 1. F2P uses the shared repository Fast2PeakRPE architecture via `models/f2p.py`, with a pretrained checkpoint and adjacent YAML. The adapter handles its current shared name `F2P` while preserving the original checkpoint metadata and loading weights strictly. The saved benchmark checkpoint was epoch 856. Its standard inference aligns 1,024 points to 1,600, normalizes, runs the model, restores the original length/scale and applies amplitude correction using the noisy input and prediction. No IR fine-tuning is performed. RAW is the unchanged noisy input.

These noise and validation protocols adapt the original section 3.1 reference selection; they are not an exact reproduction of the original paper's noise/validation setup.

## Outputs and public metrics

| Directory | Files |
|---|---|
| `outputs/autoencoder/` | `model.pt`, `test_reconstructed.npy`, `history.csv`, `test_metrics.csv`, `summary.json` |
| `outputs/evaluation/` | `f2p_reconstructed_raw.npy`, `f2p_reconstructed_corrected.npy`, `comparison_metrics.csv`, `summary.json`, `leaderboard.csv` |
| `docs/` | Public `leaderboard.csv`, `evaluation_summary.json`, data description and provenance metadata |

Run `python otherbench/IR/main.py export` after evaluation to refresh the public aggregate metric snapshots. No plotting functions, notebooks or image outputs are retained.

## Local Slurm helpers

From the repository root:

```bash
conda activate fast2peak
mkdir -p otherbench/IR/outputs/logs
train_job=$(sbatch --parsable otherbench/IR/scripts/train.sh)
sbatch --dependency=afterok:${train_job%%;*} otherbench/IR/scripts/evaluate.sh
```

Scripts inherit the activated environment and request one GPU. Pass cluster-specific node/partition options to `sbatch` as needed; `PYTHON` overrides the interpreter. Scripts are intentionally Git-ignored; the public Python commands above provide the portable workflow.
