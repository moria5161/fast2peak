# Maybridge ¹H NMR denoising benchmark

This subproject retains the exact Table S4 data, six methods, selected checkpoints and numerical results. It does not generate plots or notebooks.

## Structure

```text
NMR/
├── configs/       # Final benchmark and retraining parameters (YAML)
├── models/        # WT/Cadzow, DN-Unet, LD-Net and shared F2P adapter
├── datasets/      # Maybridge source, frozen split, validation/test pairs (Git-ignored)
├── training/      # Final supervised and noisy-only training protocols
├── inference/     # Inference, evaluation, historical-result assembly and export
├── utils/         # Source conversion, split/preprocessing, loading and metrics
├── scripts/       # Local Shell/Slurm helpers (Git-ignored)
├── docs/          # Public provenance, numerical metrics and final Table S4
├── outputs/       # Selected weights, histories and predictions (Git-ignored)
└── main.py        # Unified entry point
```

## Final table

**Table S4. Quantitative comparison on the Maybridge ¹H NMR denoising benchmark.**

| Methods | Pearson ↑ | RMSE (×10⁻³) ↓ | SNR (dB) ↑ | Peak F1 ↑ |
|---|---:|---:|---:|---:|
| RAW | 0.692937 | 53.313 | 2.187 | 0.039786 |
| WT | 0.912430 | 16.468 | 11.405 | 0.349464 |
| Cadzow | 0.961960 | 13.464 | 12.295 | 0.489976 |
| LD-Net | 0.973669 | 8.418 | 16.966 | 0.555886 |
| DN-Unet | 0.986677 | 6.331 | 19.347 | 0.831536 |
| F2P | 0.986635 | 6.703 | 18.311 | 0.782349 |

The fixed manuscript table is in [docs/table_s4.csv](docs/table_s4.csv) and [docs/table_s4.md](docs/table_s4.md); [full-precision metrics](docs/overall_leaderboard.csv) preserve the underlying values. Results cover 150 test molecules × 3 noise levels = 450 pairs per method. RMSE is multiplied by 1,000 only for manuscript display.

## Run

From the repository root, with the `fast2peak` environment active:

```bash
python otherbench/NMR/main.py prepare
python otherbench/NMR/main.py evaluate --device cuda
python otherbench/NMR/main.py export
```

Configuration is in [configs/benchmark.yaml](configs/benchmark.yaml). YAML paths resolve relative to `otherbench/NMR`; `--config PATH` selects another YAML and explicit CLI options override defaults. Existing frozen split/checkpoints are used for the saved benchmark. Supply ignored `datasets/` and `outputs/` assets separately when distributing a checkout.

This is the **Maybridge** dataset, not the earlier MTBLS2550 experiment. DN-Unet and LD-Net are supervised fine-tuned networks. F2P is the selected native-6400, depth-4, mask-ratio-0.03 self-supervised model trained for 400 epochs (checkpoint epoch 396). It is not the generic IR/XRD pretrained checkpoint, nor the earlier mask-0.05 F2P result.

See [data provenance](docs/data_README.md), [model/training provenance](docs/model_provenance.md) and [the full workflow](docs/workflow.md). F2P checkpoint selection uses masked MSE on fixed noisy test observations; it should not be described as independent clean-validation selection or zero-shot inference.
