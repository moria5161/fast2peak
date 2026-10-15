# Experimental PXRD denoising benchmark

This subproject compares RAW, SavGol, Wavelet and pretrained F2P on the final experimental PXRD benchmark. It uses zero-shot inference and classical denoising, with numerical metric outputs only.

## Structure

```text
XRD/
├── configs/       # YAML configuration
├── models/        # Classical denoisers and shared pretrained F2P adapter
├── datasets/      # Raw XY files and prepared paired data (Git-ignored)
├── training/      # Reserved; this benchmark does not train or fine-tune
├── inference/     # F2P inference, evaluation and numerical export
├── utils/         # QC, noise generation, normalization, metrics and reporting
├── scripts/       # Local Shell/Slurm helpers (Git-ignored)
├── docs/          # Public data provenance, metrics and final Table S6
├── outputs/       # Predictions and detailed experiment metrics (Git-ignored)
├── main.py        # Unified entry point
└── requirements.txt
```

## Run

From the repository root:

```bash
conda activate fast2peak
python otherbench/XRD/main.py prepare
python otherbench/XRD/main.py evaluate --device cuda
python otherbench/XRD/main.py export
```

Parameters are in [configs/benchmark.yaml](configs/benchmark.yaml). YAML paths resolve relative to `otherbench/XRD`, independent of the working directory. Select another configuration with `--config PATH`; explicit CLI options override defaults. Raw data and model weights must be supplied separately for a checkout, since datasets and outputs are Git-ignored.

[Data provenance](docs/data_README.md) explains the experimental references and synthetic noise. [Workflow instructions](docs/workflow.md) describe the algorithms, output artifacts and Slurm execution.

## Final manuscript table

**Table S6. Quantitative comparison on the experimental PXRD denoising benchmark.**

| Methods | Pearson ↑ | RMSE ↓ | SNR (dB) ↑ | Peak F1 ↑ |
|---|---:|---:|---:|---:|
| RAW | 0.8766 | 0.0455 | 9.11 | 0.264 |
| SavGol | 0.9375 | 0.0290 | 12.08 | 0.521 |
| Wavelet | 0.9472 | 0.0274 | 12.75 | 0.469 |
| F2P | 0.9592 | 0.0246 | 13.69 | 0.563 |

The supplied final table is preserved in [table_s6.csv](docs/table_s6.csv) and [table_s6.md](docs/table_s6.md). Computed metrics are retained separately in [leaderboard.csv](docs/leaderboard.csv), [leaderboard_macro_average.csv](docs/leaderboard_macro_average.csv) and [evaluation_summary.json](docs/evaluation_summary.json). The supplied SavGol Pearson (0.9375) differs from the stored calculated value (0.93744151, rounded 0.9374); the numerical source artifacts remain unchanged.
