# SMAE baseline

This subproject separates the project-adapted SMAE implementation from the original author's denoising sources. It preserves historical local checkpoints and TensorBoard logs without treating them as a finalized benchmark table.

## Structure

```text
smae/
├── configs/       # YAML runtime defaults and original environment export
├── models/        # Adapted architectures; upstream/ contains original denoising models
├── datasets/      # Local data (Git-ignored)
├── training/      # Adapted pretraining, upstream denoising and legacy classification
├── inference/     # Adapted numerical evaluation and historical upstream evaluation
├── utils/         # Data, augmentation, metrics and training helpers
├── scripts/       # Local Shell/SLURM helpers (Git-ignored)
├── docs/          # Provenance, workflow and preservation manifest
├── outputs/       # All historical weights, TensorBoard logs and new results (Git-ignored)
├── main.py        # Unified entry point for adapted workflows
└── requirements.txt
```

## Run

From the repository root, activate the existing `fast2peak` environment:

```bash
conda activate fast2peak
python othermethods/smae/main.py train --help
python othermethods/smae/main.py evaluate --help
python othermethods/smae/main.py run --device cuda:0
# Or run the two stages separately:
python othermethods/smae/main.py train --device cuda:0
python othermethods/smae/main.py evaluate --device cuda:0
```

The matched defaults in [configs/benchmark.yaml](configs/benchmark.yaml) train the adapted patch-4/mask-0.1 model and evaluate its `last.pth` checkpoint. `run` executes both stages and passes the newly produced checkpoint directly to evaluation. Model name and masking ratio are read from checkpoint metadata, including CLI training overrides. New training writes to `outputs/retrained/`, not the retained historical checkpoint directories. Use a new `--output-dir` or `--resume` if that training directory already contains `last.pth`.

Training and test HDF5 inputs are shared with F2P through `../../datasets/train/train.h5` and `../../datasets/test/Simu.h5`; they are not duplicated here. SMAE maintains its own configurations, weights, logs and evaluation outputs.

Use `--config PATH` to supply another YAML. YAML asset paths resolve relative to this subproject; explicit CLI paths resolve relative to the current directory. Use `--print-config` to inspect resolved options without starting training. Each executed stage saves `config.yaml`, an immutable timestamped configuration snapshot and a numerical summary in its output directory. Training checkpoints include model/settings, epoch, optimizer and scheduler state. The configured `wd` is passed to AdamW, including after resume.

Run `torchrun --standalone --nproc_per_node=4 othermethods/smae/main.py train-ddp` for CUDA distributed training. Local SLURM scripts preserve the project's `#!/bin/bash`, node declarations, stdout/stderr settings and four-GPU `torchrun` training style. Two-process CPU/Gloo training and resume were verified; four-GPU CUDA/NCCL execution has not been run during this work.

## Preserved local assets

All existing data and checkpoint files are retained. TensorBoard event files keep their names and run-directory hierarchy under `outputs/logs/`; none were discarded or regenerated.

```bash
tensorboard --logdir othermethods/smae/outputs/logs
```

Ignored assets must be supplied separately for a fresh checkout. No scientific benchmark was retrained or replaced during this work; temporary small synthetic datasets were used for runtime verification only. Numerical evaluation retains the historical SNR/RMSE/SSS definitions; it does not generate figures or notebooks.

See [workflow](docs/workflow.md), [source provenance](docs/provenance.md), [local data](docs/data_README.md) and [asset checksums](docs/asset_manifest.json).

## Reference

Ren P, Zhou R, Li Y. *A Self-supervised Learning Method for Raman Spectroscopy based on Masked Autoencoders*. Expert Systems with Applications, 2025, 128576. Original repository: [pengjuRen99/SpectraMAENet](https://github.com/pengjuRen99/SpectraMAENet).
