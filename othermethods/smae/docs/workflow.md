# Workflow and file migration

## Adapted runtime

The public entry point is `main.py`; it imports fully qualified lowercase package names so local `models` and `utils` do not shadow the shared repository packages. Run from the repository root or supply an absolute path to `main.py`.

| Previous location | Current location |
|---|---|
| `model/` | `models/` |
| `main_unsupervisePretrain.py` | `training/pretrain.py` |
| `main_unsupervisePretrain_ddp.py` | `training/pretrain_ddp.py` |
| `smae_test.py` | `inference/evaluate.py` |
| `main_finetune.py` | `training/legacy/classification_finetune.py` |
| `main_supervisePretrian.py` | `training/legacy/classification_supervised.py` |
| `denoising/SMAE_denoising/model/` | `models/upstream/` |
| `denoising/SMAE_denoising/utils/` | `utils/upstream/` |
| `denoising/SMAE_denoising/main_denoisingPretrain.py` | `training/upstream/pretrain.py` |
| `denoising/SMAE_denoising/main_denoisingFunetune.py` | `training/upstream/finetune.py` |
| `denoising/SMAE_denoising/main_denoisingTest.py` | `inference/upstream/evaluate.py` |
| `experiments/` | `outputs/experiments/` |
| `Pretrain_weight_moria/` | `outputs/pretrained/` |
| `logs/` | `outputs/logs/` |

The root-level historical `main_denoisingPretrain.py` and `main_denoisingTest.py` are archived rather than presented as active entry points: they combine older denoising entry logic with the independently modified root model/helpers. The separately retained upstream branch documents the original denoising workflow.

## Training

```bash
python othermethods/smae/main.py train --data-path /path/to/train.h5 --device cuda:0
torchrun --standalone --nproc_per_node=4 othermethods/smae/main.py train-ddp \
  --data-path /path/to/train.h5 --resume /path/to/checkpoint.pth
```

Training reads the HDF5 `noisy` matrix. Existing augmentations, masked reconstruction, AdamW learning-rate/beta settings, scheduler and training-loss checkpoint criterion are retained. The previous omission of `weight_decay=args.wd` is corrected for both single-device and distributed training; resumed optimizer groups also use the requested `wd`. This changes future training relative to the former implicit AdamW default, not any stored historical checkpoint or result. No new validation split is introduced. Distributed loss is reduced across ranks, only rank zero saves checkpoints/configurations, and stopping decisions are broadcast to all ranks.

Seeds, worker counts, output paths and TensorBoard paths are explicit options. New runs write event files to the configured `log_dir`. All historical TensorBoard files remain under `outputs/logs/experiments/` with their original names and subdirectories.

Each training epoch updates a full `last.pth` checkpoint containing model weights, effective arguments, optimizer/scheduler state, epoch and early-stopping state. Historical-style `epoch_*.pth` files are also retained. `--resume /path/to/last.pth` continues a run; `--epochs` is the final total epoch number, not the number of additional epochs. Distributed resume writes its actual suffixed TensorBoard directory into the configuration snapshot.

`save_path/config.yaml` records effective absolute paths and CLI-overridden values. Immutable `config_<timestamp>_<pid>.yaml` snapshots preserve previous invocation settings; `summary.json` records the actual checkpoint, loss, epoch count, log path and optimizer weight decay. `--print-config` displays parsed settings without starting a run.

The classification scripts in `training/legacy/` preserve earlier workflows and defaults; they are not part of numerical denoising evaluation or the unified default commands.

## End-to-end task

```bash
python othermethods/smae/main.py run --device cuda:0
```

`run` performs single-device training and then evaluates that invocation's `last.pth`. It never falls back to the historical patch-1 checkpoint. Model and mask ratio are inferred from saved training metadata. Training CLI options also work with `run`; use `--input-path`, `--evaluation-dir` and `--evaluation-batch-size` to override the evaluation input, destination and batch size. `pipeline_config.yaml` and `pipeline_summary.json` record the complete task in the training directory.

For the original four-GPU SLURM workflow, activate `fast2peak` and submit training followed by a dependent evaluation job:

```bash
mkdir -p othermethods/smae/outputs/logs/slurm
train_job=$(sbatch --parsable othermethods/smae/scripts/train.sh)
sbatch --dependency=afterok:${train_job%%;*} othermethods/smae/scripts/evaluate.sh
```

The local training script retains node7, four GPUs and `torchrun --nproc_per_node=4`; evaluation retains node5 and one GPU. Both explicitly use `configs/benchmark.yaml`. SLURM stdout/stderr are directed to `outputs/logs/slurm/`, rather than the submission directory. Cluster-specific overrides may be passed to `sbatch`. These scripts remain Git-ignored.

## Evaluation

```bash
python othermethods/smae/main.py evaluate \
  --input-path /path/to/test.h5 \
  --checkpoint /path/to/checkpoint.pth \
  --arch spectraMAE_base_ss1600_ps1 \
  --mask-ratio 0.5 --batch-size 24 --device cuda:0 \
  --output-dir /path/to/results
```

The evaluator accepts HDF5 (`noisy`, optional matching `gt` and `wavenumbers`), one-column intensity TXT, two-column axis/intensity TXT, or a directory of HDF5/TXT files. Input lengths must match the architecture exactly. As in the previous evaluator, no implicit interpolation or input normalization is introduced. Predictions are the complete unpatchified output from one stochastic masking pass, not an ensemble or a visible-input/prediction blend. Seed, batch size, device and masking ratio affect reproducibility.

Bare state dictionaries and dictionaries containing `model_state_dict` are supported, including `module.`-prefixed keys. New checkpoints infer architecture and mask ratio automatically; historical bare checkpoints require explicit `--arch` and `--mask-ratio`, as in the example above. Strict loading is used. CPU inference is available via `--device cpu`; inference disables gradient recording. Input spectra and checkpoints cannot be used as output targets, and duplicate output basenames are rejected.

Each input produces a prediction file. Inputs with `gt` additionally produce per-spectrum `*_metrics_details.csv` and an aggregate `*_eval.log`. The metric definitions are unchanged: SNR in dB, RMSE, and `SSS = (1 - SAD) * PCC`, where SAD is in radians. This SSS is a historical project-specific metric, not Peak F1. Zero-norm or constant reference spectra retain the original metric's undefined-value behavior.

Evaluation saves effective settings in its own `config.yaml` and immutable configuration snapshot. `summary.json` includes the consumed checkpoint's SHA-256 digest, so predictions remain traceable even if a later resumed run updates `last.pth`.

## Archived exploratory work

All notebooks, visualization-only content, `Unsupervised_Mapping/`, cycleGAN and empty peripheral examples are preserved in the external pre-reorganization snapshot, not the active SMAE workflow. Existing user deletions of `model/tmp2.py` and `model/tmp2.sh` were respected. No original data, weights or TensorBoard events were deleted.
