# Fast2Peak: High-Fidelity Restoration Algorithm by Single-Pixel Masking towards Real-Time Spectral Measurements

## Abstract

Improving the temporal and spatial resolution of spectroscopy inevitably reduces the photons available to each measurement, pushing rapid measurements into photon-starved conditions where weak spectral features are obscured by stochastic background noise. Here, we present Fast2Peak (F2P), a self-supervised Transformer framework that transforms spectral restoration task into a context-prediction problem. Given that physically constrained spectral signals exhibit strong local continuity, whereas the stochastic noise fluctuate randomly and does not possess the continuity across neighboring spectral pixels, single-pixel masking enables F2P to restore the predictable signal structure while suppressing unpredictable fluctuations even without clean reference spectra. Unlike conventional offline spectral restoration, F2P achieves high-fidelity restoration within milliseconds, enabling real-time “acquire-and-process” operation. We demonstrate this capability in time-resolved electrochemical surface-enhanced Raman spectroscopy and hyperspectral Raman imaging, where spectral restoration is performed during data acquisition. In addition, F2P has demonstrated its robust generalization across different instruments and spectral modalities, including Raman, nuclear magnetic resonance, X-ray diffraction and Fourier-transform infrared spectroscopy.

## Framework of F2P

![Overview of the Fast2Peak framework](docs/METHODS.svg)


## Datasets and Checkpoints

Datasets, model configurations and pretrained weights will be released through [moria5161 on Hugging Face](https://huggingface.co/moria5161).

**Release status: pending upload.** The resources are not yet available. Direct dataset/checkpoint links and placement instructions will be added when the release is ready. Keep each checkpoint together with its accompanying `config.yaml`, which the inference entry point uses to restore the model configuration.

## Getting Started

Run commands from the repository root. The current development environment uses Python 3.12, PyTorch 2.7.0 and torchvision 0.22.0 with CUDA 11.8. Direct dependency versions in `requirements.txt` match this environment.

```bash
conda create -n fast2peak python=3.12
conda activate fast2peak
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

For CPU-only installation, replace the PyTorch index URL with `https://download.pytorch.org/whl/cpu`. For other platforms, consult the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v270). CPU execution is not intended for reproducing GPU runtime measurements. No environment installation or training is performed automatically by this repository.

## Usage

Run the following commands from the repository root after preparing the data and, for evaluation, the checkpoint. Downloads are still pending as noted above. On a cluster, run training and inference on an allocated compute node, not the login node.

### Training

Prepare `datasets/train/train.h5` and `datasets/train/val.h5`. Each file must contain a `noisy` array of shape `[N, L]`; the main F2P configuration uses `L = 1600`. Self-supervised training and validation use noisy observations, not clean `gt` targets.

**Current implementation note:** the `F2P` branch in `utils/spec_loader.py` currently selects `BlockMaskDataset` with a default `--block_size 8`; its `MaskDataset` calls are commented out. The commands below document the current training interface, not a verified reproduction of the paper's single-pixel masking protocol. Restore the single-pixel dataset branch before reproducing that protocol; `--mask_ratio 0.05` alone does not change the masking strategy.

```bash
python main.py --train \
    --exp_name f2p_train \
    --model_name F2P \
    --train_data_dir datasets/train \
    --spectrum_length 1600 \
    --patch_size 16 --embed_dim 256 --depth 8 --heads 8 \
    --mask_ratio 0.05 --gamma 11 \
    --epochs 1000 --batch_size 512 \
    --lr 1e-4 --min_lr 1e-6 --warmup_epochs 10 \
    --weight_decay 1e-3 --seed 615
```

For single-node multi-GPU training, replace `python main.py --train` with `torchrun --standalone --nproc_per_node=4 main.py --train --ddp`, keeping the other arguments. Set the process count to the allocated GPU count. `--batch_size` is per process, so changing the GPU count changes the effective global batch size.

The best checkpoint is selected by validation masked MSE and saved as `experiments/f2p_train/checkpoints/best_checkpoint.pth`, alongside `config.yaml`. TensorBoard logs are written to `logs/f2p_train`.

### Evaluation

For quantitative evaluation, the test HDF5 file must contain matching `noisy` and `gt` arrays of shape `[N, L]`. Without `gt`, the evaluator saves denoised spectra but skips reference-based metrics.

```bash
python main.py --test \
    --test_model_path experiments/f2p_train/checkpoints/best_checkpoint.pth \
    --test_datapath datasets/test/Simu.h5 \
    --test_batch_size 1024
```

Replace the checkpoint path with the released pretrained checkpoint to evaluate without training. Its adjacent `config.yaml` is required: the evaluator restores the architecture from that file, including the legacy `Fast2PeakRPE` to `F2P` name mapping. Use a single process for evaluation; do not add `--ddp`.

For the example above, outputs are saved under `experiments/f2p_train/output/YYYY-MM-DD/`:

- `Simu.h5`: predictions under the `denoised` key.
- `Simu_eval.log`: mean SNR (dB), RMSE and SSS when `gt` is available.
- `Simu_metrics_details.csv`: per-spectrum metrics.

Self-supervised F2P inference includes the existing noisy-input-based amplitude correction. IR, XRD and NMR use their own benchmark evaluators and metric definitions linked below.

### Key Arguments

| Argument | Description |
|---|---|
| `--train` / `--test` | Train a model or run inference/evaluation. |
| `--exp_name` | Training experiment name used for checkpoint and log directories. |
| `--train_data_dir` | Directory containing `train.h5` and `val.h5`. |
| `--mask_ratio` | Fraction of spectral points selected for masking; explicitly set to `0.05` above rather than the CLI default `0.25`. |
| `--gamma` | Neighborhood parameter used to replace selected points. |
| `--test_model_path` | Checkpoint path; keep its `config.yaml` in the same directory. |
| `--test_datapath` | Test file, or a directory of test files. |
| `--test_batch_size` | Inference batch size; reduce it if GPU memory is limited. |

### Training Logs

```bash
tensorboard --logdir logs/f2p_train
```

## Benchmark Reproduction

Data and weights from the pending release are required before running the benchmarks. Code and configurations are organized by modality:

- [IR](otherbench/IR/)
- [XRD](otherbench/XRD/)
- [NMR](otherbench/NMR/)

See [benchmark results](otherbench/RESULTS.md) for the finalized tables.

Local scheduler settings are not required by the Python entry points. The optional Peak2Peak Slurm helper should be submitted from the repository root: `sbatch othermethods/peak2peak/p2p.sh`. Select a node or partition through `sbatch` options if required by your cluster.

## Citation

The manuscript citation will be added when available.

## License

Copyright 2026 Fast2Peak contributors. Original Fast2Peak code and documentation are licensed under the [Apache License 2.0](LICENSE).

Third-party code retains its original ownership and terms. The IR utilities in `otherbench/IR/utils/upstream/` are GPL-3.0-or-later; the [GPL text](LICENSES/GPL-3.0.txt) and source notices are retained. See the [SMAE provenance](othermethods/smae/docs/provenance.md) and source attribution in the NMR model files. Baseline files with unresolved provenance or permissions are excluded from the Apache grant pending review; attribution alone does not grant redistribution rights.

Datasets, model weights and third-party fonts are not covered by this code license and require separate terms.

## Contact
Welcome to contact us or raise issues if you have any questions. 
Email: moria5161@stu.xmu.edu.cn
