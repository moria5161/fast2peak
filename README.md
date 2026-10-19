# Fast2Peak: High-Fidelity Restoration Algorithm by Single-Pixel Masking towards Real-Time Spectral Measurements

## Abstract
Improving the temporal and spatial resolution of spectroscopy inevitably reduces the photons available to each measurement, pushing rapid measurements into photon-starved conditions where weak spectral features are obscured by stochastic background noise. Here, we present Fast2Peak (F2P), a self-supervised Transformer framework that transforms spectral restoration task into a context-prediction problem. Given that physically constrained spectral signals exhibit strong local continuity, whereas the stochastic noise fluctuate randomly and does not possess the continuity across neighboring spectral pixels, single-pixel masking enables F2P to restore the predictable signal structure while suppressing unpredictable fluctuations even without clean reference spectra. F2P achieves high-fidelity restoration within milliseconds, making real-time “acquire-and-process” operation possible.

## Framework of F2P

![Overview of the Fast2Peak framework](docs/METHODS.svg)


## Datasets and Checkpoints

Datasets, model configurations and pretrained weights will be released through [Moria5161 on Hugging Face](https://huggingface.co/datasets/Moria5161).

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

## Evaluation and Reproducing

Data and weights from the pending release are required before running the benchmarks. Each subproject documents its preparation, training/inference, evaluation protocol and numerical results:

- [IR benchmark](otherbench/IR/README.md)
- [XRD benchmark](otherbench/XRD/README.md)
- [NMR benchmark](otherbench/NMR/README.md)

Local scheduler settings are not required by the Python entry points. The optional Peak2Peak Slurm helper should be submitted from the repository root: `sbatch othermethods/peak2peak/p2p.sh`. Select a node or partition through `sbatch` options if required by your cluster.

## Citation

The manuscript citation will be added when available.

## License

Original Fast2Peak code and documentation are licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party components, exclusions and outstanding permission checks. Third-party code retains its own terms; it is not relicensed by the root license. Datasets, model weights and third-party fonts are not covered by this code license and require separate terms.

## Contact
Welcome to contact us or raise issues if you have any questions. 
Email: moria5161@stu.xmu.edu.cn
