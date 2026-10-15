# Fast2Peak: High-Fidelity Restoration Algorithm by Single-Pixel Masking towards Real-Time Spectral Measurements

[![](https://img.shields.io/badge/huggingface-ckpt--of--vib2mol-dd9029)](https://huggingface.co/xinyulu/vib2mol)
[![](https://img.shields.io/badge/huggingface-vibench-dd9029)](https://huggingface.co/datasets/xinyulu/vibench)

## Abstract
Improving the temporal and spatial resolution of spectroscopy inevitably reduces the photons available to each measurement, pushing rapid measurements into photon-starved conditions where weak spectral features are obscured by stochastic background noise. Here, we present Fast2Peak (F2P), a self-supervised Transformer framework that transforms spectral restoration task into a context-prediction problem. Given that physically constrained spectral signals exhibit strong local continuity, whereas the stochastic noise fluctuate randomly and does not possess the continuity across neighboring spectral pixels, single-pixel masking enables F2P to restore the predictable signal structure while suppressing unpredictable fluctuations even without clean reference spectra. F2P achieves high-fidelity restoration within milliseconds, making real-time “acquire-and-process” operation possible.

# Framework of F2P
![figure1](./xxxxx)


## Datasets and Checkpoints


## Getting Started


## Evaluation and Reproducing


## Citation


## Contact
Welcome to contact us or raise issues if you have any questions. 
Email: moria5161@stu.xmu.edu.cn