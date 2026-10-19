# Source provenance

## Original SMAE denoising implementation

The source is [pengjuRen99/SpectraMAENet](https://github.com/pengjuRen99/SpectraMAENet), associated with Ren, Zhou and Li, *A Self-supervised Learning Method for Raman Spectroscopy based on Masked Autoencoders*, Expert Systems with Applications (2025), 128576.

The seven local core files previously in `denoising/SMAE_denoising/` were compared with the corresponding upstream directory at commit [`7939f2e6d3e6e975b687e4f07e56b9ce51cae5e3`](https://github.com/pengjuRen99/SpectraMAENet/tree/7939f2e6d3e6e975b687e4f07e56b9ce51cae5e3/denoising/SMAE_denoising). This is a content-comparison reference, not evidence of the original local download date or revision.

`ViT_1D.py`, `DenoisingData.py` and `Engine.py` matched after line-ending normalization. The model and three entry scripts matched after normalizing import prefixes and trailing whitespace. This comparison did not establish that the exploratory notebooks were unmodified.

These seven files are retained across `models/upstream/`, `utils/upstream/`, `training/upstream/` and `inference/upstream/`. Package imports were rewritten during reorganization. Publication cleanup also replaced the author's absolute data path in three commented-out argument examples with a repository-relative example. Their numerical implementation and historical control flow were not repaired or merged with the adapted branch. The adapted legacy classification entry points now default to `othermethods/smae/datasets/bacteria_ID`, relative to the repository root, instead of the author's home directory.

The original sources contain API inconsistencies: the denoising model requires a `truespectra` argument, whereas some pretraining/evaluation calls omit it. They also retain original relative data paths and legacy NumPy aliases. Consequently, these entry scripts are provenance/reference sources, not validated drop-in commands for the adapted checkpoints. Use the unified adapted entry point for the documented current workflow.

Original authorship and source attribution are retained. No new third-party license grant is asserted; the archived source tree did not include a LICENSE file.

## Project-adapted implementation

`models/Spectra_MAE.py` is the locally adapted implementation, including the custom 1,600-point patch-size variants and the 2,048-point variant. It is not an unchanged copy of the author's default model. `utils/data.py`, `utils/Engine.py`, and the adapted training/inference entry points retain the project-specific behavior.

Architecture and state-dictionary names are unchanged. Model/core-helper changes are limited to import relocation. Runtime fixes expose portable paths, seeds and worker counts; save effective settings and full resumable checkpoints; connect training and evaluation through `last.pth`; and pass configured weight decay to AdamW, including after resume. Distributed training now reduces losses and shares a rank-zero stopping decision. These optimizer/distributed-runtime corrections affect future runs; original upstream sources and historical artifacts are unchanged. The evaluator removes plotting, supports CPU execution and both existing checkpoint formats, checks input/output paths and shapes, and disables gradients. Previous metric formulas and one-pass masking/reconstruction behavior are retained.

The model files and the upstream files can be compared with the full external snapshot; all data, weights and event-file hashes are recorded in [asset_manifest.json](asset_manifest.json). Historical checkpoints are retained without assigning a new performance ranking or identifying a new final benchmark result.
