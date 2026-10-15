# Models and training provenance

## WT and Cadzow

WT is Symlet-8, level-4, symmetric-boundary wavelet shrinkage with a MAD-based universal soft threshold. Cadzow uses two low-rank Hankel projection iterations, width 128 and automatic rank capped at 64, on a pseudo-FID reconstructed from the processed real spectrum. It does not use original acquired complex FID data.

## DN-Unet

The PyTorch implementation maps the official authors' TensorLayer architecture and weights layer-for-layer. The original download page is [Lin Group](https://lingroup.xmu.edu.cn/info/1150/1117.htm); the paper is [10.1021/acs.analchem.0c03087](https://doi.org/10.1021/acs.analchem.0c03087). The converted initial checkpoint is `outputs/pretrained/dn_unet.pth`.

The Table S4 checkpoint is the supervised fine-tuned `outputs/training/dn_unet/final_train_plus_val_checkpoint.pth`. Stage A selects an epoch using 701 training and 150 independent validation references; Stage B restarts from the official checkpoint and refits 851 train+validation references for the selected 49 epochs. Batch-normalization statistics are frozen, the author relative-L2 loss is retained, and the held-out 150 test references do not contribute to this training/selection. Historical Stage-A weights, configuration and histories are retained alongside final weights.

## LD-Net

The LD-3 `ModeltransCompare` architecture is from [HaolinZhan/LD-Net-for-NMR-spectroscopy-denoising](https://github.com/HaolinZhan/LD-Net-for-NMR-spectroscopy-denoising), commit `38130d612b8cb02e61c48afbad5cf6c30db9fabf`; paper [10.3866/PKU.WHXB202310045](https://doi.org/10.3866/PKU.WHXB202310045). Only its required model and two block classes are retained in `models/ld_net.py`, with unused plotting imports removed. The official pickled initial checkpoint is retained in `outputs/pretrained/ld_net.pth`; a compatibility loader handles its original module name.

The Table S4 model is supervised fine-tuned with the author batch-global relative-L2 loss, independent validation selection, then a fresh train+validation refit for 99 epochs. Final and Stage-A checkpoints, configurations and histories are in `outputs/training/ld_net/`. Both author networks infer at 8,192 points with positive-max normalization and linear adaptation from/back to the 6,400-point test grid.

The downloaded author repositories did not include a declared software license in the retained records. Their original attribution is preserved; no new third-party license is asserted here. Full original downloaded resources remain in the external cleanup backup.

## F2P

The final architecture is relative-attention F2P, length 6,400, patch size 16, embedding width 256, depth 4, 8 heads. It is trained from scratch with noisy-only point-mask MSE, mask ratio 0.03, gamma 11, AdamW learning rate 0.0001, zero weight decay and 400 epochs. The selected checkpoint is epoch 396:

`outputs/training/f2p/checkpoints/best_test_noisy_masked_mse_checkpoint.pth`

Training uses synthetic noisy observations of the 701 training references; clean targets are not used in the optimization loss. **Checkpoint selection uses masked MSE on fixed noisy test observations**, not independent validation data. Test clean spectra are used to generate those synthetic observations and to compute final metrics, but not as clean targets for selection loss. This distinction is retained from the original experiment and should be reported explicitly.

Some historical free-text metadata says "fixed 5%" despite the actual `mask_ratio: 0.03` parameter passed to both training and noisy-observation selection datasets. The original files are preserved; the final model's numerical mask-ratio setting is 0.03. The new training module describes the configured ratio rather than repeating that old label.

Final inference matches the historical comparison run: per-input min-max with epsilon 1e-8, native-grid inference with CUDA autocast, scale restoration and the repository's noisy-input-based amplitude correction. CPU diagnostics disable CUDA autocast and can differ slightly from the saved GPU output. The adapter supports the shared repository rename from `Fast2PeakRPE` to `F2P` and loads all weights strictly.
