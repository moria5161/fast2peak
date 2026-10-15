# Local data, checkpoints and TensorBoard logs

## Data

The existing ignored `datasets/` directory preserves `X_reference.npy`, `y_reference.npy`, `X_finetune.npy` and `y_finetune.npy` without modification. These historical classification arrays are distinct from the HDF5 denoising data used by the adapted training and evaluation examples. Their presence does not make them the data source of the three benchmarks in `otherbench/RESULTS.md`.

Adapted pretraining expects a HDF5 `noisy` array with shape `[N, L]`, where `L` matches the model input length. Evaluation uses the same layout and optionally matching `gt`. Data-generation or resplitting is not performed by this reorganization.

Default denoising inputs are shared with F2P and are stored at the repository root:

| File | Content |
|---|---|
| `../../datasets/train/train.h5` | 360,000 training rows × 1,600 points; `noisy`, `gt`, `wavenumbers` |
| `../../datasets/test/Simu.h5` | 1,024 test rows × 1,600 points; `noisy`, `gt` |

Paths above are relative to `othermethods/smae`. The YAML references the shared files directly; no denoising-data copies are kept in the SMAE subproject. The temporary duplicate files were removed after a byte-for-byte comparison with the retained shared originals. These shared HDF5 files are distinct from the four historical NPY arrays listed above and are not part of the historical 116-asset preservation manifest.

The original repository describes bacteria-ID and DEEPER datasets; its data links and citation are available in the [upstream README](https://github.com/pengjuRen99/SpectraMAENet/blob/7939f2e6d3e6e975b687e4f07e56b9ce51cae5e3/README.md). Dataset identity and permission should be confirmed before separately distributing local data.

## Historical checkpoints

All previous `experiments/` files are copied intact to `outputs/experiments/`, with run names unchanged. All previous `Pretrain_weight_moria/` files are copied intact to `outputs/pretrained/`. No checkpoint is discarded based on performance or epoch number. New runs use `outputs/retrained/` by default.

## TensorBoard

All previous `logs/` files are copied intact to `outputs/logs/`, preserving event filenames and relative run directories. Resume runs therefore remain separate event files rather than being merged, filtered or rewritten.

```bash
tensorboard --logdir othermethods/smae/outputs/logs
```

File sizes and SHA-256 digests for data, weights and TensorBoard events are recorded in [asset_manifest.json](asset_manifest.json). The full pre-reorganization snapshot is retained outside the Git repository; the manifest identifies it by directory name. These assets are Git-ignored, not deleted, and must be supplied separately when reproducing runs from a clean checkout.
