# Spectroscopic Denoising Benchmark Results

This document summarizes the final NMR, IR and experimental PXRD benchmark tables. Within each benchmark, all methods are evaluated using the same test references and noisy inputs. Higher Pearson correlation, SNR and Peak F1 are better; lower RMSE is better.

## NMR: Maybridge ¹H NMR

**Table S4. Quantitative comparison on the Maybridge ¹H NMR denoising benchmark.**

| Methods | Pearson ↑ | RMSE (×10⁻³) ↓ | SNR (dB) ↑ | Peak F1 ↑ |
|---|---:|---:|---:|---:|
| RAW | 0.692937 | 53.313 | 2.187 | 0.039786 |
| WT | 0.912430 | 16.468 | 11.405 | 0.349464 |
| Cadzow | 0.961960 | 13.464 | 12.295 | 0.489976 |
| LD-Net | 0.973669 | 8.418 | 16.966 | 0.555886 |
| DN-Unet | 0.986677 | 6.331 | 19.347 | 0.831536 |
| F2P | 0.986635 | 6.703 | 18.311 | 0.782349 |

The benchmark uses the Maybridge supplementary experimental spectra, with a molecule-level split of 701 training, 150 validation and 150 test records. Test references are interpolated to 6,400 points before independent Gaussian noise is added at σ = 0.01, 0.05 and 0.10, yielding 450 test pairs per method. No clipping or post-noise normalization is applied. Metrics are averaged over all test pairs; RMSE is multiplied by 1,000 for display.

WT and Cadzow are classical denoisers. LD-Net and DN-Unet use supervised fine-tuning. F2P uses self-supervised training with a depth-4 model and a mask ratio of 0.03. Its checkpoint is determined by masked MSE on fixed noisy test observations, without clean targets in the selection criterion; this is not independent validation-based model selection or zero-shot inference.

DN-Unet has the highest Pearson correlation, SNR and Peak F1 and the lowest RMSE in this table. F2P improves all four metrics relative to RAW, WT, Cadzow and LD-Net.

Sources: [Table S4](NMR/docs/table_s4.csv), [full-precision aggregate metrics](NMR/docs/overall_leaderboard.csv), [data protocol](NMR/docs/data_README.md) and [training protocol](NMR/docs/model_provenance.md).

## IR: ATR Reference Spectra

**Quantitative comparison on the IR denoising benchmark.**

| Methods | Pearson mean ↑ | Pearson median ↑ | RMSE ↓ | SNR (dB) ↑ |
|---|---:|---:|---:|---:|
| RAW | 0.8503 | 0.8726 | 0.0748 | 5.57 |
| SavGol | 0.9143 | 0.9316 | 0.0457 | 10.04 |
| Autoencoder | 0.9444 | 0.9462 | 0.0408 | 11.55 |
| F2P | 0.9849 | 0.9869 | 0.0217 | 16.38 |

The benchmark uses 110 reference spectra, split by reference identity into 81 training, 9 validation and 20 test references. Each reference is normalized before independent zero-mean Gaussian noise is added, with σ sampled uniformly from [0.05, 0.10] for each realization. No baseline, clipping or post-noise normalization is applied. The test set contains 4,000 pairs from the 20 held-out references.

The autoencoder is trained for 100 epochs, with its checkpoint determined by validation MSE. SavGol is applied directly, and pretrained F2P is evaluated without IR fine-tuning. The reported F2P outputs include amplitude correction. Pearson mean and median are computed across test spectra; RMSE and SNR are means of per-spectrum metrics.

F2P has the highest Pearson mean, Pearson median and SNR and the lowest RMSE among the compared methods.

Sources: [full-precision aggregate metrics](IR/docs/leaderboard.csv), [evaluation summary](IR/docs/evaluation_summary.json), [data provenance](IR/docs/data_README.md) and [evaluation protocol](IR/docs/workflow.md).

## XRD: Experimental PXRD

**Table S6. Quantitative comparison on the experimental PXRD denoising benchmark.**

| Methods | Pearson ↑ | RMSE ↓ | SNR (dB) ↑ | Peak F1 ↑ |
|---|---:|---:|---:|---:|
| RAW | 0.8766 | 0.0455 | 9.11 | 0.264 |
| SavGol | 0.9375 | 0.0290 | 12.08 | 0.521 |
| Wavelet | 0.9472 | 0.0274 | 12.75 | 0.469 |
| F2P | 0.9592 | 0.0246 | 13.69 | 0.563 |

The benchmark contains 715 quality-filtered experimental PXRD references from the single-metal and binary-metal synthesis collections. Min-max-normalized references are evaluated under seven noise conditions: Gaussian σ = 0.02, 0.05 and 0.10; Poisson peak counts = 100, 50 and 10; and mixed Poisson peak counts = 500 with Gaussian σ = 0.03. Generated negative intensities are clipped to zero.

SavGol and Wavelet are applied directly. Pretrained F2P is evaluated without XRD training or fine-tuning. Each method is evaluated on all 715 references under every condition. The reported aggregate metrics are unweighted means of the seven condition-level means.

F2P has the highest aggregate Pearson correlation, SNR and Peak F1 and the lowest aggregate RMSE among the compared methods.

Reporting note: Table S6 retains the finalized SavGol Pearson value of 0.9375. The stored full-precision macro-average is 0.9374415128, which rounds to 0.9374 at four decimal places. The underlying computed metrics are retained unchanged.

Sources: [Table S6](XRD/docs/table_s6.csv), [full-precision macro-averages](XRD/docs/leaderboard_macro_average.csv), [per-condition metrics](XRD/docs/leaderboard.csv), [data provenance](XRD/docs/data_README.md) and [evaluation protocol](XRD/docs/workflow.md).

## Interpretation

The experimental NMR and PXRD spectra serve as measured reference spectra rather than physically noise-free ground truth. Metric values should be compared within each benchmark, because the datasets, noise levels and training protocols differ across benchmarks.
