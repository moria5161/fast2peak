# PXRD macro-average results

Unweighted macro averages over 7 noise conditions. Each condition has equal weight.

| Method | Pearson ↑ | RMSE ↓ | SNR (dB) ↑ | Peak F1 ↑ |
|---|---:|---:|---:|---:|
| RAW | 0.8766 | 0.0455 | 9.11 | 0.2639 |
| SavGol | 0.9374 | 0.0290 | 12.08 | 0.5210 |
| Wavelet | 0.9472 | 0.0274 | 12.75 | 0.4693 |
| F2P | 0.9592 | 0.0246 | 13.69 | 0.5634 |

Conditions: gaussian_sigma_0.02, gaussian_sigma_0.05, gaussian_sigma_0.10, mixed_peak_500_sigma_0.03, poisson_peak_10, poisson_peak_100, poisson_peak_50.

GT is a high-quality measured PXRD reference, not physically noise-free ground truth.
