# Benchmark Results

Finalized results; compare methods within each benchmark. Training and evaluation protocols differ across benchmarks.

## NMR

| Method | Pearson ↑ | RMSE (×10⁻³) ↓ | SNR (dB) ↑ | Peak F1 ↑ |
|---|---:|---:|---:|---:|
| RAW | 0.692937 | 53.313 | 2.187 | 0.039786 |
| WT | 0.912430 | 16.468 | 11.405 | 0.349464 |
| Cadzow | 0.961960 | 13.464 | 12.295 | 0.489976 |
| LD-Net | 0.973669 | 8.418 | 16.966 | 0.555886 |
| DN-Unet | 0.986677 | 6.331 | 19.347 | 0.831536 |
| F2P | 0.986635 | 6.703 | 18.311 | 0.782349 |

## IR

| Method | Pearson mean ↑ | Pearson median ↑ | RMSE ↓ | SNR (dB) ↑ |
|---|---:|---:|---:|---:|
| RAW | 0.8503 | 0.8726 | 0.0748 | 5.57 |
| SavGol | 0.9143 | 0.9316 | 0.0457 | 10.04 |
| Autoencoder | 0.9444 | 0.9462 | 0.0408 | 11.55 |
| F2P | 0.9849 | 0.9869 | 0.0217 | 16.38 |

## XRD

| Method | Pearson ↑ | RMSE ↓ | SNR (dB) ↑ | Peak F1 ↑ |
|---|---:|---:|---:|---:|
| RAW | 0.8766 | 0.0455 | 9.11 | 0.264 |
| SavGol | 0.9375 | 0.0290 | 12.08 | 0.521 |
| Wavelet | 0.9472 | 0.0274 | 12.75 | 0.469 |
| F2P | 0.9592 | 0.0246 | 13.69 | 0.563 |

XRD SavGol Pearson follows the finalized table; its computed value is 0.9374415.
