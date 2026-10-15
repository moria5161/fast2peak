**Table S6. Quantitative comparison on the experimental PXRD denoising benchmark.**

| Methods | Pearson ↑ | RMSE ↓ | SNR (dB) ↑ | Peak F1 ↑ |
|---|---:|---:|---:|---:|
| RAW | 0.8766 | 0.0455 | 9.11 | 0.264 |
| SavGol | 0.9375 | 0.0290 | 12.08 | 0.521 |
| Wavelet | 0.9472 | 0.0274 | 12.75 | 0.469 |
| F2P | 0.9592 | 0.0246 | 13.69 | 0.563 |

This is the final manuscript table supplied by the project author. The seven-condition computed macro-average is retained separately in `leaderboard_macro_average.csv`. Its SavGol Pearson is 0.93744151 (0.9374 to four decimal places), whereas the supplied final table reports 0.9375. Stored per-condition metrics were not modified to remove this discrepancy. Metric export does not overwrite the supplied Table S6 files.
