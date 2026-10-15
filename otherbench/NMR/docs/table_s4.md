**Table S4. Quantitative comparison on the Maybridge ¹H NMR denoising benchmark.**

| Methods | Pearson ↑ | RMSE (×10⁻³) ↓ | SNR (dB) ↑ | Peak F1 ↑ |
|---|---:|---:|---:|---:|
| RAW | 0.692937 | 53.313 | 2.187 | 0.039786 |
| WT | 0.912430 | 16.468 | 11.405 | 0.349464 |
| Cadzow | 0.961960 | 13.464 | 12.295 | 0.489976 |
| LD-Net | 0.973669 | 8.418 | 16.966 | 0.555886 |
| DN-Unet | 0.986677 | 6.331 | 19.347 | 0.831536 |
| F2P | 0.986635 | 6.703 | 18.311 | 0.782349 |

Values match the final table supplied by the project author. RMSE values in this table equal the raw per-spectrum mean RMSE multiplied by 1,000. Full-precision computed metrics are in `overall_leaderboard.csv`. The export command does not overwrite this fixed manuscript table.
