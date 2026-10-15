# Maybridge data and pairing

The benchmark uses the Maybridge supplementary ¹H NMR matrix associated with Castillo et al., *Journal of Cheminformatics* **2014**, *6*, 9, DOI [10.1186/1758-2946-6-9](https://doi.org/10.1186/1758-2946-6-9). Local source files are `datasets/raw/MOESM2_ESM.txt` and `datasets/raw/molfile.sdf`. They are retained with the ignored data assets.

The TXT has 1,001 ordered spectra × 1,024 values; the SDF has the matching 1,001 unique `compound_code` records. The original article describes 1,000 samples, while the actual paired supplementary files contain 1,001; the workflow preserves all paired records. The earlier MTBLS2550 download/conversion experiment is archived and is not part of Table S4.

Each source row is divided by its positive maximum and linearly interpolated from 1,024 to 1,600 points. Since the supplementary TXT lacks a chemical-shift axis, descending bin centers are reconstructed from the published 20.693 ppm window centered at 6.175 ppm. The stored 1,600-point endpoints are approximately 16.511396 and −4.161396 ppm. These are reconstructed coordinates, not a directly supplied acquisition-axis array.

The seed-42, molecule-level split is 701 training, 150 validation and 150 test records. IDs and source indices are frozen in `datasets/splits/splits.h5` and the public [splits.json](splits.json); the three identity sets do not overlap.

For the final test, each frozen 1,600-point clean reference is linearly interpolated to 6,400 points **before** adding independent Gaussian noise at sigma 0.01, 0.05 and 0.10. The signed noisy values are retained: no clipping or post-noise normalization is applied. For condition index `c` and test sample index `i`, the RNG seed is `42 + 40000079 + c*1000003 + i*97`. The resulting `datasets/benchmark/benchmark.h5` contains 150 GT rows, a 6,400-point ppm axis, IDs/indices and three noisy datasets.

Supervised fine-tuning uses a separate fixed 1,600-point validation benchmark in `datasets/validation/benchmark.h5`. Its clean references are positive-max normalized and noise uses condition seeds 42, 43 and 44. During DN-Unet/LD-Net training, noisy/clean 1,600-point pairs are adapted to the authors' native 8,192-point grid.

The processed measured spectra are reference pseudo-GT; they are not physically noise-free truth. The Cadzow implementation reconstructs a pseudo-FID through Hilbert/FFT operations on the real spectrum; the original complex acquisition FID is not read.

## Local assets

| Location | Contents |
|---|---|
| `datasets/raw/` | Original TXT and matching SDF |
| `datasets/source/clean.h5` | Rebuilt normalized 1,600-point source matrix, ppm and compound IDs |
| `datasets/splits/` | Frozen train/validation/test spectra, IDs, indices and split metadata |
| `datasets/validation/` | Frozen supervised validation data and historical numerical metadata |
| `datasets/benchmark/benchmark.h5` | Final 6,400-point test references and noisy pairs |

Run `main.py source`, then `main.py split`, to rebuild the initial matrix/split. `main.py prepare` regenerates native-grid test pairs and the supervised validation benchmark from the frozen split. Historical metadata may contain original absolute paths; these are retained as provenance.
