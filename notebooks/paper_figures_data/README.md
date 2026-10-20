# Paper Figure Spectra

This directory contains all spectral inputs read by `../plot_paper.ipynb`.
The notebook does not need the project's `datasets/` or `experiments/` directories.
Numerical metric arrays are embedded in the corresponding plotting cells.

| Directory | Figure content |
|---|---|
| `simulated_denoising/` | F2P and baseline predictions at five simulated noise levels |
| `cross_instrument_spectra/` | RAW and F2P spectra from five Raman instruments |
| `experimental_benchmark/` | Selected MoS2, WS2, WSe2, HeLa and SSEs benchmark spectra |
| `simulated_reference_spectra/` | Shared clean and noisy references for ablation figures |
| `masking_ablation_spectra/` | F2P trained with and without masking |
| `mask_vs_full/` | F2P masked-loss and full-spectrum-loss comparison |
| `transformer_vs_cnn/` | F2P Transformer and CNN predictions |
| `cross_modality_spectra/` | RAW and F2P examples for IR, NMR and XRD |

Copied TXT spectra retain their original bytes. Their first two columns contain
the spectral coordinate and intensity; coordinate units follow the figure labels.
Cross-modality examples are retained as plotted examples, not relabeled as the
final IR/NMR/XRD quantitative benchmark datasets.

## Experimental Benchmark Export

Each TXT file has 1,600 rows and the following 11 columns:

```text
raman_shift raw sg313 wtd3 peer resunet p2p smae f2p f2p_ft gt
```

The source rows are zero-based: MoS2 = `2DMs[12]`, WS2 = `2DMs[24]`,
WSe2 = `2DMs[49]`, HeLa = `Hela[21]`, and SSEs = `SSEs[2]`.

The Raman axes reproduce the original notebook's linear grids:
148–642, 238.424–555.45, 2–331, 500–3000 and 177–2718 cm⁻¹, respectively.
They are plotting grids, not newly recovered acquisition metadata.
Intensities are exported without normalization, offsets or rescaling.
The notebook retains its existing display operations, including the factor of
0.5 applied to the WSe2 P2P trace.

`manifest.json` records source files, HDF5 keys and rows, and SHA-256 checksums.
These source paths are provenance records only; they are not needed for plotting.
No model inference or metric recalculation was performed during packaging.
