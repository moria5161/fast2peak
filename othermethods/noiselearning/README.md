# NoiseLearning

`NoiseLearning.py` and `checkpoints/562800.pt` were relocated from
`notebooks/backup/NoiseLearning/`. The model architecture, weights, DCT
processing, resampling and original scale-factor sweep were not changed.
The existing inference implementation requires CUDA.

Run from the project root on a compute node:

```bash
python othermethods/noiselearning/NoiseLearning.py \
    --input-dir path/to/txt_spectra \
    --output-dir experiments/noiselearning
```

The default checkpoint is resolved relative to this file; `--checkpoint`
can override it. Instrument-noise data now live in `datasets/noise/`.
Historical preparation notebooks live in `notebooks/data_preparation/`.
No inference was performed during this reorganization.
