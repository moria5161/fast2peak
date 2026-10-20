# Low-Rank Methods

Relocated from `notebooks/backup/` without changing the denoising algorithms:

- `ALRMA.py`: adaptive low-rank processing of hyperspectral measurements.
- `TSVD.py`: truncated SVD for a two-dimensional data matrix.
- `SVD1D.py`: Hankel-matrix SVD for a one-dimensional spectrum.

These are separate implementations, not interchangeable interfaces.
The ALRMA demonstration reads the Horiba HeLa measurement under
`datasets/experimental/horiba/` and saves to `experiments/alrma/`.
Run numerical evaluations on a compute node. No evaluation was run during migration.
