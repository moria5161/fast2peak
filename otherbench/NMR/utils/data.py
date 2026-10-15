"""Frozen-grid interpolation and native-grid noise construction."""

import numpy as np


def resample_rows(values: np.ndarray, length: int) -> np.ndarray:
    """Linear spectral interpolation; stable and free of cubic ringing."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] == length:
        return values.copy()
    old_x = np.linspace(0.0, 1.0, values.shape[-1], dtype=np.float64)
    new_x = np.linspace(0.0, 1.0, length, dtype=np.float64)
    output = np.empty((len(values), length), dtype=np.float32)
    for index, row in enumerate(values):
        output[index] = np.interp(new_x, old_x, row).astype(np.float32)
    return output


def build_fixed_native_noisy(clean: np.ndarray, length: int,
                             noise_levels: list[float], seed: int) -> dict[str, np.ndarray]:
    """Create reproducible native-6400 noisy validation inputs from test sources."""
    clean_6400 = resample_rows(clean, length)
    result = {}
    for condition_index, sigma in enumerate(noise_levels):
        name = f"gaussian_sigma_{sigma:.3f}"
        noisy = np.empty_like(clean_6400)
        for sample_index, spectrum in enumerate(clean_6400):
            rng = np.random.default_rng(
                seed + condition_index * 1_000_003 + sample_index * 97
            )
            noisy[sample_index] = spectrum + rng.normal(
                0.0, sigma, spectrum.shape
            ).astype(np.float32)
        result[name] = noisy
    return result
