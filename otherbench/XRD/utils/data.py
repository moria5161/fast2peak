"""Spectrum normalization shared by preprocessing and metrics."""

import numpy as np


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    values -= values.min(axis=1, keepdims=True)
    maxima = values.max(axis=1, keepdims=True)
    np.divide(values, maxima, out=values, where=maxima > 0)
    return values
