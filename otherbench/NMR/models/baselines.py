"""WT and Cadzow pseudo-FID baseline implementations."""

import numpy as np
import pywt
from scipy.signal import hilbert


def wavelet_denoise(spectra: np.ndarray, config: dict) -> np.ndarray:
    output = np.empty_like(spectra, dtype=np.float32)
    family = config["family"]
    level = int(config["level"])
    for index, spectrum in enumerate(spectra):
        coefficients = pywt.wavedec(spectrum, family, mode="symmetric", level=level)
        finest = coefficients[-1]
        sigma = np.median(np.abs(finest - np.median(finest))) / 0.67448975
        threshold = sigma * np.sqrt(2.0 * np.log(len(spectrum)))
        thresholded = [coefficients[0]] + [
            pywt.threshold(detail, threshold, mode=config["threshold_mode"])
            for detail in coefficients[1:]
        ]
        output[index] = pywt.waverec(thresholded, family, mode="symmetric")[: len(spectrum)]
    return output


def _hankelize(matrix: np.ndarray, output_length: int) -> np.ndarray:
    rows, columns = matrix.shape
    output = np.zeros(output_length, dtype=matrix.dtype)
    counts = np.zeros(output_length, dtype=np.float64)
    row_indices = np.arange(rows)[:, None]
    column_indices = np.arange(columns)[None, :]
    indices = (row_indices + column_indices).ravel()
    np.add.at(output, indices, matrix.ravel())
    np.add.at(counts, indices, 1.0)
    return output / counts


def _automatic_cadzow_rank(singular_values: np.ndarray, rows: int, columns: int, max_rank: int) -> int:
    beta = min(rows, columns) / max(rows, columns)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    threshold = omega * float(np.median(singular_values))
    return max(1, min(int(np.sum(singular_values > threshold)), max_rank))


def cadzow_pseudo_fid(spectra: np.ndarray, config: dict) -> np.ndarray:
    """Cadzow on a pseudo-FID reconstructed from the frequency-domain real part.

    Original acquisition FID is unavailable for the processed Maybridge matrix. The Hilbert
    transform supplies a quadrature component before inverse FFT. This is a
    documented compatibility approximation, not access to the original FID.
    """
    width = int(config["hankel_width"])
    iterations = int(config["iterations"])
    max_rank = int(config["max_rank"])
    output = np.empty_like(spectra, dtype=np.float32)
    for sample_index, spectrum_descending in enumerate(spectra):
        spectrum_ascending = spectrum_descending[::-1].astype(np.float64)
        complex_spectrum = hilbert(spectrum_ascending)
        fid = np.fft.ifft(np.fft.ifftshift(complex_spectrum))
        for _ in range(iterations):
            hankel = np.lib.stride_tricks.sliding_window_view(fid, width)
            left, singular_values, right_h = np.linalg.svd(hankel, full_matrices=False)
            rank = _automatic_cadzow_rank(
                singular_values, hankel.shape[0], hankel.shape[1], max_rank
            )
            low_rank = (left[:, :rank] * singular_values[:rank]) @ right_h[:rank]
            fid = _hankelize(low_rank, len(fid))
        reconstructed = np.fft.fftshift(np.fft.fft(fid)).real[::-1]
        output[sample_index] = reconstructed.astype(np.float32)
    return output
