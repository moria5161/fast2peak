"""Classical Coiflet wavelet and Anscombe denoisers."""

import numpy as np
import pywt


def wavelet_denoise(
    spectra: np.ndarray, family: str, level: int, threshold_mode: str
) -> np.ndarray:
    output = np.empty_like(spectra, dtype=np.float32)
    for index, spectrum in enumerate(spectra):
        coefficients = pywt.wavedec(spectrum, family, mode="symmetric", level=level)
        finest = coefficients[-1]
        sigma = np.median(np.abs(finest - np.median(finest))) / 0.67448975
        threshold = sigma * np.sqrt(2.0 * np.log(len(spectrum)))
        denoised_coefficients = [coefficients[0]] + [
            pywt.threshold(detail, threshold, mode=threshold_mode) for detail in coefficients[1:]
        ]
        output[index] = pywt.waverec(
            denoised_coefficients, family, mode="symmetric"
        )[: len(spectrum)]
    return np.clip(output, 0.0, None)


def anscombe_wavelet_denoise(
    noisy: np.ndarray, peak_counts: float, wavelet_config: dict
) -> np.ndarray:
    counts = np.clip(noisy, 0.0, None) * peak_counts
    transformed = 2.0 * np.sqrt(counts + 3.0 / 8.0)
    denoised = wavelet_denoise(
        transformed,
        wavelet_config["family"],
        int(wavelet_config["level"]),
        wavelet_config["threshold_mode"],
    )
    inverse_counts = np.clip((denoised / 2.0) ** 2 - 3.0 / 8.0, 0.0, None)
    return (inverse_counts / peak_counts).astype(np.float32)
