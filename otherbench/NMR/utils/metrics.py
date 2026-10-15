"""Original native-grid spectral and peak-matching metrics."""

import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import linear_sum_assignment


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def normalize_positive(spectrum: np.ndarray) -> np.ndarray:
    maximum = float(np.max(spectrum))
    return spectrum / maximum if maximum > 0 else spectrum.copy()


def peak_metrics(
    target: np.ndarray, prediction: np.ndarray, ppm: np.ndarray, config: dict
) -> tuple[float, float, float, float]:
    target_n = normalize_positive(target)
    prediction_n = normalize_positive(prediction)
    kwargs = {
        "prominence": float(config["min_prominence"]),
        "distance": int(config["min_distance_points"]),
    }
    target_peaks, _ = find_peaks(target_n, **kwargs)
    prediction_peaks, _ = find_peaks(prediction_n, **kwargs)
    if len(target_peaks) == 0 and len(prediction_peaks) == 0:
        return 1.0, 0.0, 0.0, 0.0
    if len(target_peaks) == 0 or len(prediction_peaks) == 0:
        return 0.0, float("nan"), float("nan"), float("nan")

    distances = np.abs(ppm[target_peaks, None] - ppm[prediction_peaks][None, :])
    target_match, prediction_match = linear_sum_assignment(distances)
    valid = distances[target_match, prediction_match] <= float(config["matching_tolerance_ppm"])
    matched_target = target_peaks[target_match[valid]]
    matched_prediction = prediction_peaks[prediction_match[valid]]
    true_positive = int(valid.sum())
    precision = true_positive / len(prediction_peaks)
    recall = true_positive / len(target_peaks)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    if not true_positive:
        return f1, float("nan"), float("nan"), float("nan")

    position_mae = float(np.mean(np.abs(ppm[matched_target] - ppm[matched_prediction])))
    intensity_mae = float(
        np.mean(np.abs(target_n[matched_target] - prediction_n[matched_prediction]))
    )
    ppm_per_point = float(np.median(np.abs(np.diff(ppm))))
    target_width = peak_widths(target_n, matched_target, rel_height=0.5)[0] * ppm_per_point
    prediction_width = peak_widths(prediction_n, matched_prediction, rel_height=0.5)[0] * ppm_per_point
    width_mae = float(np.mean(np.abs(target_width - prediction_width)))
    return f1, position_mae, intensity_mae, width_mae


def metrics_for_sample(
    target: np.ndarray,
    prediction: np.ndarray,
    noisy: np.ndarray,
    ppm: np.ndarray,
    peak_config: dict,
) -> dict[str, float]:
    error = target - prediction
    noisy_error = target - noisy
    target_power = max(float(np.sum(target**2)), 1e-12)
    error_power = max(float(np.sum(error**2)), 1e-12)
    noisy_error_power = max(float(np.sum(noisy_error**2)), 1e-12)
    snr = 10.0 * np.log10(target_power / error_power)
    input_snr = 10.0 * np.log10(target_power / noisy_error_power)
    cosine_denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    cosine = float(np.dot(target, prediction) / cosine_denominator) if cosine_denominator else 0.0
    peak_f1, position_mae, intensity_mae, width_mae = peak_metrics(
        target, prediction, ppm, peak_config
    )
    target_area = abs(float(np.trapezoid(np.abs(target), ppm)))
    prediction_area = abs(float(np.trapezoid(np.abs(prediction), ppm)))
    return {
        "pearson": pearson(target, prediction),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "snr_db": float(snr),
        "snr_improvement_db": float(snr - input_snr),
        "spectral_angle_degree": float(
            np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        ),
        "first_derivative_rmse": float(np.sqrt(np.mean(np.diff(error) ** 2))),
        "peak_f1": peak_f1,
        "peak_position_mae_ppm": position_mae,
        "peak_intensity_mae": intensity_mae,
        "peak_width_mae_ppm": width_mae,
        "integrated_absolute_intensity_relative_error": float(
            abs(prediction_area - target_area) / max(target_area, 1e-12)
        ),
    }


def aggregate(rows: list[dict]) -> dict[str, float]:
    result: dict[str, float] = {}
    excluded = {"condition", "method", "sample_index"}
    for name in (key for key in rows[0] if key not in excluded):
        values = np.asarray([row[name] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        result[f"{name}_mean"] = float(finite.mean()) if len(finite) else float("nan")
        result[f"{name}_median"] = float(np.median(finite)) if len(finite) else float("nan")
    return result
