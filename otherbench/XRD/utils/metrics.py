"""Original per-spectrum and peak-matching benchmark metrics."""

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from otherbench.XRD.utils.data import normalize_rows


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def peak_metrics(
    target: np.ndarray, prediction: np.ndarray, axis: np.ndarray, config: dict
) -> tuple[float, float, float]:
    target_n = normalize_rows(target[None, :])[0]
    prediction_n = normalize_rows(prediction[None, :])[0]
    kwargs = {
        "prominence": float(config["min_prominence"]),
        "distance": int(config["min_distance_points"]),
    }
    target_peaks, _ = find_peaks(target_n, **kwargs)
    prediction_peaks, _ = find_peaks(prediction_n, **kwargs)
    if len(target_peaks) == 0 and len(prediction_peaks) == 0:
        return 1.0, 0.0, 0.0
    if len(target_peaks) == 0 or len(prediction_peaks) == 0:
        return 0.0, float("nan"), float("nan")
    distances = np.abs(axis[target_peaks, None] - axis[prediction_peaks][None, :])
    target_match, prediction_match = linear_sum_assignment(distances)
    valid = distances[target_match, prediction_match] <= float(config["matching_tolerance_degree"])
    matched_target = target_peaks[target_match[valid]]
    matched_prediction = prediction_peaks[prediction_match[valid]]
    true_positive = int(valid.sum())
    precision = true_positive / len(prediction_peaks)
    recall = true_positive / len(target_peaks)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    if true_positive:
        position_mae = float(np.mean(np.abs(axis[matched_target] - axis[matched_prediction])))
        intensity_mae = float(np.mean(np.abs(target_n[matched_target] - prediction_n[matched_prediction])))
    else:
        position_mae, intensity_mae = float("nan"), float("nan")
    return f1, position_mae, intensity_mae


def metrics_for_sample(
    target: np.ndarray, prediction: np.ndarray, noisy: np.ndarray, axis: np.ndarray, peak_config: dict
) -> dict[str, float]:
    error = target - prediction
    noisy_error = target - noisy
    target_power = np.sum(target**2)
    error_power = max(float(np.sum(error**2)), 1e-12)
    noisy_error_power = max(float(np.sum(noisy_error**2)), 1e-12)
    snr = 10.0 * np.log10(target_power / error_power)
    input_snr = 10.0 * np.log10(target_power / noisy_error_power)
    cosine_denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    cosine = float(np.dot(target, prediction) / cosine_denominator) if cosine_denominator > 0 else 0.0
    peak_f1, peak_position_mae, peak_intensity_mae = peak_metrics(
        target, prediction, axis, peak_config
    )
    return {
        "pearson": pearson(target, prediction),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "snr_db": float(snr),
        "snr_improvement_db": float(snr - input_snr),
        "spectral_angle_degree": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
        "peak_f1": peak_f1,
        "peak_position_mae_degree": peak_position_mae,
        "peak_intensity_mae": peak_intensity_mae,
        "integrated_intensity_relative_error": float(
            abs(np.trapezoid(prediction, axis) - np.trapezoid(target, axis))
            / max(abs(np.trapezoid(target, axis)), 1e-12)
        ),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    result = {}
    metric_names = [key for key in rows[0] if key not in ("condition", "method", "sample_index")]
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        result[f"{name}_mean"] = float(finite.mean()) if len(finite) else float("nan")
        result[f"{name}_median"] = float(np.median(finite)) if len(finite) else float("nan")
    return result
