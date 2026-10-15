"""Original 8192-point author-network inference and window adaptation."""

import time
import numpy as np
import torch


def _window_starts(length: int, window_length: int, stride: int) -> list[int]:
    if window_length > length:
        return []
    starts = list(range(0, length - window_length + 1, stride))
    final_start = length - window_length
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _infer_overlapping_windows(
    model: torch.nn.Module,
    values: np.ndarray,
    window_length: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run a fixed/native-length model and blend overlapping output windows."""
    starts = _window_starts(values.shape[1], window_length, stride)
    if not starts:
        raise ValueError(
            f"input length {values.shape[1]} is shorter than window {window_length}"
        )
    windows = np.stack(
        [values[:, start : start + window_length] for start in starts], axis=1
    )
    flat = windows.reshape(-1, window_length)
    predicted_batches = []
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            batch = torch.from_numpy(flat[start : start + batch_size]).unsqueeze(1)
            prediction = model(batch.to(device=device, dtype=torch.float32))
            predicted_batches.append(prediction.squeeze(1).cpu().numpy())
    predicted = np.concatenate(predicted_batches).reshape(
        len(values), len(starts), window_length
    )

    weight = np.maximum(np.hanning(window_length).astype(np.float32), 1e-3)
    accumulator = np.zeros_like(values, dtype=np.float32)
    denominator = np.zeros(values.shape[1], dtype=np.float32)
    for window_index, start in enumerate(starts):
        accumulator[:, start : start + window_length] += predicted[:, window_index] * weight
        denominator[start : start + window_length] += weight
    return accumulator / np.maximum(denominator[None, :], 1e-8)


def infer_author_network(
    model: torch.nn.Module,
    noisy: np.ndarray,
    native_length: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    normalization: str = "positive_max",
) -> tuple[np.ndarray, float]:
    if normalization == "none":
        scale = np.ones((len(noisy), 1), dtype=np.float32)
    elif normalization == "maxabs":
        scale = np.max(np.abs(noisy), axis=1, keepdims=True).astype(np.float32)
        scale = np.where(scale > 0, scale, 1.0)
    elif normalization == "positive_max":
        scale = noisy.max(axis=1, keepdims=True).astype(np.float32)
        scale = np.where(scale > 0, scale, 1.0)
    else:
        raise ValueError(f"unsupported author-network normalization: {normalization}")
    normalized = (noisy / scale).astype(np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.time()
    if normalized.shape[1] < native_length:
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(noisy), batch_size):
                batch = torch.from_numpy(normalized[start : start + batch_size]).unsqueeze(1)
                batch = batch.to(device=device, dtype=torch.float32)
                batch = torch.nn.functional.interpolate(
                    batch, size=native_length, mode="linear", align_corners=True
                )
                prediction = model(batch)
                prediction = torch.nn.functional.interpolate(
                    prediction, size=noisy.shape[1], mode="linear", align_corners=True
                )
                predictions.append(prediction.squeeze(1).cpu().numpy())
        normalized_prediction = np.concatenate(predictions, axis=0)
    elif normalized.shape[1] == native_length:
        normalized_prediction = _infer_overlapping_windows(
            model, normalized, native_length, native_length, batch_size, device
        )
    else:
        normalized_prediction = _infer_overlapping_windows(
            model, normalized, native_length, stride, batch_size, device
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - started
    return (normalized_prediction * scale).astype(np.float32), elapsed
