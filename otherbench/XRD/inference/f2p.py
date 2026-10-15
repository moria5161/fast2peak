"""Zero-shot F2P inference and amplitude restoration."""

import time
import numpy as np
import torch
from otherbench.XRD.utils.data import normalize_rows


def infer_f2p(
    model: torch.nn.Module, noisy: np.ndarray, batch_size: int, device: torch.device
) -> tuple[np.ndarray, float]:
    from utils.base import spectra_correction

    input_normalized = normalize_rows(noisy)
    minimum = noisy.min(axis=1, keepdims=True)
    maximum = noisy.max(axis=1, keepdims=True)
    batches = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(noisy), batch_size):
            batch = torch.from_numpy(input_normalized[start : start + batch_size]).unsqueeze(1)
            prediction = model(batch.to(device=device, dtype=torch.float32))
            batches.append(prediction.squeeze(1).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - started
    reconstructed_normalized = np.concatenate(batches, axis=0)
    reconstructed = reconstructed_normalized * (maximum - minimum + 1e-8) + minimum
    return spectra_correction(noisy, reconstructed).astype(np.float32), elapsed
