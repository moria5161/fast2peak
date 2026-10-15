"""Final 6400-point F2P inference, matching the mask-0.03 comparison run."""

import time
import numpy as np
import torch


def infer_f2p(model, noisy, native_length, stride, batch_size, device):
    from utils.base import spectra_correction
    if noisy.shape[1] != native_length or native_length != 6400:
        raise ValueError("The final Table S4 model uses native 6400-point spectra")
    minimum = noisy.min(axis=1, keepdims=True)
    span = noisy.max(axis=1, keepdims=True) - minimum
    normalized = ((noisy - minimum) / (span + 1e-8)).astype(np.float32)
    outputs = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(noisy), batch_size):
            batch = torch.from_numpy(normalized[start:start + batch_size]).unsqueeze(1).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs.append(model(batch).squeeze(1).float().cpu().numpy())
    prediction = np.concatenate(outputs) * (span + 1e-8) + minimum
    return spectra_correction(noisy, prediction).astype(np.float32), time.time() - started
