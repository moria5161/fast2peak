"""Historical numerical metrics used by the project-adapted SMAE evaluator."""

import numpy as np


def calculate_metrics(spec_denoised, spec_gt):
    """Return SNR, RMSE and the existing SSS=(1-SAD)*PCC metric; SAD is in radians."""
    x = np.array(spec_denoised)
    y = np.array(spec_gt)
    rmse = np.sqrt(np.mean((x - y) ** 2))
    snr = 10 * np.log10(np.sum(y ** 2) / (np.sum((x - y) ** 2) + 1e-8))
    inner_product = np.dot(x, y)
    norms = np.linalg.norm(x) * np.linalg.norm(y)
    sad = np.arccos(np.clip(inner_product / norms, -1.0, 1.0))
    pcc = np.corrcoef(x, y)[0, 1]
    sss = (1 - sad) * pcc
    return snr, rmse, sss
