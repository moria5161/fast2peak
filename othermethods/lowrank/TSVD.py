import numpy as np
import torch
from typing import Tuple

def svd_denoise(data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    使用奇异值分解(SVD)对二维矩阵进行降噪。
    """
    if data.ndim != 2:
        raise ValueError("输入数据必须是一个二维矩阵 (batch, length)。")

    matrix = torch.from_numpy(data).float()

    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    print("Original singular values:", S)
    if S.numel() <= 2:
        S_filtered = S
    else:
        log_S = torch.log1p(S)
        diff2 = torch.diff(torch.diff(log_S))

        mask = torch.ones_like(S, dtype=torch.bool)
        mask[1:-1] = torch.abs(diff2) > threshold

        S_filtered = S * mask

    reconstructed_matrix = U @ torch.diag(S_filtered) @ Vh

    return reconstructed_matrix.cpu().numpy()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new file.')

    data = np.load(args.input)
    print("Original data shape:", data.shape) # (batch, length)
    denoised_data = svd_denoise(data, threshold=7e-2)
    print("Denoised data shape:", denoised_data.shape)
    with args.output.open('xb') as f:
        np.save(f, denoised_data)
