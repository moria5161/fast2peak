import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


def svd_denoise(signal, window_size=20, rank=5):

    # 构造 Hankel 矩阵
    L = window_size
    K = len(signal) - L + 1  # 列数
    H = np.column_stack([signal[i:i+L] for i in range(K)])

    # SVD 分解
    U, S, Vt = svd(H, full_matrices=False)

    # 选择前 rank 个奇异值
    S[rank:] = 0  # 截断较小奇异值

    # 重构信号
    H_denoised = (U * S) @ Vt

    # 通过 Hankel 矩阵对角线平均恢复信号
    denoised_signal = np.zeros(len(signal))
    counts = np.zeros(len(signal))

    for i in range(L):
        denoised_signal[i:i+K] += H_denoised[i]
        counts[i:i+K] += 1

    denoised_signal /= counts  # 取平均，保证长度不变

    return denoised_signal
