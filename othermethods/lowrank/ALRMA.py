# -*- coding: utf-8 -*-
"""
Reconstruct the ALRMA algorithm
Created on November 8th, 2025
@author: moria, Ren Group, Xiamen University
"""


import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
from scipy.io import loadmat
from scipy.signal import savgol_filter


def cube2spec(cube_tensor, patch_rows_list, patch_cols_list):

    row_coords = np.stack([np.meshgrid(sx, sy, indexing='ij')[0] for sx, sy in zip(patch_rows_list, patch_cols_list)])
    col_coords = np.stack([np.meshgrid(sx, sy, indexing='ij')[1] for sx, sy in zip(patch_rows_list, patch_cols_list)])

    rows_tensor = torch.tensor(row_coords, device=cube_tensor.device, dtype=torch.long)
    cols_tensor = torch.tensor(col_coords, device=cube_tensor.device, dtype=torch.long)

    all_patches = cube_tensor[rows_tensor, cols_tensor]

    return torch.mean(all_patches, dim=(1, 2))


def ALRMA(data, width, high, wave_range, windowsize=5, count=100, rank_cutoff_thr=1e-3, snr_gain_thr=1e-2):
    data = data.T
    n_bands = len(wave_range)
    mean_img = np.mean(data[wave_range, :], axis=0).reshape((high, width))

    padding_gap = windowsize - 1
    patch_cv_map = np.zeros((high - padding_gap, width - padding_gap))
    for i in range(high - padding_gap):
        for j in range(width - padding_gap):
            patch = mean_img[i:i + windowsize, j:j + windowsize]
            patch_mean = np.mean(patch)
            if patch_mean != 0:
                squared_diffs = (patch - patch_mean) ** 2
                patch_cv_map[i, j] = np.mean(squared_diffs) / (patch_mean ** 2)

    flat_cv_map = patch_cv_map.flatten()
    smooth_patch_indices = np.argpartition(flat_cv_map, count)[:count]
    rows_flat, cols_flat = np.unravel_index(smooth_patch_indices, patch_cv_map.shape)
    center_x, center_y = rows_flat + padding_gap // 2, cols_flat + padding_gap // 2

    patch_rows_list, patch_cols_list = [], []
    for i in range(len(center_x)):
        patch_rows_list.append(np.arange(center_x[i] - padding_gap // 2, center_x[i] + padding_gap // 2 + 1))
        patch_cols_list.append(np.arange(center_y[i] - padding_gap // 2, center_y[i] + padding_gap // 2 + 1))

    device = torch.device("cpu")

    data_tensor = torch.from_numpy(data).float().to(device)
    wave_range_tensor = torch.tensor(wave_range, device=device, dtype=torch.long)

    data_cube = data_tensor[wave_range_tensor, :].view(n_bands, high, width).permute(1, 2, 0)
    ref_specs_tensor = cube2spec(data_cube, patch_rows_list, patch_cols_list)

    ref_specs_np = ref_specs_tensor.cpu().numpy()
    thresholds = np.std(ref_specs_np, axis=1, keepdims=True) / 3
    denoised_ref_specs_np = pywt.threshold(ref_specs_np, thresholds, 'soft')
    ref_specs_tensor = torch.from_numpy(denoised_ref_specs_np).float().to(device)

    U, s, V = torch.svd(data_tensor)

    s_cpu = s.cpu().numpy()
    s_positive = s_cpu[s_cpu > 0]
    log_s_curvature = np.diff(np.log(s_positive), n=2)
    log_s_curvature = savgol_filter(log_s_curvature, 9, 3)
    try:
        est_rank = np.where(np.abs(log_s_curvature) <= rank_cutoff_thr)[0][0] + 2
    except IndexError:
        est_rank = len(s_positive)

    snrs = torch.zeros((count, est_rank), device=device)
    recon_data = torch.zeros_like(data_tensor)

    for i in range(est_rank):
        recon_data += s[i] * torch.outer(U[:, i], V[:, i])
        recon_cube = recon_data[wave_range_tensor, :].view(n_bands, high, width).permute(1, 2, 0)
        recon_specs_tensor = cube2spec(recon_cube, patch_rows_list, patch_cols_list)

        residuals = ref_specs_tensor - recon_specs_tensor
        signal_power = torch.mean(ref_specs_tensor ** 2, dim=1)
        noise_power = torch.mean(residuals ** 2, dim=1)

        safe_noise_power = torch.where(noise_power > 0, noise_power, 1.0)
        snrs[:, i] = 10 * torch.log10(signal_power / safe_noise_power)

    snrs_cpu = snrs.cpu().numpy()
    snr_padding = np.zeros((count, 1))
    snrs_with_pad = np.hstack([snr_padding, snrs_cpu])
    snr_gains = np.diff(snrs_with_pad, axis=1)
    mean_snr_gain = np.mean(snr_gains, axis=0)
    optimal_indices = np.where(mean_snr_gain >= snr_gain_thr)[0]

    U_selected = U[:, optimal_indices]
    s_selected = s[optimal_indices]
    V_selected = V[:, optimal_indices]

    denoised_data = (U_selected @ torch.diag(s_selected) @ V_selected.T).cpu().numpy()

    return denoised_data, optimal_indices, est_rank


def main():
    # --- 参数定义 ---
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / 'datasets/experimental/horiba/Helacell/HeLaCell_0.1s.mat'
    feng_wavenumber = 2932   # 成像峰的目标波数
    width = 72              # 高光谱数据宽度
    high = 97                # 高光谱数据高度

    print("正在加载数据...")
    data = loadmat(data_path)
    target = data['Noisy']  # target data, (波数, 像素)
    # target = data['cube']  # target data, (波数, 像素)
    x = data['x'].flatten()      # 波数轴
    leng = len(x)
    print("数据加载完成。")

    print("开始执行ALRMA算法...")
    s_time = time.time()
    recon, optimal_indices, est_rank = ALRMA(target, width, high, range(0, leng), 5, 100, 1e-3, 1e-2)
    e_time = time.time()
    print(f"ALRMA算法完成，耗时: {e_time - s_time}s")
    print(f"估计秩: {est_rank}, 选择的最优秩索引: {optimal_indices}")

    wave_index = np.argmin(np.abs(x - feng_wavenumber))
    print(f"目标波数 {feng_wavenumber} cm^-1 对应的索引为: {wave_index}")

    denoised_image = recon[wave_index, :].reshape((high, width))

    # 标准化以便于显示
    min_val = np.min(denoised_image)
    max_val = np.max(denoised_image)
    normalized_image = (denoised_image - min_val) / (max_val - min_val)

    # 绘制图像
    aspect_ratio = width / high
    plt.figure(figsize=(aspect_ratio * 5, 5))
    plt.imshow(normalized_image, cmap='jet', aspect='auto')
    output_dir = project_root / 'experiments/alrma'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'HeLaCell_0.1s.png')

if __name__ == '__main__':
    main()
