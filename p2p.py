from __future__ import print_function, division
import argparse
import os
import h5py
import time
import numpy as np
import math
import random
import numpy.linalg as la
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.base import seed_everything, spectra_resample_single, calculate_metrics
import torch.multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser(description="Peak2Peak")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=615)
    parser.add_argument('--max_workers', type=int, default=10)
    parser.add_argument('--length', type=int, default=1600)
    parser.add_argument('--extend-by', type=int, default=32)
    parser.add_argument('--data_aug', type=str, default='add_raman_peaks')
    parser.add_argument('--model_name', type=str, default='FCN')
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--data_savedir', type=str, default='./data_denoised')
    return parser.parse_args()


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super(conv_block, self).__init__()
        self.up = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding='same', bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.up(x)


class FCN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, kernel_size=7):
        super(FCN, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]
        self.Conv1 = conv_block(in_ch, filters[2], kernel_size=kernel_size)
        self.Conv2 = conv_block(filters[2], filters[2], kernel_size=kernel_size)
        self.Conv3 = conv_block(filters[2], filters[2], kernel_size=kernel_size)
        self.Conv6 = nn.Conv1d(filters[2], out_ch, kernel_size=1, stride=1, padding='same')
    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Conv2(e1)
        e3 = self.Conv3(e2)
        e6 = self.Conv6(e3)
        return e6


def add_raman_peaks(spectrum):

    length = len(spectrum)
    fwhm_range = (5, 100)
    position_range = (10, length - 10)
    intensity_range = (0.1, 0.5)
    
    def pseudo_voigt(position, fwhm, length, intensity):
        # 修复1：使用精确的整数长度，确保维度绝对一致
        x = np.arange(length)
        
        # 修复2：直接构造最高点为 1 的峰，避免 min-max 归一化破坏长尾
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gaussian = np.exp(-(x - position)**2 / (2 * sigma**2))
        
        gamma = fwhm / 2
        lorentzian = 1 / (1 + ((x - position) / gamma)**2)
        
        alpha = 0.6785
        pseudo_voigt_curve = (1 - alpha) * gaussian + alpha * lorentzian
        
        return pseudo_voigt_curve * intensity

    modified_spectrum = np.copy(spectrum)
    num_peaks = random.randint(3, 9)
    
    for _ in range(num_peaks):
        position = random.randint(position_range[0], position_range[1])
        fwhm = random.randint(fwhm_range[0], fwhm_range[1])
        intensity = random.uniform(intensity_range[0], intensity_range[1])
        
        raman_peak = pseudo_voigt(position, fwhm, length, intensity)
        modified_spectrum += raman_peak
          
    return modified_spectrum


def normalization(data):
    return data / np.max(data)


def extend_spectrum(spectrum, extend_by=5):
    left_extension = np.full(extend_by, spectrum[0])
    right_extension = np.full(extend_by, spectrum[-1])
    return np.concatenate([left_extension, spectrum, right_extension])


def least_squares_correction(raw_spec, denoised_spec):
    A = np.vstack([denoised_spec, np.ones(len(denoised_spec))]).T
    a, b = la.lstsq(A, raw_spec, rcond=None)[0]
    return a * denoised_spec + b


def mean_center_correction(raw_spec, denoised_spec):
    S_corrected = least_squares_correction(raw_spec, denoised_spec)
    mean_shift = np.mean(raw_spec) - np.mean(S_corrected)
    return S_corrected + mean_shift


class SpectrumDataset(Dataset):
    def __init__(self, spectrum_raw, de_func, num_samples=100):
        self.data_aug_list1 = [de_func(spectrum_raw) for _ in range(num_samples)]
        self.data_aug_list2 = [de_func(spectrum_raw) for _ in range(num_samples)]
    def __len__(self):
        return len(self.data_aug_list1)
    def __getitem__(self, idx):
        data1 = torch.tensor(self.data_aug_list1[idx], dtype=torch.float32).unsqueeze(0)
        data2 = torch.tensor(self.data_aug_list2[idx], dtype=torch.float32).unsqueeze(0)
        return data1, data2


def train_single_spec(model, args, spectrum_raw, device, writer=None):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.5)
    de_func = globals()[args.data_aug]
    dataset = SpectrumDataset(spectrum_raw, de_func)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = next(model.parameters()).device
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if writer:
            writer.add_scalar('Loss/train', epoch_loss/len(dataloader), epoch)
            writer.add_scalar('Learning Rate', current_lr, epoch)


def inference_single_spec(model, spectrum_raw, device):
    model.eval()
    device = next(model.parameters()).device
    spectrum = normalization(spectrum_raw) * 10
    spectrum_tensor = torch.tensor(spectrum, dtype=torch.float32).reshape(1, 1, -1).to(device)
    with torch.no_grad():
        output = model(spectrum_tensor)
    return output.cpu().numpy()[0, 0, :]


def run_single_worker(args_packet):
    idx, spectrum, args = args_packet
    torch.set_num_threads(1)
    local_seed = args.seed + idx
    seed_everything(local_seed)
    try:
        raw_length = len(spectrum)
        spectrum_resampled = spectra_resample_single(spectrum, args.length)
        max_val = np.max(spectrum_resampled) + 1e-12
        spec_norm = spectrum_resampled / max_val
        spectrum_raw = extend_spectrum(spec_norm, args.extend_by)
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_id = idx % num_gpus
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
        model = globals()[args.model_name](kernel_size=args.kernel_size)
        model.to(device)
        train_single_spec(model, args, spec_norm, device, writer=None)
        output = inference_single_spec(model, spectrum_raw, device)
        output = output[args.extend_by:-args.extend_by]
        spectrum_raw = spectrum_raw[args.extend_by:-args.extend_by]
        output_corrected = mean_center_correction(spectrum_raw, output)
        output_corrected = spectra_resample_single(output_corrected, raw_length)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            time.sleep(0.5)
        return idx, output_corrected, None
    except Exception as e:
        return idx, None, str(e)


def main():
    args = parse_args()
    
    # --- 修改部分：支持文件夹读取 ---
    if os.path.isdir(args.data_dir):
        # 读取文件夹下所有 .h5 或 .txt 文件
        files_to_process = [os.path.join(args.data_dir, f) 
                            for f in os.listdir(args.data_dir) 
                            if f.endswith('.h5') or f.endswith('.txt')]
        files_to_process.sort()
    elif os.path.isfile(args.data_dir):
        files_to_process = [args.data_dir]
    else:
        print(f"Error: Path not found -> {args.data_dir}")
        return

    if not files_to_process:
        print("No .h5 or .txt files found to process.")
        return

    os.makedirs(args.data_savedir, exist_ok=True)

    # 开始循环处理文件
    for current_file in files_to_process:
        print(f"\n{'='*20} Processing: {os.path.basename(current_file)} {'='*20}")
        
        wavenumbers = None
        if current_file.endswith('.h5'):
            with h5py.File(current_file, 'r') as f:
                spectra_noisy = f['noisy'][:].astype(np.float32)
                gt = f['gt'][:].astype(np.float32) if 'gt' in f else None
        elif current_file.endswith('.txt'):
            try:
                data = np.loadtxt(current_file)
                if data.ndim == 2:
                    wavenumbers = data[:, 0]
                    spectra_noisy = data[:, 1].astype(np.float32).reshape(1, -1)
                elif data.ndim == 1:
                    wavenumbers = np.arange(len(data))
                    spectra_noisy = data.astype(np.float32).reshape(1, -1)
                else:
                    print(f"Skipping {current_file}: unsupported format.")
                    continue
                gt = None
            except Exception as e:
                print(f"Error loading {current_file}: {e}")
                continue
        else:
            continue

        if isinstance(spectra_noisy, torch.Tensor):
            spectra_noisy = spectra_noisy.cpu().numpy()
        num_samples = spectra_noisy.shape[0]    
        print(f"Loaded {num_samples} spectra.")
        
        results_array = np.zeros((spectra_noisy.shape[0], spectra_noisy.shape[1]), dtype=np.float32)
        tasks = [(i, spectra_noisy[i], args) for i in range(num_samples)]
        
        start_time = datetime.now()
        with mp.Pool(processes=args.max_workers) as pool:
            for idx, result, error in tqdm(pool.imap_unordered(run_single_worker, tasks), total=len(tasks)):
                if error:
                    print(f"[Error] Spectrum {idx} failed: {error}")
                    results_array[idx] = np.zeros(spectra_noisy.shape[1])
                else:
                    results_array[idx] = result
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds()

        # 保存结果
        file_name = os.path.basename(current_file)
        output_filename = os.path.join(args.data_savedir, file_name)
        if current_file.endswith('.h5'):
            with h5py.File(output_filename, 'w') as f:
                f.create_dataset('denoised', data=results_array, compression="gzip", compression_opts=4, chunks=True)
            print(f"Saved to {output_filename}")
        elif current_file.endswith('.txt'):
            save_data = np.column_stack((wavenumbers, results_array[0]))
            np.savetxt(output_filename, save_data, fmt='%.6f', delimiter='\t')
            print(f"Saved to {output_filename}")

        # 指标评估
        if gt is not None:
            num_samples = gt.shape[0]
            all_metrics = np.array([calculate_metrics(gt[i], results_array[i]) for i in range(num_samples)])
            
            avg_snr = np.round(np.mean(all_metrics[:, 0]), 2)
            avg_rmse = np.round(np.mean(all_metrics[:, 1]), 4)
            avg_sss = np.round(np.mean(all_metrics[:, 2]), 4)
            
            base_name = os.path.splitext(file_name)[0]
            eval_filepath = os.path.join(args.data_savedir, f"{base_name}_eval.log")
            csv_filepath = os.path.join(args.data_savedir, f"{base_name}_metrics_details.csv")
            np.savetxt(csv_filepath, all_metrics, delimiter=',', header='SNR,RMSE,SSS', comments='')
            
            with open(eval_filepath, 'w', encoding='utf-8') as f_eval:
                f_eval.write(f"================ {args.model_name} Evaluation Results ================\n")
                f_eval.write(f"Test Date        : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f_eval.write(f"File Name        : {file_name}\n")
                f_eval.write(f"Num of Spectra   : {num_samples}\n")
                f_eval.write(f"Processing Time  : {inference_time:.2f} seconds\n")
                f_eval.write("SNR/dB\tRMSE\tSSS\n")
                f_eval.write(f"{avg_snr}\t{avg_rmse}\t{avg_sss}\n")
                f_eval.write("=========================================================\n")
            
            print(f"Metrics saved to: {eval_filepath}")
            print(f"Detailed metrics saved to: {csv_filepath} (For Boxplots)")
            print(f"Final Results -> SNR: {avg_snr} | RMSE: {avg_rmse} | SSS: {avg_sss}")
        else:
            print("Ground truth not found. Skipping evaluation.")

            continue


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()