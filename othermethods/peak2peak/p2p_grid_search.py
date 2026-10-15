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
    parser = argparse.ArgumentParser(description="Peak2Peak - Weight Decay Search")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=615)
    parser.add_argument('--max_workers', type=int, default=64)
    parser.add_argument('--length', type=int, default=1600)
    parser.add_argument('--extend-by', type=int, default=32)
    parser.add_argument('--data_aug', type=str, default='add_raman_peaks')
    parser.add_argument('--model_name', type=str, default='FCN')
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='datasets/test')
    parser.add_argument('--data_savedir', type=str, default='experiments/eval p2p/')
    # 核心修改：支持接收一个或多个 float 数值作为列表
    parser.add_argument('--wd_list', nargs='+', type=float, default=[0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25], 
                        help='List of weight decay values to search, e.g., --wd_list 0.01 0.1 1.0')
    
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
        x = np.arange(length)
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


def train_single_spec(model, args, spectrum_raw, device, current_wd, writer=None):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=current_wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.5)
    de_func = globals()[args.data_aug]
    dataset = SpectrumDataset(spectrum_raw, de_func)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()


def inference_single_spec(model, spectrum_raw, device):
    model.eval()
    spectrum = normalization(spectrum_raw) * 10
    spectrum_tensor = torch.tensor(spectrum, dtype=torch.float32).reshape(1, 1, -1).to(device)
    with torch.no_grad():
        output = model(spectrum_tensor)
    return output.cpu().numpy()[0, 0, :]


def run_single_worker(args_packet):
    idx, spectrum_noisy, spectrum_gt, args = args_packet
    torch.set_num_threads(1)
    
    weight_decays = args.wd_list
    # 【修复 1】：寻找最大值，初始值必须设为负无穷 (-inf)
    best_sss = float('-inf') 
    best_output = None
    best_metrics = None
    best_wd = None
    
    try:
        raw_length = len(spectrum_noisy)
        spectrum_resampled = spectra_resample_single(spectrum_noisy, args.length)
        max_val = np.max(spectrum_resampled) + 1e-12
        spec_norm = spectrum_resampled / max_val
        spectrum_raw_extended = extend_spectrum(spec_norm, args.extend_by)
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_id = idx % num_gpus
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device) 
            torch.backends.cudnn.benchmark = True 
        else:
            device = torch.device("cpu")

        for wd in weight_decays:
            local_seed = args.seed + idx
            seed_everything(local_seed)
            
            model = globals()[args.model_name](kernel_size=args.kernel_size).to(device)
            
            train_single_spec(model, args, spec_norm, device, current_wd=wd)
            
            output = inference_single_spec(model, spectrum_raw_extended, device)
            output = output[args.extend_by:-args.extend_by]
            spectrum_raw_crop = spectrum_raw_extended[args.extend_by:-args.extend_by]
            
            output_corrected = mean_center_correction(spectrum_raw_crop, output)
            output_corrected = spectra_resample_single(output_corrected, raw_length)
            
            # 【修复 2】：纠正传参顺序，前面放降噪结果，后面放真值(GT)
            metrics = calculate_metrics(output_corrected, spectrum_gt)
            current_sss = metrics[2]
            
            if current_sss > best_sss:
                best_sss = current_sss
                best_output = output_corrected
                best_metrics = metrics
                best_wd = wd
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return idx, best_output, best_metrics, best_wd, None

    except Exception as e:
        return idx, None, None, None, str(e)


def main():
    args = parse_args()
    
    if os.path.isdir(args.data_dir):
        files_to_process = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.h5')]
        files_to_process.sort()
    elif os.path.isfile(args.data_dir):
        files_to_process = [args.data_dir]
    else:
        print(f"Error: Path not found -> {args.data_dir}")
        return

    if not files_to_process:
        print("No .h5 files found to process.")
        return

    os.makedirs(args.data_savedir, exist_ok=True)
    
    if torch.cuda.is_available():
        print(f"\n[System Info] Detected {torch.cuda.device_count()} GPUs. Distributing {args.max_workers} workers evenly...")

    for current_file in files_to_process:
        print(f"\n{'='*20} Processing & Searching: {os.path.basename(current_file)} {'='*20}")
        
        with h5py.File(current_file, 'r') as f:
            spectra_noisy = f['noisy'][:].astype(np.float32)
            if 'gt' not in f:
                print(f"Error: No 'gt' found in {current_file}. GT is required for parameter search based on RMSE. Skipping.")
                continue
            gt = f['gt'][:].astype(np.float32)

        if isinstance(spectra_noisy, torch.Tensor):
            spectra_noisy = spectra_noisy.cpu().numpy()
            gt = gt.cpu().numpy()
            
        num_samples = spectra_noisy.shape[0]    
        print(f"Loaded {num_samples} spectra. Commencing parameter search {args.wd_list} per spectrum...")
        
        results_array = np.zeros((num_samples, spectra_noisy.shape[1]), dtype=np.float32)
        all_metrics = np.zeros((num_samples, 3), dtype=np.float32)
        best_wds = np.zeros(num_samples, dtype=np.float32)
        
        tasks = [(i, spectra_noisy[i], gt[i], args) for i in range(num_samples)]
        
        start_time = datetime.now()
        with mp.Pool(processes=args.max_workers) as pool:
            for idx, b_output, b_metrics, b_wd, error in tqdm(pool.imap_unordered(run_single_worker, tasks), total=len(tasks)):
                if error:
                    print(f"[Error] Spectrum {idx} failed: {error}")
                else:
                    results_array[idx] = b_output
                    all_metrics[idx] = b_metrics
                    best_wds[idx] = b_wd
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds()

        file_name = os.path.basename(current_file)
        
        output_filename = os.path.join(args.data_savedir, file_name)
        with h5py.File(output_filename, 'w') as f:
            f.create_dataset('denoised', data=results_array, compression="gzip", compression_opts=4, chunks=True)
        print(f"Saved optimal denoised spectra to {output_filename}")

        base_name = os.path.splitext(file_name)[0]
        csv_filepath = os.path.join(args.data_savedir, f"{base_name}_metrics_details.csv")
        csv_data = np.hstack((all_metrics, best_wds.reshape(-1, 1)))
        np.savetxt(csv_filepath, csv_data, delimiter=',', header='SNR,RMSE,SSS,Best_WD', comments='')
        
        avg_snr = np.round(np.mean(all_metrics[:, 0]), 2)
        avg_rmse = np.round(np.mean(all_metrics[:, 1]), 4)
        avg_sss = np.round(np.mean(all_metrics[:, 2]), 4)
        
        eval_filepath = os.path.join(args.data_savedir, f"{base_name}_eval.log")
        with open(eval_filepath, 'w', encoding='utf-8') as f_eval:
            f_eval.write(f"================ {args.model_name} Optimal Search Results ================\n")
            f_eval.write(f"Test Date        : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_eval.write(f"File Name        : {file_name}\n")
            f_eval.write(f"Num of Spectra   : {num_samples}\n")
            f_eval.write(f"Search Space (WD): {args.wd_list}\n")
            f_eval.write(f"Total Time       : {inference_time:.2f} seconds\n")
            f_eval.write("SNR/dB\tRMSE\tSSS\n")
            f_eval.write(f"{avg_snr}\t{avg_rmse}\t{avg_sss}\n")
            f_eval.write("=========================================================\n")
            
            unique, counts = np.unique(best_wds, return_counts=True)
            f_eval.write("\nWeight Decay Hit Frequency:\n")
            for u, c in zip(unique, counts):
                f_eval.write(f"WD = {u:.4f} : {c} spectra\n")

        print(f"Metrics saved to: {eval_filepath}")
        print(f"Final Results -> SNR: {avg_snr} | RMSE: {avg_rmse} | SSS: {avg_sss}")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
