from __future__ import print_function, division
import torch.nn as nn
import torch
import numpy as np
import torch
import torch.optim
from torch.autograd import Variable
import os
import random

from scipy.stats import norm
import time



class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.up = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=1, padding='same', bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class FCN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(FCN, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]
        self.Conv1 = conv_block(in_ch, filters[2])
        self.Conv2 = conv_block(filters[2], filters[2])
        self.Conv3 = conv_block(filters[2], filters[2])
        # self.Conv4 = conv_block(filters[2], filters[2])
        # self.Conv5 = conv_block(filters[2], filters[2])

        self.Conv6 = nn.Conv1d(filters[2], out_ch, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Conv2(e1)
        e3 = self.Conv3(e2)
        # e4 = self.Conv4(e3)
        # e5 = self.Conv5(e4)
        e6 = self.Conv6(e3)
        return e6

def F_CN():
    return FCN(1, 1)



def add_Gau_peaks(spectrum, a, b, c):#a=10,b=40,c=0.5
    spectrum = normalization(spectrum)
    lenth = len(spectrum)
    y_tot = [0] * lenth
    for g in range(random.randint(1, a)):
        a1 = random.randint(0, lenth)
        b1 = random.randint(1, lenth // b)
        keys_ = range(lenth)
        c1 = random.uniform(0, c)
        gauss = norm(loc=a1, scale=b1)
        y = gauss.pdf(keys_)
        y = normalization(y)
        y = y * c1
        y_tot = y_tot + y
    spectrum = spectrum + y_tot
    spectrum = normalization(spectrum)
    spectrum = np.array(spectrum)
    return spectrum

def read(filename):
    file = open(filename, encoding='utf-8')
    data_lines = file.readlines()
    file.close
    orign_keys = []
    orign_values = []
    for data_line in data_lines:
        pair = data_line.split()
        key = float(pair[0])
        value = float(pair[1])
        orign_keys.append(key)
        orign_values.append(value)
    return orign_keys, orign_values

def write(filename, files, values):
    file = open(filename, 'w')
    for k, v in zip(files, values):
        file.write(str(k) + " " + str(v) + "\n")
    file.close()

def normalization(data):
    #0-1 normalization
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
    ## max=1 normalization
    # data = data/np.max(data)
    # return data

def data_process(spectrum_name):
    global x1
    global x2
    global cycler
    keys, spectrum = read(spectrum_name)
    spectrum_true_raw = spectrum
    x1 = np.mean(spectrum[:len(spectrum) // 2])
    x2 = np.mean(spectrum[len(spectrum) // 2:])
    spectrum = list(spectrum)
    start = spectrum[0]
    end = spectrum[len(spectrum) - 1]
    cycler = 5
    for i in range(cycler):
        spectrum.insert(0, start)
        spectrum.append(end)
    return keys, spectrum, spectrum_true_raw

def train(model, device, optimizer, epoch, spectrum_raw):
    model.train()
    sum_loss = 0
    sum_num = 0
    GS_peak_intensity = 0.45
    for i in range(100):
        sum_num = sum_num+1
        data = add_Gau_peaks(spectrum_raw, 10, 40, GS_peak_intensity)
        data = torch.as_tensor(data, dtype=torch.float32)
        data = data.reshape(1, data.shape[0])
        data = data.reshape(1, data.shape[0], data.shape[1])#data_type: batch_size x embedding_size x text_len
        data = data.permute(1, 0, 2)
        target = add_Gau_peaks(spectrum_raw, 10, 40, GS_peak_intensity)
        target = torch.as_tensor(target, dtype=torch.float32)
        target = target.reshape(1, target.shape[0])
        target = target.reshape(1, target.shape[0], target.shape[1])  # data_type: batch_size x embedding_size x text_len
        target = target.permute(1, 0, 2)
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
    # ave_loss = sum_loss / sum_num
    # print('epoch:{},loss:{}, enhance_num:{}'.format(epoch, ave_loss, sum_num))


def test(model, device, keys, spectrum, epoch, out_path, criterion):
    global spectrum_true_raw
    global cycler
    model.eval()
    spectrum = Variable(spectrum).to(device)
    output = model(spectrum)
    loss = criterion(output, spectrum)
    print_loss = loss.data.item()
    pl_output = np.array(output.cpu().detach().numpy()[0, 0, :])
    pl_spectrum = np.array(spectrum.cpu().detach().numpy()[0, 0, :])
    for i in range(cycler):
        pl_output = np.delete(pl_output, 0, axis=None)
        pl_output = np.delete(pl_output, len(pl_output) - 1, axis=None)
    for i in range(cycler):
        pl_spectrum = np.delete(pl_spectrum, 0, axis=None)
        pl_spectrum = np.delete(pl_spectrum, len(pl_spectrum) - 1, axis=None)
    spectrum = pl_spectrum/max(pl_spectrum)
    output = pl_output/max(pl_output)
    y1 = np.mean(output[:len(spectrum)//2])
    y2 = np.mean(output[len(spectrum)//2:])
    
    a = (y1-y2)/(x1-x2) if (x1-x2) != 0 else 1.0
    b = (x1*y2-x2*y1)/(y1-y2) if (y1-y2) != 0 else 0.0
    output_corrected = (output/a)-b
    if out_path:
        write(out_path, keys, output_corrected)
    return output_corrected

def adjust_learning_rate(optimizer, epoch, modellr):
    modellrnew = modellr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew

import argparse

import h5py
from utils.base import calculate_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="P2P Raw processing script")
    parser.add_argument('--input_dir', type=str, required=True, help="输入包含txt文件的目录")
    parser.add_argument('--output_dir', type=str, required=True, help="输出保存txt文件的目录")
    parser.add_argument('--model_name', type=str, default='P2P', help="模型名称")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Setting Global Parameters
    modellr = 1e-3
    EPOCHS = 40
    BATCH_SIZE = 1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()#loss_function

    files = [f for f in os.listdir(args.input_dir) if f.endswith(('.txt', '.h5'))]
    for filename in files:
        begin_time = time.perf_counter()
        file_path = os.path.join(args.input_dir, filename)
        file_path = file_path.replace('\\', '/')
        print('Begin: ' + file_path)
        
        if filename.endswith('.txt'):
            model = F_CN()
            model.to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=modellr, weight_decay=1)
            
            keys, spectrum, spectrum_true_raw = data_process(file_path)
            
            spectrum_raw = spectrum
            spectrum_raw = normalization(spectrum_raw)
            spectrum = normalization(spectrum) * 2
            
            spectrum = torch.tensor(spectrum)
            spectrum = spectrum.reshape(1, spectrum.shape[0])
            spectrum = spectrum.reshape(1, spectrum.shape[0], spectrum.shape[1])
            spectrum = torch.as_tensor(spectrum, dtype=torch.float32)
            spectrum = spectrum.permute(1, 0, 2)
            
            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(args.output_dir, f"{base_name}_P2P.txt")
            
            for epoch in range(1, EPOCHS + 1):
                adjust_learning_rate(optimizer, epoch, modellr)
                train(model, DEVICE, optimizer, epoch, spectrum_raw)
            
            test(model, DEVICE, keys, spectrum, epoch, out_path, criterion)
            end_time = time.perf_counter()
            print(f"time consumed for {filename}: {end_time - begin_time:0.4f} seconds")
            del model

        elif filename.lower().endswith('.h5'):
            with h5py.File(file_path, 'r') as f:
                spectra_noisy = f['noisy'][:].astype(np.float32)
                gt = f['gt'][:].astype(np.float32) if 'gt' in f else None
            
            num_samples = spectra_noisy.shape[0]
            denoised_corrected = np.zeros_like(spectra_noisy)
            
            # Use keys implicitly (e.g. 0 to length-1) since h5 might not store keys linearly like txt does,
            # or we generate fake keys for test function if it needs it.
            keys = np.arange(spectra_noisy.shape[1])
            
            for i in range(num_samples):
                model = F_CN()
                model.to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=modellr, weight_decay=1)
                
                # emulate data_process behavior for single spectrum
                spectrum_numpy = spectra_noisy[i]
                
                # set globals that data_process sets used inside test
                global x1, x2, cycler
                x1 = np.mean(spectrum_numpy[:len(spectrum_numpy) // 2])
                x2 = np.mean(spectrum_numpy[len(spectrum_numpy) // 2:])
                cycler = 5
                
                # pad spectrum_numpy based on cycler
                padded_spectrum = list(spectrum_numpy)
                start_val = padded_spectrum[0]
                end_val = padded_spectrum[-1]
                for _ in range(cycler):
                    padded_spectrum.insert(0, start_val)
                    padded_spectrum.append(end_val)
                padded_spectrum = np.array(padded_spectrum)
                
                spectrum_raw = padded_spectrum
                spectrum_raw = normalization(spectrum_raw)
                spectrum_tnsr = normalization(padded_spectrum) * 2
                
                spectrum_tnsr = torch.tensor(spectrum_tnsr)
                spectrum_tnsr = spectrum_tnsr.reshape(1, spectrum_tnsr.shape[0])
                spectrum_tnsr = spectrum_tnsr.reshape(1, spectrum_tnsr.shape[0], spectrum_tnsr.shape[1])
                spectrum_tnsr = torch.as_tensor(spectrum_tnsr, dtype=torch.float32)
                spectrum_tnsr = spectrum_tnsr.permute(1, 0, 2)
                
                for epoch in range(1, EPOCHS + 1):
                    adjust_learning_rate(optimizer, epoch, modellr)
                    train(model, DEVICE, optimizer, epoch, spectrum_raw)
                
                output_corrected = test(model, DEVICE, keys, spectrum_tnsr, epoch, None, criterion)
                denoised_corrected[i] = output_corrected
                del model
                
            end_time = time.perf_counter()
            inference_time = end_time - begin_time
            print(f"time consumed for {filename}: {inference_time:0.4f} seconds")

            output_filename = os.path.join(args.output_dir, filename)
            with h5py.File(output_filename, 'w') as f_out:
                f_out.create_dataset('denoised', data=denoised_corrected)
                if gt is not None:
                    f_out.create_dataset('gt', data=gt)

            if gt is not None:
                all_metrics = np.array([calculate_metrics(gt[i], denoised_corrected[i]) for i in range(num_samples)])
                
                avg_snr = np.round(np.mean(all_metrics[:, 0]), 2)
                avg_rmse = np.round(np.mean(all_metrics[:, 1]), 4)
                avg_sss = np.round(np.mean(all_metrics[:, 2]), 4)
                
                base_name = os.path.splitext(filename)[0]
                eval_filepath = os.path.join(args.output_dir, f"{base_name}_eval.log")
                csv_filepath = os.path.join(args.output_dir, f"{base_name}_metrics_details.csv")
                np.savetxt(csv_filepath, all_metrics, delimiter=',', header='SNR,RMSE,SSS', comments='')
                
                with open(eval_filepath, 'w', encoding='utf-8') as f_eval:
                    f_eval.write(f"================ {args.model_name} Evaluation Results ================\n")
                    f_eval.write(f"Test Date        : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f_eval.write(f"File Name        : {filename}\n")
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





