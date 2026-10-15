import os
import time
import h5py
import torch
import numpy as np
import argparse
import inspect
import copy
import pywt
from scipy.signal import savgol_filter
from utils.base import calculate_metrics, seed_everything

# ========================================== 
# 任何新算法只需在这里定义，保持输入第一项为 signal 即可
# ==========================================

def wtd(data, wavelet='db3', level=4):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(data))) * np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
    denoised_signal = pywt.waverec(coeffs_denoised, wavelet)[:len(data)]
    return denoised_signal

def sg(data, window_length=7, polyorder=3):
    return savgol_filter(data, window_length, polyorder)

# --- PEER 算法及其依赖 ---

def distance_Reciprocal(a, b):
    d_all = a**2 + b**2
    d = d_all**0.5
    d = 1 / d
    return d


def weight_result(data, c):
    m = (c - 1) // 2
    n = len(data)
    new_data = [0 for x in range(n)]
    for i in range(0, n):
        w_sum = 0
        if i <= m - 1:  # 左端不完整的几个点
            lx = 0
            ly = i + m
            p_sum = 0
            w_sum = w_sum + data[0] * (m - i)
            a = 0
            happens = [0 for x in range(ly - lx + 1)]
            for j in range(lx, ly + 1):
                w_sum = w_sum + data[j]
            avg = w_sum / c
            if avg in data[lx : ly + 1]:
                new_data[i] = avg
            else:
                for k in range(lx, ly + 1):
                    p_sum = p_sum + distance_Reciprocal(data[k], avg)
                for k in range(lx, ly + 1):
                    happens[a] = data[k] * (distance_Reciprocal(data[k], avg) / p_sum)
                    new_data[i] = new_data[i] + happens[a]
                    a = a + 1
        elif i >= n - m:  # 右端不完整的几个点
            lx = i - m
            ly = n - 1
            p_sum = 0
            w_sum = w_sum + data[n - 1] * (m - n + i)
            a = 0
            happens = [0 for x in range(ly - lx + 1)]
            for j in range(lx, ly + 1):
                w_sum = w_sum + data[j]
            avg = w_sum / c
            if avg in data[lx : ly + 1]:
                new_data[i] = avg
            else:
                for k in range(lx, ly + 1):
                    p_sum = p_sum + distance_Reciprocal(data[k], avg)
                for k in range(lx, ly + 1):
                    happens[a] = data[k] * (distance_Reciprocal(data[k], avg) / p_sum)
                    new_data[i] = new_data[i] + happens[a]
                    a = a + 1
        else:
            lx = i - m
            ly = i + m
            p_sum = 0
            happens = [0 for x in range(ly - lx + 1)]
            a = 0
            for j in range(lx, ly + 1):
                w_sum = w_sum + data[j]
            avg = w_sum / c
            if avg in data[lx : ly + 1]:
                new_data[i] = avg
            else:
                for k in range(lx, ly + 1):
                    p_sum = p_sum + distance_Reciprocal(data[k], avg)
                for k in range(lx, ly + 1):
                    happens[a] = data[k] * (distance_Reciprocal(data[k], avg) / p_sum)
                    new_data[i] = new_data[i] + happens[a]
                    a = a + 1
    return new_data


def derivative(data):
    data = weight_result(data, 3)  # 识别用窗口参数
    result = [data[i] - data[i - 1] for i in range(1, len(data))]
    result.insert(0, result[0])
    result2 = [result[i] - result[i - 1] for i in range(1, len(result))]
    return result, result2


def move_average(data, weight):
    k = len(weight)
    if k == 0:
        return data
    m = (k - 1) // 2
    orign = data
    n = len(orign)
    result = []
    for i in range(0, n):
        lx = 0
        ly = 0
        if i <= m - 1:  # 左端不完整的几个点
            lx = 0
            ly = i + m
        elif i >= n - m:  # 右端不完整的几个点
            lx = i - m
            ly = n
        else:
            lx = i - m
            ly = i + m
        sum_y = 0
        if i <= m - 1:
            for k in range(m - i):  # 左端缺少的点的补算
                sum_y += orign[0] * weight[k]
        elif i >= n - m:
            for k in range(m - n + i + 1):  # 右边缺少的点的补算
                sum_y += orign[n - 1] * weight[n - i + k + m]

        for j in range(lx, ly):
            sum_y += orign[j] * weight[j - i + m]
        average_y = sum_y
        result.append(average_y)
    return result


def find_extreme(data, d_data, d2_data, half_k_threshold=2):
    big = []
    small = []
    big_mis = []
    data_temp = copy.deepcopy(data)
    for i in range(len(d_data) - 1):
        if d_data[i] < 0 and d_data[i + 1] > 0:
            small.append(i)
    for i in range(len(small) - 1):
        for j in range(small[i], small[i + 1]):
            if d_data[j] > 0 and d_data[j + 1] < 0:
                big.append([[small[i], small[i + 1]], j])

    big_premis = []
    big_premis_t = []
    for s in range(len(big)):
        half_k = (big[s][0][1] - big[s][0][0]) // 2
        #####################################kye#############################################################
        if half_k >= 5:  #####重要参数，二阶导的选取比例
            half_k = half_k_threshold
        for w in range(half_k):
            if big[s][1] + w < len(d2_data) and big[s][1] - w > 0:
                if d2_data[big[s][1] + w] >= 0 or d2_data[big[s][1] - w] >= 0:
                    big_premis.append(s)
    big_premis = list(set(big_premis))
    for s in big_premis:
        big_premis_t.append(big[s])
    for s in big_premis_t:
        big.remove(s)
    for m in range(len(big) - 1):
        if data[big[m][0][1]] > data[big[m][0][0]]:
            if data[big[m][0][1]] - data[big[m][0][0]] > 0:
                list1 = []
                for f in range(big[m][1], big[m][0][1]):
                    list1.append(abs(data[f] - data[big[m][0][1]]))
                big[m][0][1] = list1.index(min(list1)) + big[m][1]
                del list1
        else:
            if data[big[m][0][0]] - data[big[m][0][1]] > 0:
                list1 = []
                for f in range(big[m][0][0], big[m][1]):
                    list1.append(abs(data[f] - data[big[m][0][1]]))
                big[m][0][0] = list1.index(min(list1)) + big[m][0][0]
                del list1

    m = 4  # m-参数，调节保留峰的宽度（反比）
    for y in range(len(big) - 1):
        q = big[y][0][1] - big[y][0][0]
        rush = q // 2 - int(q // m)
        data_temp1 = data[big[y][0][0] + rush : big[y][1] + 1]
        data_temp2 = data[big[y][1] : big[y][0][1] + 1 - rush]
        flag1 = 0
        flag2 = 0

        for h in range(len(data_temp1) - 1):
            if data_temp1[h + 1] - data_temp1[h] < 0:
                flag1 = flag1 + 1
        for x in range(len(data_temp2) - 1):
            if data_temp2[x + 1] - data_temp2[x] > 0:
                flag2 = flag2 + 1
    #        if q<5 :
    #            big_mis.append(big[y])
    for s in big_mis:
        if s in big:
            big.remove(s)

    big_new = []
    small_new = []
    for b in range(len(big)):
        small_new.append(big[b][0][0])
        small_new.append(big[b][0][1])
    for u in range(len(big)):
        big_new.append(big[u][1])
    for r in range(0, len(small_new) - 1, 2):
        q = small_new[r + 1] - small_new[r]
        for t in range(1, int(q // m) + 1):
            big_new.append(big_new[r // 2] + t)
            big_new.append(big_new[r // 2] - t)
    big_new.sort()

    big_new_searching = []
    for i in big_new:
        if i > len(data) - 2 or i < 1:
            big_new_searching.append(i)
    for i in big_new_searching:
        big_new.remove(i)
    return big_new, data_temp


def weight_resultX2(data, half_k_threshold=2, windowsize=5):  # PEER
    d_data, d2_data = derivative(data)
    big_new, data_temp = find_extreme(data, d_data, d2_data, half_k_threshold)
    new_data = weight_result(data_temp, c=windowsize)  # 窗口参数
    for l in big_new:
        new_data[l] = data[l]
    return new_data


def read(filename):
    file = open(filename, encoding="utf-8")
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


def peer(data, loops=1, half_k_threshold=0, windowsize=5):
    for _ in range(loops):
        data = weight_resultX2(data, half_k_threshold, windowsize)
    data = np.array(data)
    return data


# ==========================================
# 第二部分：算法注册表与动态 Pipeline (Registry)
# ==========================================

# 注册所有支持的去噪方法
DENOISERS = {
    'wtd': wtd,
    'sg': sg,
    'peer': peer
}

def process_pipeline(spectra, method_name, **kwargs):
    """通用的批处理去噪 Pipeline，自动过滤不支持的参数"""
    if method_name not in DENOISERS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(DENOISERS.keys())}")
    
    func = DENOISERS[method_name]
    # 动态获取当前算法函数支持的参数名
    sig = inspect.signature(func)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    
    denoised_spectra = []
    for spec in spectra:
        # 只传入该算法需要的参数
        denoised_spec = func(spec, **valid_kwargs)
        denoised_spectra.append(denoised_spec)
        
    return np.stack(denoised_spectra, axis=0)


# ==========================================
# 第三部分：主程序与命令行解析 (Main)
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Universal Denoising Pipeline")
    
    # 核心设置
    parser.add_argument('--data_dir', type=str, required=True, help="Input .h5 file or directory")
    parser.add_argument('--data_savedir', type=str, default='experiments/results')
    parser.add_argument('--method', type=str, required=True, choices=DENOISERS.keys(), help="Denoising algorithm to use")
    parser.add_argument('--seed', type=int, default=615)
    
    # 算法特定的超参数 (在此统一定义，按需增加)
    parser.add_argument('--level', type=int, default=4, help="wtd parameter")
    parser.add_argument('--wavelet', type=str, default='db3', help="wtd parameter")
    parser.add_argument('--window_length', type=int, default=7, help="sg parameter")
    parser.add_argument('--polyorder', type=int, default=3, help="sg parameter")
    
    parser.add_argument('--loops', type=int, default=5, help="peer parameter")
    parser.add_argument('--half_k_threshold', type=int, default=5, help="peer parameter")
    parser.add_argument('--windowsize', type=int, default=5, help="peer parameter")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.data_savedir, exist_ok=True)

    if os.path.isdir(args.data_dir):
        files_to_process = sorted([os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.h5') or f.endswith('.txt')])
    elif os.path.isfile(args.data_dir):
        files_to_process = [args.data_dir]
    else:
        print(f"Error: Path not found -> {args.data_dir}")
        exit(1)

    if not files_to_process:
        print("No .h5 or .txt files found to process.")
        exit(0)
        
    for current_file in files_to_process:
        print(f"\n{'='*20} Processing: {os.path.basename(current_file)} | Method: {args.method} {'='*20}")
        
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
            
        print(f"Loaded {spectra_noisy.shape[0]} spectra.")
        start_time = time.time()
        
        
        # ---------------- 核心 Pipeline 调用 ----------------
        args_dict = vars(args)
        denoised_spectra = process_pipeline(spectra_noisy, args.method, **args_dict)
        # ---------------------------------------------------
        
        file_name = os.path.basename(current_file)
        if current_file.endswith('.h5'):
            output_path = os.path.join(args.data_savedir, f"{os.path.splitext(file_name)[0]}.h5")
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('denoised', data=denoised_spectra, compression="gzip", compression_opts=4, chunks=True)
            print(f"Saved to {output_path}")
        elif current_file.endswith('.txt'):
            output_path = os.path.join(args.data_savedir, file_name)
            save_data = np.column_stack((wavenumbers, denoised_spectra[0]))
            np.savetxt(output_path, save_data, fmt='%.6f', delimiter='\t')
            print(f"Saved to {output_path}")
        
        inference_time = time.time() - start_time
        print(f"Processing Time  : {inference_time:.4f} seconds\n")
        if gt is not None:
            num_samples = gt.shape[0]
            all_metrics = np.array([calculate_metrics(gt[i], denoised_spectra[i]) for i in range(num_samples)])
            
            avg_snr = np.round(np.mean(all_metrics[:, 0]), 2)
            avg_rmse = np.round(np.mean(all_metrics[:, 1]), 4)
            avg_sss = np.round(np.mean(all_metrics[:, 2]), 4)
            
            base_name = os.path.splitext(file_name)[0]
            eval_filepath = os.path.join(args.data_savedir, f"{base_name}_eval.log")
            csv_filepath = os.path.join(args.data_savedir, f"{base_name}_metrics_details.csv")
            np.savetxt(csv_filepath, all_metrics, delimiter=',', header='SNR,RMSE,SSS', comments='')
            
            with open(eval_filepath, 'w', encoding='utf-8') as f_eval:
                f_eval.write(f"================ {args.method.upper()} Evaluation Results ================\n")
                f_eval.write(f"Test Date        : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f_eval.write(f"File Name        : {file_name}\n")
                f_eval.write(f"Num of Spectra   : {num_samples}\n")
                f_eval.write(f"Processing Time  : {inference_time:.4f} seconds\n")
                f_eval.write("SNR/dB\tRMSE\tSSS\n")
                f_eval.write(f"{avg_snr}\t{avg_rmse}\t{avg_sss}\n")
                f_eval.write("=========================================================\n")
            
            print(f"Metrics saved to: {eval_filepath}")
            print(f"Detailed metrics saved to: {csv_filepath} (For Boxplots)")
            print(f"Final Results -> SNR: {avg_snr} | RMSE: {avg_rmse} | SSS: {avg_sss}")
        else:
            print("Ground truth not found. Skipping evaluation.")
            continue
