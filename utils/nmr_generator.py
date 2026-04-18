import h5py
import numpy as np
import math
import os
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# ==========================================
# 全局配置
# ==========================================
CONFIG = {
    'simulation': {
        'num_samples': 10000,      # 生成样本总数
        'length': 16000,            # 光谱长度
        'ppm_range': (0, 50),       # NMR ppm 范围
    },
    'noise': {
        'type': 'gaussian',
        'gaussian_range': (1e-4, 3e-3), # 高斯噪声强度范围 (可根据需要微调)
    },
    'system': {
        'num_cores': 24,            # 并行核心数
        'save_path': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5161-zhangnannan/fast2peak/datasets/nmr/train.h5'
    }
}

class NMRSpectraGenerator:
    def pseudo_voigt(self, position, fwhm, length):
        """生成单个尖锐的 Pseudo-Voigt 峰 (NMR通常洛伦兹成分偏高)"""
        x = np.linspace(0, length - 1, length)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gaussian = (1 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-(x - position) ** 2 / (2 * sigma**2)))
        lorentzian = (1 / math.pi) * (fwhm / 2) / ((x - position) ** 2 + (fwhm / 2) ** 2)
        
        alpha = 0.85 # 洛伦兹权重，NMR峰尾部通常较长
        pv_curve = (1 - alpha) * gaussian + alpha * lorentzian
        
        pv_min, pv_max = np.min(pv_curve), np.max(pv_curve)
        if pv_max - pv_min == 0: return pv_curve
        return (pv_curve - pv_min) / (pv_max - pv_min)

    def generate_dataset(self, num_samples, length, cfg_noise):
        spec_gt = np.zeros((num_samples, length))
        
        for idx in range(num_samples):
            n_peaks = np.random.randint(10, 21) # 10-20个峰
            
            # 位置避开边缘
            pos_arr = np.random.randint(200, length - 200, size=n_peaks)
            # FWHM 极小，保证峰尖锐 (相较于16000的长度，3~15个点非常窄)
            fwhm_arr = np.random.uniform(3, 15, size=n_peaks) 
            # 使用指数分布生成强弱峰 (大量弱峰，少数强峰)
            int_arr = np.random.exponential(scale=20, size=n_peaks) + 2 

            for i in range(n_peaks):
                spec_gt[idx, :] += self.pseudo_voigt(pos_arr[i], fwhm_arr[i], length) * int_arr[i]

        # 归一化 GT
        max_vals = np.max(spec_gt, axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1 # 防止除以 0
        spec_gt = spec_gt / max_vals

        # 仅添加高斯噪声
        levels = np.random.uniform(*cfg_noise['gaussian_range'], size=num_samples)
        noise = np.random.normal(0, 1, spec_gt.shape) * levels[:, np.newaxis]
        spec_noisy = spec_gt + noise

        return spec_gt, spec_noisy

def worker_task(chunk_size, length, cfg_noise):
    np.random.seed() # 确保子进程随机数种子不同
    gen = NMRSpectraGenerator()
    return gen.generate_dataset(chunk_size, length, cfg_noise)

def main():
    print("=== 开始生成 NMR 模拟数据 ===")
    t0 = time.time()
    
    total_samples = CONFIG['simulation']['num_samples']
    length = CONFIG['simulation']['length']
    ppm = np.linspace(*CONFIG['simulation']['ppm_range'], length)
    n_cores = min(CONFIG['system']['num_cores'], cpu_count())
    
    # 任务分配
    chunk_size = total_samples // n_cores
    remainder = total_samples % n_cores
    tasks = [chunk_size + 1 if i < remainder else chunk_size for i in range(n_cores)]
    
    worker_func = partial(worker_task, length=length, cfg_noise=CONFIG['noise'])
    
    with Pool(n_cores) as pool:
        results = pool.map(worker_func, tasks)

    print("数据生成完毕，正在合并...")
    gt_list, noisy_list = zip(*results)
    
    # 注意：10万条 x 16000点 占用内存较大 (约 25GB)，确保你的服务器有充足 RAM
    data_gt = np.concatenate(gt_list, axis=0)
    data_noisy = np.concatenate(noisy_list, axis=0)

    save_path = CONFIG['system']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"正在写入 H5 文件至: {save_path}")
    with h5py.File(save_path, 'w') as f:
        # 为了与你给出的代码命名对应，X轴仍保存为 wavenumbers，实际指代 ppm
        f.create_dataset('noisy', data=data_noisy, compression="gzip", compression_opts=4, chunks=True)
        f.create_dataset('gt', data=data_gt, compression="gzip", compression_opts=4, chunks=True)
        f.create_dataset('wavenumbers', data=ppm, compression="gzip", compression_opts=4, chunks=True)

    print(f"完成！耗时: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()