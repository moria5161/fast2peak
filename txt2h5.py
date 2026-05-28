import os
import h5py
import numpy as np
import argparse

# ==========================================
# 核心逻辑：处理与合成
# ==========================================

def process_and_combine(input_raw_dir, temp_dir, h5_output_path):
    """
    第一阶段：从原始带头部的txt中提取数据
    第二阶段：将提取的数据合并为H5
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # 1. 提取数据到临时文件夹
    print(f"--- 阶段 1: 正在从 {input_raw_dir} 提取数据 ---")
    raw_files = sorted([f for f in os.listdir(input_raw_dir) if f.endswith('.txt')])
    
    for filename in raw_files:
        input_path = os.path.join(input_raw_dir, filename)
        output_path = os.path.join(temp_dir, filename)
        
        processed_data = []
        start_reading = False
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'Raman Shift' in line and 'Raw data' in line:
                    start_reading = True
                    continue
                if start_reading:
                    parts = line.strip().split(';')
                    if len(parts) >= 7:
                        try:
                            # 索引3: Raman Shift, 索引6: Raw data #1
                            processed_data.append([float(parts[3]), float(parts[6])])
                        except ValueError: continue
        
        if processed_data:
            np.savetxt(output_path, np.array(processed_data), fmt='%.6f', delimiter='\t')

    # 2. 合并临时txt到H5
    print(f"--- 阶段 2: 正在合成 H5 文件 -> {h5_output_path} ---")
    processed_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.txt')])
    
    all_intensities = []
    waves = None

    for filename in processed_files:
        data = np.loadtxt(os.path.join(temp_dir, filename))
        if waves is None:
            waves = data[:, 0]
        all_intensities.append(data[:, 1])

    noisy_data = np.array(all_intensities)

    with h5py.File(h5_output_path, 'w') as hf:
        hf.create_dataset('waves', data=waves, chunks=True, compression="gzip")
        hf.create_dataset('noisy', data=noisy_data, chunks=True, compression="gzip")
    
    print(f"任务完成！Noisy 形状: {noisy_data.shape}")


# ==========================================
# 核心逻辑：解码回 TXT
# ==========================================

def decode_h5_to_txt(h5_path, output_dir, data_key='denoised'):
    """
    给定H5文件，将其中的数据键（默认为denoised）和waves键提取
    并还原为多个独立的txt文件。
    如果没有 waves 键，将自动根据数据长度生成虚拟的连续序列。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- 正在从 {h5_path} 解码数据 ({data_key}) ---")
    
    with h5py.File(h5_path, 'r') as hf:
        # 只强制检查数据键是否存在
        if data_key not in hf:
            print(f"错误: H5文件中不存在数据键 '{data_key}'")
            return
        
        data_matrix = hf[data_key][:]  # 形状: batch * length
        length = data_matrix.shape[1]  # 获取序列长度
        
        # 判断并获取 waves
        if 'waves' in hf:
            waves = hf['waves'][:]
            print(f"发现 'waves' 键，使用真实序列。")
        else:
            # 自动生成虚拟的波序列 [0.0, 1.0, 2.0, ..., length-1]
            waves = np.arange(length, dtype=float)
            print(f"未发现 'waves' 键，已自动生成长度为 {length} 的虚拟序列。")

        for i in range(data_matrix.shape[0]):
            # 生成文件名，如 sample_0000.txt
            out_filename = os.path.join(output_dir, f"sample_{i:04d}.txt")
            
            # 合并 waves 和 当前行的 intensity
            # column_stack 将两个 1D 数组转为 (N, 2)
            combined = np.column_stack((waves, data_matrix[i]))
            
            np.savetxt(out_filename, combined, fmt='%.6f', delimiter='\t')
            
    print(f"解码完成！共生成 {data_matrix.shape[0]} 个文件到 {output_dir}")


# ==========================================
# 主函数入口
# ==========================================

if __name__ == "__main__":
    # --- 路径配置 ---
    RAW_DIR = '/home/slurmusers/moria/codes/fast2peak_v2/datasets/anjisuan'
    TEMP_DIR = '/home/slurmusers/moria/codes/fast2peak_v2/datasets/anjisuan2'
    H5_FILE = '/home/slurmusers/moria/codes/fast2peak_v2/smae/experiments/smae with duotai ss2048 ps16/duotai.h5'
    DECODE_DIR = '/home/slurmusers/moria/codes/fast2peak_v2/datasets/smae_duotai_results'

    # 1. 运行压缩流程 (Raw TXT -> Clean TXT -> H5)
    # process_and_combine(RAW_DIR, TEMP_DIR, H5_FILE)

    # 2. 运行解码流程示例 (H5 -> TXT)
    # 注意：这里演示解码 'noisy' 键，你可以按需改为 'denoised'
    decode_h5_to_txt(H5_FILE, DECODE_DIR, data_key='denoised')