import random
import time
import h5py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import OtherMethods.smae.model.Spectra_MAE as spectra_mae

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def show_spectra(y):
    plt.figure(figsize=(5, 3), dpi=100)
    plt.plot(y)
    plt.show()

def prepare_model(chkpt_dir, arch='spectraMAE_base_patch100'):
    model = getattr(spectra_mae, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint)
    model = model.cuda()
    print(msg)
    return model

def model_inference(y, model_mae, mask_ratio=0.50):
    spectra_tensor = torch.from_numpy(y).view(1, -1).float()
    spectra_tensor = spectra_tensor.view(1, 1, spectra_tensor.shape[-1])
    loss, pred, mask = model_mae(spectra_tensor, mask_ratio=mask_ratio)
    pred = model_mae.unpatchify(pred)
    pred = pred.detach().numpy()
    pred = pred.reshape(spectra_tensor.shape[-1], )
    return pred

def model_inference_batch(y, model_mae, batch_size=64, mask_ratio=0.50):
    preds = []
    device = next(model_mae.parameters()).device
    spectra_tensor = torch.from_numpy(y).float().to(device)
    
    if spectra_tensor.dim() == 2:
        spectra_tensor = spectra_tensor.unsqueeze(1) 
    for i in range(0, spectra_tensor.shape[0], batch_size):
        batch = spectra_tensor[i:i+batch_size]
        loss, pred, mask = model_mae(batch, mask_ratio=mask_ratio)
        pred = model_mae.unpatchify(pred)
        pred = pred.detach().cpu().numpy()
        pred = pred.squeeze(1)
        preds.append(pred)
    
    return np.vstack(preds)

def calculate_metrics(spec_denoised, spec_gt):
    x = np.array(spec_denoised)
    y = np.array(spec_gt)
    
    rmse = np.sqrt(np.mean((x - y) ** 2))
    snr = 10 * np.log10(np.sum(y ** 2) / (np.sum((x - y) ** 2) + 1e-8))

    inner_product = np.dot(x, y)
    norms = np.linalg.norm(x) * np.linalg.norm(y)
    sad = np.arccos(np.clip(inner_product / norms, -1.0, 1.0))

    pcc = np.corrcoef(x, y)[0, 1]
    
    # 按照此前记录的首选计算公式：SSS = (1 - SAD) * PCC
    sss = (1-sad) * pcc
    
    return snr, rmse, sss

def test(
    input_path: str,
    output_dir: str,
    chkpt_dir: str,
    arch: str = 'spectraMAE_base_ss1600_ps2',
    batch_size: int = 64,
    seed: int = 615,
    mask_ratio: float = 0.50
):
    """
    加载数据并执行模型批量推理，支持处理目录或单文件，保存结果并输出评估日志 (Log/CSV)。
    """
    # 1. 初始化
    seed_everything(seed)

    # 2. 获取要处理的文件列表
    if os.path.isdir(input_path):
        files_to_process = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.h5', '.txt'))])
    elif os.path.isfile(input_path):
        files_to_process = [input_path]
    else:
        raise ValueError(f"输入路径不存在: {input_path}")

    if not files_to_process:
        print("未找到 .h5 或 .txt 文件。")
        return None

    # 3. 准备模型 (只加载一次)
    print(f"正在加载模型 (架构: {arch}) \n权重路径: {chkpt_dir}")
    model_mae = prepare_model(chkpt_dir=chkpt_dir, arch=arch)
    model_mae.eval()

    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存至目录: {output_dir}\n")

    # 4. 遍历处理每个文件
    for current_file in files_to_process:
        print(f"--- 开始处理: {current_file} ---")
        wavenumbers = None
        gt = None
        
        # 读取数据
        if current_file.endswith('.h5'):
            with h5py.File(current_file, 'r') as f:
                y = f['noisy'][:]
                gt = f['gt'][:] if 'gt' in f else None
        elif current_file.endswith('.txt'):
            data = np.loadtxt(current_file)
            if data.ndim == 2:
                wavenumbers = data[:, 0]
                y = data[:, 1].astype(np.float32).reshape(1, -1)
            elif data.ndim == 1:
                wavenumbers = np.arange(len(data))
                y = data.astype(np.float32).reshape(1, -1)
            else:
                print(f"警告: 不受支持的 txt 数据维度: {data.ndim}，已跳过。")
                continue
        
        print(f"输入数据形状: {y.shape}")

        # 执行推理
        time_start = time.time()
        pred = model_inference_batch(y, model_mae, batch_size=batch_size, mask_ratio=mask_ratio)
        time_end = time.time()
        inference_time = time_end - time_start
        print(f"推理耗时: {inference_time:.4f} 秒")

        # 保存结果
        file_name = os.path.basename(current_file)
        
        # 注意: 这里使用原文件名作为输出文件的名字
        out_file = os.path.join(output_dir, file_name)

        if current_file.endswith('.h5'):
            with h5py.File(out_file, 'w') as f:
                f.create_dataset('denoised', data=pred)
        elif current_file.endswith('.txt'):
            save_data = np.column_stack((wavenumbers, pred[0])) if wavenumbers is not None else pred[0]
            np.savetxt(out_file, save_data, fmt='%.6f', delimiter='\t')
        
        print(f"已保存结果至: {out_file}")

        # 5. 计算评估指标
        if gt is not None:
            num_samples = y.shape[0]
            all_metrics = np.array([calculate_metrics(pred[i], gt[i]) for i in range(num_samples)])

            avg_snr = np.round(np.mean(all_metrics[:, 0]), 2)
            avg_rmse = np.round(np.mean(all_metrics[:, 1]), 4)
            avg_sss = np.round(np.mean(all_metrics[:, 2]), 4)

            base_name = os.path.splitext(file_name)[0]
            eval_filepath = os.path.join(output_dir, f"{base_name}_eval.log")
            csv_filepath = os.path.join(output_dir, f"{base_name}_metrics_details.csv")
            
            np.savetxt(csv_filepath, all_metrics, delimiter=',', header='SNR,RMSE,SSS', comments='')
            
            with open(eval_filepath, 'w', encoding='utf-8') as f_eval:
                f_eval.write(f"================ {arch} Evaluation Results ================\n")
                f_eval.write(f"Test Date        : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f_eval.write(f"File Name        : {file_name}\n")
                f_eval.write(f"Num of Spectra   : {num_samples}\n")
                f_eval.write(f"Processing Time  : {inference_time:.4f} seconds\n")
                f_eval.write("SNR/dB\tRMSE\tSSS\n")
                f_eval.write(f"{avg_snr}\t{avg_rmse}\t{avg_sss}\n")
                f_eval.write("=========================================================\n")
            
            print(f"指标已生成 -> SNR: {avg_snr} | RMSE: {avg_rmse} | SSS: {avg_sss}")
        else:
            print("测试数据中未找到 'gt'，跳过指标计算环节。")
        print()

    print("\n--- 批量测试流程结束 ---")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for Spectra MAE model inference.")
    
    # 配置文件路径相关参数
    parser.add_argument('--input_path', type=str, 
                        default='/home/slurmusers/moria/codes/fast2peak_v2/datasets/test/2DMs.h5',
                        help='输入的测试数据文件路径，或包含多个 txt/h5 的目录')
    
    parser.add_argument('--output_dir', type=str, 
                        default='/home/slurmusers/moria/codes/fast2peak_v2/experiments/smae raw ps1/',
                        help='推理结果保存目录的路径')
    
    parser.add_argument('--chkpt_dir', type=str, 
                        default='/home/slurmusers/moria/codes/fast2peak_v2/smae/experiments/smae with train.h5 ss1600 ps1/epoch_300_final.pth',
                        help='模型权重文件 (.pth) 的路径')
    
    # 模型和训练配置相关参数
    parser.add_argument('--arch', type=str, 
                        default='spectraMAE_base_ss1600_ps1',
                        help='模型架构名称')
    
    parser.add_argument('--batch_size', type=int, 
                        default=128,
                        help='推理时的批量大小')
    
    parser.add_argument('--seed', type=int, 
                        default=615,
                        help='随机种子')

    parser.add_argument('--mask_ratio', type=float, 
                        default=0.50,
                        help='推理时的掩码比例')

    # 解析命令行参数
    args = parser.parse_args()

    print("\n" + "="*40)
    print("🚀 开始执行测试任务，参数如下:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("="*40 + "\n")

    # 调用测试函数
    test(
        input_path=args.input_path,
        output_dir=args.output_dir,
        chkpt_dir=args.chkpt_dir,
        arch=args.arch,
        batch_size=args.batch_size,
        seed=args.seed,
        mask_ratio=args.mask_ratio
    )