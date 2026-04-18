import os
import time
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from timm.scheduler import CosineLRScheduler

from utils.base import TensorBoardLogger, EarlyStop, AverageMeter, spectra_alignment, spectra_alignment_reverse, spectra_correction, calculate_metrics
from utils.spec_loader import data_loaders


def train(model, args, device, local_rank, world_size, log_dir, checkpoint_dir):
    start_time = time.time()            
    loader_train, loader_val, sampler_train = data_loaders(args)
    if local_rank == 0:
        print(f"Data Loaded. Train batches: {len(loader_train)}, Val batches: {len(loader_val)}")
        
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=args.lr * 0.01, 
                                warmup_t=10, warmup_lr_init=args.lr * 0.1, cycle_limit=1, t_in_epochs=True)
    
    start_epoch = 1
    if args.resume:
        if local_rank == 0:
            print(f"Attempting to resume optimizer and scheduler from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict):
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if local_rank == 0: print("Optimizer state loaded.")
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if local_rank == 0: print("Scheduler state loaded.")
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                if local_rank == 0: print(f"Resuming from epoch {start_epoch}")

    loss_function = nn.MSELoss()
    logger = TensorBoardLogger(log_dir) if local_rank == 0 else None
    scaler = GradScaler(device="cuda", enabled=torch.cuda.is_available())
    
    train_meter = AverageMeter()
    val_meter = AverageMeter()
    
    early_stopping = EarlyStop(patience=int(args.epochs * 0.8), mode='min', delta=1e-10)

    for epoch in range(start_epoch, args.epochs + 1):
        train_meter.reset()
        val_meter.reset()

        if args.ddp:
            sampler_train.set_epoch(epoch) # Ensure shuffling for each epoch in DDP
        model.train()

        for batch in loader_train:
            label, input_spec, mask = batch['label'].to(device), batch['input'].to(device), batch['mask'].to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                
                if args.model_name == 'SMAE':
                    loss, _, _ = model(input_spec, mask_ratio=args.smae_mask_ratio)
                else:
                    output = model(input_spec)
                    loss = loss_function(output * (1 - mask), label * (1 - mask))
                    # 计算全谱的损失
                    # loss = loss_function(output, label)
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)  
            scaler.update()

            train_meter.update(loss.item(), input_spec.size(0))

            if logger:
                logger.log_batch_loss(loss.item())
        
        avg_train_loss = train_meter.avg

        if args.ddp:
            loss_tensor = torch.tensor(avg_train_loss).to(device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = loss_tensor.item() / world_size

        model.eval()
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                for data in loader_val:
                    label, input_spec, mask = data['label'].to(device), data['input'].to(device), data['mask'].to(device)
                    if args.model_name == 'SMAE':
                        loss, _, _ = model(input_spec, mask_ratio=args.smae_mask_ratio)
                    else:
                        output = model(input_spec)
                        loss = loss_function(output * (1 - mask), label * (1 - mask))
                    # 计算全谱的损失
                    # loss = loss_function(output, label)
                    val_meter.update(loss.item(), input_spec.size(0))
        
        avg_val_loss = val_meter.avg

        if args.ddp:
            loss_tensor = torch.tensor(avg_val_loss).to(device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = loss_tensor.item() / world_size
            
        model_to_save = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
        
        best_model_path = None
        if local_rank == 0:
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_epoch_metrics(epoch, avg_train_loss, avg_val_loss, current_lr)
            best_model_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
            
        early_stopping(
            epoch_score=avg_val_loss, 
            model=model_to_save, 
            model_path=best_model_path, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            epoch=epoch
        )

        if early_stopping.early_stop:
            if local_rank == 0:
                print(f"Early stopping triggered at epoch {epoch}")
            break
  
        if epoch % 100 == 0 and epoch > int(args.epochs * 0.5):
            if local_rank == 0:
                print(f"Saving periodic snapshot at epoch {epoch}...")
                snapshot_path = os.path.join(checkpoint_dir, f'epoch{epoch}.pth')
                checkpoint = {
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'score': avg_val_loss,
                    'epoch': epoch
                }
                torch.save(checkpoint, snapshot_path)

        scheduler.step(epoch)

    if logger:
        logger.close()
    if local_rank == 0:
        print("Training completed")
        print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes")


def test(model, args, device, output_dir):
    test_datapath_lower = args.test_datapath.lower()

    if test_datapath_lower.endswith('.txt'):
        data_noisy = np.loadtxt(args.test_datapath)
        if data_noisy.ndim == 1:
            data_noisy = np.expand_dims(data_noisy, axis=0)
        spectra_noisy, wavenumbers = np.expand_dims(data_noisy[:, 1], axis=0), data_noisy[:, 0]
        
        input_filename = os.path.splitext(os.path.basename(args.test_datapath))[0]
        output_filename = os.path.join(output_dir, f'{input_filename}_{args.model_name}.txt')
    
    elif test_datapath_lower.endswith('.mat'):
        data_input = sio.loadmat(args.test_datapath)
        spectra_noisy = data_input['Noisy']
        output_filename = args.test_datapath
    
    elif test_datapath_lower.endswith('.h5'):
        with h5py.File(args.test_datapath, 'r') as f:
            spectra_noisy = f['noisy'][:].astype(np.float32)
            gt = f['gt'][:].astype(np.float32) if 'gt' in f else None
        file_name = os.path.basename(args.test_datapath)
        output_filename = os.path.join(output_dir, file_name)
    
    else:
        print("Unsupported test file type. Please provide .txt, .mat or .h5 file")
        return

    print(f"Loading test data: {args.test_datapath}, shape: {spectra_noisy.shape}")

    spectra_noisy_orig = spectra_noisy.copy().astype(np.float32)
    original_length = spectra_noisy_orig.shape[1]
    
    spectra_processed, alignment_mode, pad_left = spectra_alignment(spectra_noisy_orig, original_length, args.spectrum_length)

    min_vals = np.min(spectra_processed, axis=1, keepdims=True)
    max_vals = np.max(spectra_processed, axis=1, keepdims=True)
    spectra_norm = (spectra_processed - min_vals) / (max_vals - min_vals + 1e-8)

    model.eval()
    
    # # ================= (Deep Warm-up) =================
    # print("Starting warm-up...")
    # dummy_input_np = np.random.randn(1, 1, args.spectrum_length).astype(np.float32)
    # with torch.no_grad():
    #     for _ in range(1): 
    #         tmp_tensor = torch.from_numpy(dummy_input_np).to(device)
    #         if args.model_name == 'SMAE':
    #             _, tmp_out, _ = model(tmp_tensor, mask_ratio=args.smae_mask_ratio)
    #             tmp_out = model.unpatchify(tmp_out)
    #         else:
    #             tmp_out = model(tmp_tensor)
    #         _ = tmp_out.cpu().numpy()
    # torch.cuda.synchronize()
    # print("Warm-up completed, starting formal testing!")
    # # ===================================================================

    test_start_time = time.time()
    denoised_list = []
    
    with torch.no_grad():
        for i in range(0, spectra_norm.shape[0], args.test_batch_size):
            batch = spectra_norm[i:i + args.test_batch_size]
            batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(device, dtype=torch.float32)
            if args.model_name == 'SMAE':
                _, output, _ = model(batch_tensor, mask_ratio=args.smae_mask_ratio)
                output = model.unpatchify(output)
                output = output.cpu().numpy()
            else:    
                output = model(batch_tensor).cpu().numpy()
            denoised_list.append(output)
            
    inference_time = time.time() - test_start_time

    print(f"Denoising complete, time elapsed: {inference_time:.4f}s")
    # from IPython import embed; embed()  # Debug point to inspect variables before post-processing
    denoised_norm = np.concatenate(denoised_list, axis=0).squeeze(1)
    denoised_extended = denoised_norm * (max_vals - min_vals + 1e-8) + min_vals
    
    denoised_final = spectra_alignment_reverse(denoised_extended, alignment_mode, original_length, pad_left)
    denoised_corrected = spectra_correction(spectra_noisy_orig, denoised_final)

    if test_datapath_lower.endswith('.txt'):
        np.savetxt(output_filename, np.column_stack((wavenumbers, denoised_corrected.squeeze())), fmt='%.6f')
    elif test_datapath_lower.endswith('.mat'):
        if denoised_corrected.shape[1] == original_length:
            data_input['F2P'] = denoised_corrected
            sio.savemat(output_filename, data_input)
            # 计算RMSE
            # data_gt = data_input['GT']
            # diff = (data_gt - denoised_corrected)**2
            # mse_per_spectrum = np.mean(diff, axis=1)
            # rmse = np.mean(np.sqrt(mse_per_spectrum))
            # print(f"Average RMSE: {rmse:.4f}")
        else:
            print(f"Error: Denoised spectrum length ({denoised_corrected.shape[1]}) mismatch original ({original_length}), cannot save .mat file.")
    elif test_datapath_lower.endswith('.h5'):
        if denoised_corrected.shape[1] == original_length:
            with h5py.File(output_filename, 'a') as f:
                if 'denoised' in f:
                    del f['denoised']  
                f.create_dataset(f'denoised', data=denoised_corrected)
        else:
            print(f"Error: Denoised spectrum length ({denoised_corrected.shape[1]}) mismatch original ({original_length}), cannot save .h5 file.")
        
        if gt is not None:
            num_samples = gt.shape[0]
            all_metrics = np.array([calculate_metrics(gt[i], denoised_corrected[i]) for i in range(num_samples)])
            
            avg_snr = np.round(np.mean(all_metrics[:, 0]), 2)
            avg_rmse = np.round(np.mean(all_metrics[:, 1]), 4)
            avg_sss = np.round(np.mean(all_metrics[:, 2]), 4)
            
            base_name = os.path.splitext(file_name)[0]
            eval_filepath = os.path.join(output_dir, f"{base_name}_eval.log")
            csv_filepath = os.path.join(output_dir, f"{base_name}_metrics_details.csv")
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
                
    print(f'Result saved to: {output_filename}')