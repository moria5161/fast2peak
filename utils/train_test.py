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
        if args.sft:
            print('[SFT] Full-spectrum MSE against GT; no masking. Select by validation MSE.')
        
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    min_lr = args.min_lr if args.min_lr is not None else args.lr * 0.01
    scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=min_lr,
                                warmup_t=args.warmup_epochs, warmup_lr_init=args.lr * 0.1,
                                cycle_limit=1, t_in_epochs=True)
    
    start_epoch = 0 if args.sft else 1
    if args.resume:
        map_loc = f'cuda:{device}' if isinstance(device, int) else device
        checkpoint = torch.load(args.resume, map_location=map_loc)
        if isinstance(checkpoint, dict):
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                if local_rank == 0: print(f"Resuming from epoch {start_epoch}")

    loss_function = nn.MSELoss()
    logger = TensorBoardLogger(log_dir) if local_rank == 0 else None
    scaler = GradScaler(device="cuda", enabled=torch.cuda.is_available())
    
    train_meter = AverageMeter()
    val_meter = AverageMeter()
    
    patience = args.patience if args.patience is not None else int(args.epochs * 0.8)
    early_stopping = EarlyStop(patience=patience, mode='min', delta=1e-10)
    if args.sft and args.resume and 'score' in checkpoint:
        early_stopping.best_score = -float(checkpoint['score'])
        early_stopping.val_score = float(checkpoint['score'])

    for epoch in range(start_epoch, args.epochs + 1):
        train_meter.reset()
        val_meter.reset()

        if args.ddp:
            sampler_train.set_epoch(epoch) # Ensure shuffling for each epoch in DDP
        model.train()

        # Epoch 0 evaluates the pretrained model before any updates.
        for batch in (loader_train if epoch > 0 else []):
            label, input_spec = batch['label'].to(device), batch['input'].to(device)
            if not args.sft:
                mask = batch['mask'].to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                
                if args.sft:
                    output = model(input_spec)
                    loss = loss_function(output, label)
                elif args.model_name in ['SMAE', 'RamanMAE', '2DSMAE']:
                    # Mask ratio handling
                    mask_r = args.smae_mask_ratio if args.model_name in ['SMAE', '2DSMAE'] else args.mask_ratio
                    loss, _, _ = model(input_spec, mask_ratio=mask_r)
                else:
                    output = model(input_spec)
                    loss = loss_function(output * (1 - mask), label * (1 - mask))
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
                    label, input_spec = data['label'].to(device), data['input'].to(device)
                    if not args.sft:
                        mask = data['mask'].to(device)
                    if args.sft:
                        output = model(input_spec)
                        loss = loss_function(output, label)
                    elif args.model_name in ['SMAE', 'RamanMAE', '2DSMAE']:
                        mask_r = args.smae_mask_ratio if args.model_name in ['SMAE', '2DSMAE'] else args.mask_ratio
                        loss, _, _ = model(input_spec, mask_ratio=mask_r)
                    else:
                        output = model(input_spec)
                        loss = loss_function(output * (1 - mask), label * (1 - mask))
                    # Full-spectrum loss.
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
            if epoch == 0:
                print(f"Epoch 0 | Initial Val Loss: {avg_val_loss:.8f}")
                logger.writer.add_scalar('Loss/epoch_val', avg_val_loss, epoch)
            else:
                print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.8f} | Val Loss: {avg_val_loss:.8f}")
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

        if epoch > 0:
            scheduler.step(epoch)

    if logger:
        logger.close()
    if local_rank == 0:
        print("Training completed")
        print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes")


def test_single_file(model, args, device, output_dir, file_path):
    test_datapath_lower = file_path.lower()

    if test_datapath_lower.endswith('.txt'):
        data_noisy = np.loadtxt(file_path)
        if data_noisy.ndim == 1:
            data_noisy = np.expand_dims(data_noisy, axis=0)
        spectra_noisy, wavenumbers = np.expand_dims(data_noisy[:, 1], axis=0), data_noisy[:, 0]
        
        input_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = os.path.join(output_dir, f'{input_filename}_{args.model_name}.txt')
    
    elif test_datapath_lower.endswith('.mat'):
        data_input = sio.loadmat(file_path)
        spectra_noisy = data_input['Noisy']
        output_filename = file_path
    
    elif test_datapath_lower.endswith('.h5'):
        with h5py.File(file_path, 'r') as f:
            spectra_noisy = f['noisy'][:].astype(np.float32)
            gt = f['gt'][:].astype(np.float32) if 'gt' in f else None
        file_name = os.path.basename(file_path)
        output_filename = os.path.join(output_dir, file_name)
    
    else:
        print(f"Unsupported test file type. Please provide .txt, .mat or .h5 file for {file_path}")
        return

    print(f"Loading test data: {file_path}, shape: {spectra_noisy.shape}")

    spectra_noisy_orig = spectra_noisy.copy().astype(np.float32)
    original_length = spectra_noisy_orig.shape[1]
    
    spectra_processed, alignment_mode, pad_left = spectra_alignment(spectra_noisy_orig, original_length, args.spectrum_length)

    min_vals = np.min(spectra_processed, axis=1, keepdims=True)
    max_vals = np.max(spectra_processed, axis=1, keepdims=True)
    spectra_norm = (spectra_processed - min_vals) / (max_vals - min_vals + 1e-8)

    model.eval()

    test_start_time = time.time()
    denoised_list = []
    
    with torch.no_grad():
        for i in range(0, spectra_norm.shape[0], args.test_batch_size):
            batch = spectra_norm[i:i + args.test_batch_size]
            batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(device, dtype=torch.float32)
            if args.model_name in ['SMAE', '2DSMAE']:
                _, output, _ = model(batch_tensor, mask_ratio=args.smae_mask_ratio)
                output = model.unpatchify(output)
                output = output.cpu().numpy()
            elif args.model_name == 'RamanMAE':
                # Test logic: Predict the entire image with mask_ratio=0.0
                _, output, _ = model(batch_tensor, mask_ratio=0.0)
                output = output.view(batch_tensor.shape).cpu().numpy()
            else:    
                output = model(batch_tensor).cpu().numpy()
            denoised_list.append(output)
            
    inference_time = time.time() - test_start_time

    print(f"Denoising complete, time elapsed: {inference_time:.4f}s")
    denoised_norm = np.concatenate(denoised_list, axis=0).squeeze(1)
    denoised_extended = denoised_norm * (max_vals - min_vals + 1e-8) + min_vals
    
    denoised_final = spectra_alignment_reverse(denoised_extended, alignment_mode, original_length, pad_left)
    if args.sft:
        denoised_corrected = denoised_final
    else:
        denoised_corrected = spectra_correction(spectra_noisy_orig, denoised_final)

    if test_datapath_lower.endswith('.txt'):
        np.savetxt(output_filename, np.column_stack((wavenumbers, denoised_corrected.squeeze())), fmt='%.6f')
    elif test_datapath_lower.endswith('.mat'):
        if denoised_corrected.shape[1] == original_length:
            data_input['F2P'] = denoised_corrected
            sio.savemat(output_filename, data_input)
            # Compute RMSE.
            # data_gt = data_input['GT']
            # diff = (data_gt - denoised_corrected)**2
            # mse_per_spectrum = np.mean(diff, axis=1)
            # rmse = np.mean(np.sqrt(mse_per_spectrum))
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
            all_metrics = np.array([calculate_metrics(denoised_corrected[i], gt[i]) for i in range(num_samples)])
            
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
            print(f"Final Results -> SNR: {avg_snr} | RMSE: {avg_rmse} | SSS: {avg_sss}")
        else:
            print("Ground truth not found. Skipping evaluation.")
                
    print(f'Result saved to: {output_filename}')


def test(model, args, device, output_dir):
    if os.path.isdir(args.test_datapath):
        file_list = [os.path.join(args.test_datapath, f) for f in os.listdir(args.test_datapath) if os.path.isfile(os.path.join(args.test_datapath, f))]
        for file_path in file_list:
            test_single_file(model, args, device, output_dir, file_path)
    else:
        test_single_file(model, args, device, output_dir, args.test_datapath)
