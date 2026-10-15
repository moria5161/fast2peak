import os
import time
import torch
import random
import numpy as np 

from torch.utils.tensorboard import SummaryWriter
from numpy import linalg as la
from scipy.interpolate import interp1d


class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    def log_batch_loss(self, loss):
        self.writer.add_scalar('Loss/train_batch', loss, self.global_step)
        self.global_step += 1

    def log_epoch_metrics(self, epoch, train_loss, val_loss, lr):
        self.writer.add_scalar('Loss/epoch_train', train_loss, epoch)
        self.writer.add_scalar('Loss/epoch_val', val_loss, epoch)
        self.writer.add_scalar('Misc/learning_rate', lr, epoch)

    def close(self):
        self.writer.close()


class EarlyStop:
    def __init__(self, patience=10, mode='max', delta=1e-12):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == 'min':
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

    def __call__(self, epoch_score, model, model_path, optimizer=None, scheduler=None, epoch=None):
        if self.mode == 'min':
            score = -1. * epoch_score
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path, optimizer, scheduler, epoch)
        elif score < self.best_score+self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path, optimizer, scheduler, epoch)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path, optimizer=None, scheduler=None, epoch=None):
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save a dict containing state_dicts and other info
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'score': epoch_score,
                'epoch': epoch
            }
            if optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, model_path)
            print(f'Validation score improved ({self.val_score:.6f} --> {epoch_score:.6f}).  Saving model ...')
        self.val_score = epoch_score


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count


def spectra_correction(raw_spec_batch, denoised_spec_batch):
    def correct_single(raw, denoised):
        A = np.vstack([denoised, np.ones(len(denoised))]).T
        a, b = la.lstsq(A, raw, rcond=None)[0]
        corrected = a * denoised + b
        corrected += np.mean(raw) - np.mean(corrected)
        return corrected

    corrected_batch = np.zeros_like(denoised_spec_batch)
    for i in range(raw_spec_batch.shape[0]):
        corrected_batch[i] = correct_single(raw_spec_batch[i], denoised_spec_batch[i])
    return corrected_batch


def spectra_resample(y, target_len):
    original_len = y.shape[1]
    
    x_old = np.linspace(0, 1, original_len)
    x_target = np.linspace(0, 1, target_len)
    
    f_expand = interp1d(x_old, y, kind='cubic', axis=1)
    y_target = f_expand(x_target)
    
    return y_target


def spectra_resample_single(y, target_len):
    original_len = len(y)
    
    x_old = np.linspace(0, 1, original_len)
    x_target = np.linspace(0, 1, target_len)
    
    f_interp = interp1d(x_old, y, kind='cubic')
    y_target = f_interp(x_target)
    
    return y_target


def spectra_alignment(spectra_noisy_orig, original_length, target_length):
    processing_mode = 'none'
    pad_left = 0

    if original_length > target_length:
        processing_mode = 'resample'
        spectra_processed = spectra_resample(spectra_noisy_orig, target_length)
    elif original_length < target_length:
        processing_mode = 'resample'
        spectra_processed = spectra_resample(spectra_noisy_orig, target_length)
        # processing_mode = 'pad'
        # total_padding = target_length - original_length
        # pad_left = total_padding // 2
        # pad_right = total_padding - pad_left
        # padding_width = ((0, 0), (pad_left, pad_right))
        # spectra_processed = np.pad(spectra_noisy_orig, pad_width=padding_width, mode='reflect')
    else:
        spectra_processed = spectra_noisy_orig
    
    return spectra_processed, processing_mode, pad_left


def spectra_alignment_reverse(spectra, alignment_mode, original_length, pad_left):
    if alignment_mode == 'resample':
        spectra_processed = spectra_resample(spectra, original_length)
    elif alignment_mode == 'pad':
        crop_end = pad_left + original_length
        spectra_processed = spectra[:, pad_left:crop_end]
    else:
        spectra_processed = spectra
    
    return spectra_processed


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_state(net, state_dict, ddp=False, strict=False):
    # Handle new checkpoint format (dict with 'model_state_dict')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    if not ddp:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    own_state = net.state_dict()

    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
            else:
                print(f'Shape mismatch: {name} {own_state[name].shape} vs {param.shape}')
                if strict: raise RuntimeError(f'Shape mismatch: {name}')
        else:
            if strict: 
                print(f'Key missing: {name}')
                raise RuntimeError(f'Key missing: {name}')

    return net


def calculate_metrics(spec_denoised, spec_gt):
    x = np.array(spec_denoised)
    y = np.array(spec_gt)
    
    rmse = np.sqrt(np.mean((x - y) ** 2))
    snr = 10 * np.log10(np.sum(y ** 2) / np.sum((x - y) ** 2) + 1e-8)

    inner_product = np.dot(x, y)
    norms = np.linalg.norm(x) * np.linalg.norm(y)
    sad = np.arccos(np.clip(inner_product / norms, -1.0, 1.0))

    pcc = np.corrcoef(x, y)[0, 1]
    
    sss = (1-sad) * pcc
    
    return snr, rmse, sss
