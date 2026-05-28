import os
import numpy as np
import torch
import random
import h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from utils.base import spectra_resample


# ----------------------------------------------------------------
# Spectra transform

# Add graussian noise with zero mean and standard deviation 0.01 to 0.04
class AddGaussianNoise(object):
    def __call__(self, x):
        var = random.random() * 0.04 + 0.01
        noise = np.random.normal(0, var, x.shape)
        x += noise
        x = np.clip(x, 0, 1)
        return x


# Average blur with widow size 1 to 5
class RandomBlur(object):
    def __call__(self, x):
        size = random.randint(1, 5)
        x = np.convolve(x, np.ones(size) / size, mode='same')
        return x


# randomly set the intensity of spectrum to 0
class RandomDropout(object):
    def __call__(self, x, droprate=0.1):
        noise = np.random.random(x.shape)
        x = (noise > droprate) * x
        return x

# mulitiply the spectrum by a scale-factor
class RandomScaleTransform(object):
    def __call__(self, x):
        scale = np.random.uniform(0.9, 1.1, x.shape)
        x = scale * x
        x = np.clip(x, 0, 1)
        return x


# convert to Tensor with 1 channel
class ToFloatTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x).view(1, -1).float()
    

train_transform = transforms.Compose([
    transforms.RandomApply([transforms.RandomChoice([
        RandomBlur(),
        AddGaussianNoise(),
    ])],p=0.5),
    transforms.RandomApply([RandomDropout()], p=0.5),
    transforms.RandomApply([RandomScaleTransform()], p=0.5),
    ToFloatTensor()
])
valid_transform = ToFloatTensor()


class Pathogenic_Unsupervise(Dataset):
    def __init__(self, data, transform=None):
        self.data = data.astype(np.float32)
        self.transform = transform
        self.real_len = len(self.data)  # 修正：统一为 self.data

    def __len__(self):
        return self.real_len

    def __getitem__(self, index):
        # 修正：使用 self.data，并加上 .copy() 防止 transform 的 in-place 修改污染原数组
        y = self.data[index].copy() 
        
        if self.transform:
            y = self.transform(y)

        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y)

        if y.dim() == 1:
            y = y.unsqueeze(0)
            
        dummy_mask = torch.zeros_like(y)

        return {
            'label': y,
            'input': y,
            'mask': dummy_mask
        }


class MaskDataset(Dataset):
    def __init__(self, data, gamma, mask_ratio):
        self.data = data.astype(np.float32)
        self.spectrum_length = data.shape[1]
        self.gamma = gamma
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.data[index]
        input_spec, mask = self.generate_mask(label)
        return {
            'label': torch.from_numpy(label).unsqueeze(0),
            'input': torch.from_numpy(input_spec).unsqueeze(0),
            'mask': torch.from_numpy(mask).unsqueeze(0)
        }

    def generate_mask(self, spectrum):
        num_to_mask = int(self.spectrum_length * self.mask_ratio)
        mask = np.ones(self.spectrum_length, dtype=np.float32)
        input_spec = spectrum.copy()

        idx_mask = np.random.choice(self.spectrum_length, num_to_mask, replace=False)
        
        offset = np.random.randint(-self.gamma // 2, self.gamma // 2 + 1, num_to_mask)
        offset[offset == 0] = 1

        idx_neighbor = np.clip(idx_mask + offset, 0, self.spectrum_length - 1)
        
        input_spec[idx_mask] = spectrum[idx_neighbor]
        mask[idx_mask] = 0.0
        return input_spec, mask


class BlockMaskDataset(Dataset):
    def __init__(self, data, gamma, mask_ratio, block_size=8):
        self.data = data.astype(np.float32)
        self.spectrum_length = data.shape[1]
        self.gamma = gamma
        self.mask_ratio = mask_ratio
        # 新增：掩码块的大小，建议与 patch_size 保持一致或略大
        self.block_size = block_size 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.data[index]
        input_spec, mask = self.generate_mask(label)
        return {
            'label': torch.from_numpy(label).unsqueeze(0),
            'input': torch.from_numpy(input_spec).unsqueeze(0),
            'mask': torch.from_numpy(mask).unsqueeze(0)
        }

    def generate_mask(self, spectrum):
        # 计算总共需要掩码的点数
        total_mask_points = int(self.spectrum_length * self.mask_ratio)
        # 计算需要掩码的“块(Block)”的数量
        num_blocks = total_mask_points // self.block_size
        
        mask = np.ones(self.spectrum_length, dtype=np.float32)
        input_spec = spectrum.copy()

        if num_blocks > 0:
            # 随机选择块的起始索引 (确保块不会超出边界)
            valid_start_indices = self.spectrum_length - self.block_size
            block_starts = np.random.choice(valid_start_indices, num_blocks, replace=False)
            
            # 生成所有被掩码的精确索引
            idx_mask = []
            for start in block_starts:
                idx_mask.extend(range(start, start + self.block_size))
            idx_mask = np.array(idx_mask)
            
            # 保留您原有的相邻点偏移（Jitter）替换逻辑，增加任务难度
            offset = np.random.randint(-self.gamma // 2, self.gamma // 2 + 1, len(idx_mask))
            offset[offset == 0] = 1
            idx_neighbor = np.clip(idx_mask + offset, 0, self.spectrum_length - 1)
            
            # 执行掩码替换
            input_spec[idx_mask] = spectrum[idx_neighbor]
            mask[idx_mask] = 0.0
            
        return input_spec, mask


class BaseDataset(Dataset):
    def __init__(self, noisy_data, clean_data):
        self.noisy = noisy_data.astype(np.float32)
        self.clean = clean_data.astype(np.float32)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        return {
            'label': torch.from_numpy(self.clean[index]).unsqueeze(0),
            'input': torch.from_numpy(self.noisy[index]).unsqueeze(0),
            'mask': torch.zeros((1, self.noisy.shape[1]), dtype=torch.float32)
        }


class RamanDataset(Dataset):
    def __init__(self, inputs, outputs, spectrum_len=500, spectrum_shift=0., 
                 spectrum_window=False, horizontal_flip=False, mixup=False):
        self.inputs = inputs
        self.outputs = outputs
        self.spectrum_len = spectrum_len
        self.spectrum_shift = spectrum_shift
        self.spectrum_window = spectrum_window
        self.horizontal_flip = horizontal_flip
        self.mixup = mixup

    def __len__(self):
        return len(self.inputs)

    def pad_spectrum(self, input_spectrum, spectrum_length):
        if len(input_spectrum) == spectrum_length:
            padded_spectrum = input_spectrum
        elif len(input_spectrum) > spectrum_length:
            padded_spectrum = input_spectrum[0:spectrum_length]
        else:
            padded_spectrum = np.pad(input_spectrum, ((0,spectrum_length - len(input_spectrum)),(0,0)), 'reflect')

        return padded_spectrum
    
    def window_spectrum(self, input_spectrum, start_idx, window_length):
        if len(input_spectrum) <= window_length:
            output_spectrum = input_spectrum
        else:
            end_idx = start_idx + window_length
            output_spectrum = input_spectrum[start_idx:end_idx]

        return output_spectrum
            
    def flip_axis(self, x, axis):
        if np.random.random() < 0.5:
            x = np.asarray(x).swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)
        return x
    
    def shift_spectrum(self, x, shift_range):
        x = np.expand_dims(x,axis=-1)
        shifted_spectrum = x
        spectrum_shift_range = int(np.round(shift_range*len(x)))
        if spectrum_shift_range > 0:
            shifted_spectrum = np.pad(x[spectrum_shift_range:,:], ((0,abs(spectrum_shift_range)), (0,0)), 'reflect')
        elif spectrum_shift_range < 0:
            shifted_spectrum = np.pad(x[:spectrum_shift_range,:], ((abs(spectrum_shift_range), 0), (0,0)), 'reflect')
        return shifted_spectrum
    
    def mixup_spectrum(self, input_spectrum1, input_spectrum2, output_spectrum1, output_spectrum2, alpha):
        lam = np.random.beta(alpha, alpha)
        input_spectrum = (lam * input_spectrum1) + ((1 - lam) * input_spectrum2)
        output_spectrum = (lam * output_spectrum1) + ((1 - lam) * output_spectrum2)
        return input_spectrum, output_spectrum
            
    def __getitem__(self, index):       
        input_spectrum = self.inputs[index]
        output_spectrum = self.outputs[index]
        
        mixup_on = False
        if self.mixup:
            if np.random.random() < 0.5:
                spectrum_idx = int(np.round(np.random.random() * (len(self.inputs)-1)))
                input_spectrum2 = self.inputs[spectrum_idx]
                output_spectrum2 = self.outputs[spectrum_idx]
                mixup_on = True

        if self.spectrum_window:
            start_idx = int(np.floor(np.random.random() * (len(input_spectrum)-self.spectrum_len)))
            input_spectrum = self.window_spectrum(input_spectrum, start_idx, self.spectrum_len)
            output_spectrum = self.window_spectrum(output_spectrum, start_idx, self.spectrum_len)
            if mixup_on:
                input_spectrum2 = self.window_spectrum(input_spectrum2, start_idx, self.spectrum_len)
                output_spectrum2 = self.window_spectrum(output_spectrum2, start_idx, self.spectrum_len)

        input_spectrum = self.pad_spectrum(input_spectrum, self.spectrum_len)
        output_spectrum = self.pad_spectrum(output_spectrum, self.spectrum_len)
        if mixup_on:
            input_spectrum2 = self.pad_spectrum(input_spectrum2, self.spectrum_len)
            output_spectrum2 = self.pad_spectrum(output_spectrum2, self.spectrum_len)

        if self.spectrum_shift != 0.0:
            shift_range = np.random.uniform(-self.spectrum_shift, self.spectrum_shift)
            input_spectrum = self.shift_spectrum(input_spectrum, shift_range)
            output_spectrum = self.shift_spectrum(output_spectrum, shift_range)
            if mixup_on:
                input_spectrum2 = self.shift_spectrum(input_spectrum2, shift_range)
                output_spectrum2 = self.shift_spectrum(output_spectrum2, shift_range)
        else:
            input_spectrum = np.expand_dims(input_spectrum, axis=-1)
            output_spectrum = np.expand_dims(output_spectrum, axis=-1)
            if mixup_on:
                input_spectrum2 = np.expand_dims(input_spectrum2, axis=-1)
                output_spectrum2 = np.expand_dims(output_spectrum2, axis=-1)

        if self.horizontal_flip:    
            if np.random.random() < 0.5:
                input_spectrum = self.flip_axis(input_spectrum, 0)
                output_spectrum = self.flip_axis(output_spectrum, 0)
                if mixup_on:
                    input_spectrum2 = self.flip_axis(input_spectrum2, 0)
                    output_spectrum2 = self.flip_axis(output_spectrum2, 0)

        if mixup_on:
            input_spectrum, output_spectrum = self.mixup_spectrum(input_spectrum, input_spectrum2, output_spectrum, output_spectrum2, 0.2)
            
        input_spectrum = input_spectrum/np.amax(input_spectrum)
        output_spectrum = output_spectrum/np.amax(output_spectrum)
        
        input_spectrum = np.moveaxis(input_spectrum, -1, 0)
        output_spectrum = np.moveaxis(output_spectrum, -1, 0)
        
        sample = {
        'input': torch.from_numpy(input_spectrum).float(),
        'label': torch.from_numpy(output_spectrum).float(),
        'mask': torch.zeros((1, self.spectrum_len)).float()
        }
        
        return sample


def min_max_normalization(data):
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    return (data - min_vals) / (max_vals - min_vals + 1e-10)


def data_load_process(file_path: str, key: str, target_length: int) -> np.ndarray:      
    with h5py.File(file_path, 'r') as f:
        data = f[key][:].astype(np.float32)

    data = min_max_normalization(data)
    if data.shape[1] != target_length:
        data = spectra_resample(data, target_length) 
        
    return data


def data_loaders(args):
    data_dir = args.train_data_dir
    
    if os.path.isfile(data_dir):
        train_path = data_dir
        val_path = data_dir
        print(f"[Training] Using single file as dataset: {train_path}")
    else:
        train_path = os.path.join(data_dir, 'train.h5')
        val_path = os.path.join(data_dir, 'val.h5')
    

    if args.model_name in ['SMAE', 'RamanMAE', '2DSMAE']:
        if args.pretrained:
            train_path = args.fine_tune_data_path
            val_path = train_path
            print(f"[Fine-tuning] f2p using data from: {train_path}")
        train_data = data_load_process(train_path, 'noisy', args.spectrum_length)
        val_data = data_load_process(val_path, 'noisy', args.spectrum_length)
        
        ds_train = Pathogenic_Unsupervise(train_data, transform=train_transform)
        ds_val = Pathogenic_Unsupervise(val_data, transform=valid_transform)

    if args.model_name == 'Fast2Peak' or args.model_name == 'Fast2PeakConv':
        if args.pretrained:
            train_path = args.fine_tune_data_path
            val_path = train_path
            print(f"[Fine-tuning] f2p using data from: {train_path}")
        train_data = data_load_process(train_path, 'noisy', args.spectrum_length)
        val_data = data_load_process(val_path, 'noisy', args.spectrum_length)
        
        ds_train = MaskDataset(train_data, args.gamma, args.mask_ratio)
        ds_val = MaskDataset(val_data, args.gamma, args.mask_ratio)
        # ds_train = BlockMaskDataset(train_data, args.gamma, args.mask_ratio)
        # ds_val = BlockMaskDataset(val_data, args.gamma, args.mask_ratio)
        
    if args.model_name == 'Fast2PeakRPE':
        if args.pretrained:
            train_path = args.fine_tune_data_path
            val_path = train_path
            print(f"[Fine-tuning] f2p using data from: {train_path}")
        train_data = data_load_process(train_path, 'noisy', args.spectrum_length)
        val_data = data_load_process(val_path, 'noisy', args.spectrum_length)
        
        # ds_train = MaskDataset(train_data, args.gamma, args.mask_ratio)
        # ds_val = MaskDataset(val_data, args.gamma, args.mask_ratio)
        ds_train = BlockMaskDataset(train_data, args.gamma, args.mask_ratio, args.block_size)
        ds_val = BlockMaskDataset(val_data, args.gamma, args.mask_ratio, args.block_size)

    if args.model_name == 'Fast2PeakOverlap':
        if args.pretrained:
            train_path = args.fine_tune_data_path
            val_path = train_path
            print(f"[Fine-tuning] f2p using data from: {train_path}")
        train_data = data_load_process(train_path, 'noisy', args.spectrum_length)
        val_data = data_load_process(val_path, 'noisy', args.spectrum_length)

        ds_train = BlockMaskDataset(train_data, args.gamma, args.mask_ratio, args.block_size)
        ds_val = BlockMaskDataset(val_data, args.gamma, args.mask_ratio, args.block_size)


    elif args.model_name == 'ResUNet':
        train_noisy = data_load_process(train_path, 'noisy', args.spectrum_length)
        train_gt = data_load_process(train_path, 'gt', args.spectrum_length)
        
        val_noisy = data_load_process(val_path, 'noisy', args.spectrum_length)
        val_gt = data_load_process(val_path, 'gt', args.spectrum_length)
        
        ds_train = RamanDataset(
            inputs=train_noisy,
            outputs=train_gt,
            spectrum_len=args.spectrum_length,
            spectrum_shift=0.1,
            spectrum_window=False, 
            horizontal_flip=False, 
            mixup=True
        )
        ds_val = RamanDataset(
            inputs=val_noisy,
            outputs=val_gt,
            spectrum_len=args.spectrum_length,
            spectrum_shift=0.0,  
            spectrum_window=False, 
            horizontal_flip=False, 
            mixup=False 
        )
    elif args.model_name == 'AUNet':
        train_noisy = data_load_process(train_path, 'noisy', args.spectrum_length)
        train_gt = data_load_process(train_path, 'gt', args.spectrum_length)
        
        val_noisy = data_load_process(val_path, 'noisy', args.spectrum_length)
        val_gt = data_load_process(val_path, 'gt', args.spectrum_length)
        
        ds_train = BaseDataset(train_noisy, train_gt)
        ds_val = BaseDataset(val_noisy, val_gt)

    sampler_train = DistributedSampler(ds_train) if args.ddp else None
    sampler_val = DistributedSampler(ds_val, shuffle=False) if args.ddp else None

    loader_cfg = dict(
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        pin_memory=True,
        drop_last=True
    )
    
    val_cfg = loader_cfg.copy()
    val_cfg['drop_last'] = False
    
    train_loader = DataLoader(ds_train, sampler=sampler_train, shuffle=(sampler_train is None), **loader_cfg)
    val_loader = DataLoader(ds_val, sampler=sampler_val, shuffle=False, **val_cfg)

    return train_loader, val_loader, sampler_train