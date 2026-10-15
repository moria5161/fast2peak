# -*- encoding: utf-8 -*-
'''
@File        :main.py
@Description :
@InitTime    :2026/05/01 19:00:00
@Author      :Moria
@EMail       :zhangnannan@stu.xmu.edu.cn

'''

import torch
import os
import argparse
import yaml
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.train_test import train, test
from utils.base import seed_everything, load_state
from models import build_model

import warnings
warnings.simplefilter("ignore", FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='Raman spectra Denoising')

    group_exp = parser.add_argument_group('Experiment')
    group_exp.add_argument('--exp_name', default='default_exp', type=str)
    group_exp.add_argument('--log_dir', default='./logs', type=str)
    group_exp.add_argument('--seed', default=615, type=int)

    group_model = parser.add_argument_group('Model')
    group_model.add_argument('--model_name', default='F2P', type=str)
    group_model.add_argument('--spectrum_length', default=1600, type=int)
    group_model.add_argument('--patch_size', default=16, type=int)
    group_model.add_argument('--embed_dim', default=256, type=int)
    group_model.add_argument('--depth', default=8, type=int)
    group_model.add_argument('--heads', default=8, type=int)
    
    group_model.add_argument('--num_convs', default=8, type=int)
    group_model.add_argument('--ks', default=3, type=int)

    group_alg = parser.add_argument_group('Algorithm')
    group_alg.add_argument('--mask_ratio', default=0.25, type=float)
    group_alg.add_argument('--gamma', default=11, type=int)
    group_alg.add_argument('--block_size', default=8, type=int)
    
    group_alg.add_argument('--smae_mask_ratio', default=0.50, type=float)

    group_train = parser.add_argument_group('Training')
    group_train.add_argument('--epochs', default=1000, type=int)
    group_train.add_argument('--batch_size', default=512, type=int)
    group_train.add_argument('--lr', default=1e-4, type=float)
    group_train.add_argument('--min_lr', default=None, type=float)
    group_train.add_argument('--warmup_epochs', default=10, type=int)
    group_train.add_argument('--patience', default=None, type=int)
    group_train.add_argument('--weight_decay', default=1e-3, type=float)

    group_data = parser.add_argument_group('Data')
    group_data.add_argument('--train_data_dir', default='', type=str)
    group_data.add_argument('--fine_tune_data_path', default='', type=str)
    group_data.add_argument('--test_datapath', default='', type=str)
    group_data.add_argument('--test_batch_size', default=1024, type=int)
    group_data.add_argument('--workers', default=4, type=int)

    group_mode = parser.add_argument_group('Mode')
    group_mode.add_argument('--train', action='store_true')
    group_mode.add_argument('--sft', action='store_true', help='Supervised F2P training with paired noisy/gt spectra')
    group_mode.add_argument('--test', action='store_true')
    group_mode.add_argument('--ddp', action='store_true')

    group_load = parser.add_argument_group('Loading')
    group_load.add_argument('--resume', default='', type=str)
    group_load.add_argument('--pretrained', default='', type=str)
    group_load.add_argument('--test_model_path', default='', type=str)

    args = parser.parse_args()
    return args


def load_config(args):
    old_model_names = {
        'Fast2PeakRPE': 'F2P',
        'Fast2PeakRPEHeadFT': 'F2PHeadFT',
        'Fast2PeakConv': 'F2PConv',
        'Fast2PeakOverlap': 'F2POverlap',
    }
    args.model_name = old_model_names.get(args.model_name, args.model_name)
    target_path = None
    if args.resume:
        target_path = args.resume
    elif args.test and args.test_model_path:
        target_path = args.test_model_path
    elif args.sft and args.pretrained:
        target_path = args.pretrained

    if not target_path:
        return args
    
    checkpoint_dir = os.path.dirname(target_path)
    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    structure_keys = [
            'model_name', 'spectrum_length', 'embed_dim', 
            'depth', 'heads', 'patch_size', 'gamma', 'num_convs', 'ks'
        ]
    if args.sft and args.pretrained and not args.resume and not args.test and args.model_name in ['F2PHeadFT', 'F2PHeadRes']:
        structure_keys.remove('model_name')
    for key in structure_keys:
        if key in config_dict and hasattr(args, key):
            setattr(args, key, config_dict[key])
    args.model_name = old_model_names.get(args.model_name, args.model_name)
    if args.resume or (args.test and args.test_model_path):
        saved_sft = config_dict.get('sft', False)
        if args.resume and args.sft and not saved_sft:
            raise ValueError('Use --pretrained to start SFT from a self-supervised checkpoint, not --resume.')
        args.sft = saved_sft
    print(f"[Config] Restored model structure from: {config_path}")
    return args


def init_device(args):
    if args.ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = local_rank
        return device, local_rank, world_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, 0, 1


def init_logs(args):
    if args.train:
        log_dir = os.path.join(args.log_dir, args.exp_name)
        checkpoint_dir = os.path.join('./experiments', args.exp_name, 'checkpoints')

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        config_path = os.path.join(checkpoint_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)
        print(f"Config saved to: {config_path}")
        return log_dir, checkpoint_dir
    
    if args.test:
        target_dir = os.path.dirname(os.path.dirname(args.test_model_path))
        output_dir = os.path.join(target_dir, 'output', datetime.date.today().strftime("%Y-%m-%d"))
        os.makedirs(output_dir, exist_ok=True)
        return output_dir


def init_model(args, device, local_rank):
        
    model = build_model(**vars(args))
    model.to(device)    
    
    load_path = None
    strict = False
    if args.resume:
        load_path = args.resume
        if local_rank == 0: print(f"[Model] Resuming from {load_path}")

    elif args.pretrained:
        load_path = args.pretrained
        if local_rank == 0: print(f"[Model] Fine-tuning from {load_path}")

    elif args.test and args.test_model_path:
        load_path = args.test_model_path
        strict = True
        if local_rank == 0: print(f"[Model] Testing model: {load_path}")

    if load_path:
        map_loc = f'cuda:{device}' if isinstance(device, int) else device
        checkpoint = torch.load(load_path, map_location=map_loc)
        if args.sft:
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
            if args.model_name == 'F2PHeadRes' and args.pretrained and not args.resume and not args.test:
                head_state = {k: v for k, v in model.state_dict().items()
                              if k.startswith(('head_norm.', 'res_head.'))}
                if not any(k in state_dict for k in head_state):
                    state_dict.update(head_state)
            model.load_state_dict(state_dict, strict=True)
        else:
            load_state(model, checkpoint, ddp=args.ddp, strict=strict)
    
    # if args.pretrained and not args.test: 
    #     frozen_count = 0
    #     for name, param in model.named_parameters():
    #         if 'relative_position_bias_table' in name:
    #             param.requires_grad = False
    #             frozen_count += 1
    #         # if 'decoder' not in name:
    #         #     param.requires_grad = False
    #         #     frozen_count += 1
                
    
    if args.ddp:
        model = DDP(model, device_ids=[local_rank])
    
    return model


def main():
    args = parse_args()
    args = load_config(args)
    if args.sft and args.model_name not in ['F2P', 'F2PHeadFT', 'F2PHeadRes', 'F2PConv', 'F2POverlap']:
        raise ValueError('--sft currently supports F2P models only.')
    
    if args.seed is not None:
        seed_everything(args.seed)
    device, local_rank, world_size = init_device(args)
    
    if local_rank == 0:
        print(f"Using device: {'DDP on GPU' if args.ddp else device}")
    model = init_model(args, device, local_rank)

    try:
        if args.train:
            log_dir, checkpoint_dir = None, None # Initialize for non-DDP case
            if local_rank == 0:
                log_dir, checkpoint_dir = init_logs(args)
            train(model, args, device, local_rank, world_size, log_dir, checkpoint_dir)

        if args.test:
            output_dir = None
            if local_rank == 0:
                output_dir = init_logs(args)
            test(model, args, device, output_dir)

    finally:
        if args.ddp:
            dist.destroy_process_group()
        
        
if __name__ == '__main__':
    main()
