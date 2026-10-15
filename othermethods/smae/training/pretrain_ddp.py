import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ExponentialLR

from othermethods.smae.utils.data import Pathogenic_Unsupervise,Pathogenic_Unsupervise_h5, train_transform
from othermethods.smae.models import Spectra_MAE as spectra_mae
from othermethods.smae.utils.Engine import Engine, EarlyStop, seed_everything
from othermethods.smae.utils.config import normalize_paths
from othermethods.smae.utils.runtime import save_run_config, save_summary, training_state, validate_training


def get_args_parser():
    parser = argparse.ArgumentParser('Raman MAE pre-training', add_help=False)

    parser.add_argument('--data_path', '--data-path', default='datasets/train.h5', type=str)
    parser.add_argument('--dataset', default='reference', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epoch', '--epochs', default=800, type=int)
    parser.add_argument('--batch_size', '--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)

    parser.add_argument('--model', default='spectraMAE_base_ss1600_ps4', type=str)
    parser.add_argument('--mask_ratio', '--mask-ratio', default=0.3, type=float)
    parser.add_argument('--save_path', '--output-dir', default='outputs/retrained/default', type=str)
    parser.add_argument('--log_dir', '--log-dir', default='outputs/logs/retrained/default', type=str)
    parser.add_argument('--seed', default=615, type=int)
    parser.add_argument('--num_workers', '--num-workers', default=1, type=int)
    parser.add_argument('--patience', default=800, type=int)

    # DDP parameters
    parser.add_argument('--local_rank', '--local-rank', default=-1, type=int)
    parser.add_argument('--resume', default='', type=str,
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--backend', choices=('auto', 'nccl', 'gloo'), default='auto')

    return parser


def main(args, run_context=None):
    normalize_paths(args)
    validate_training(args)
    if 'LOCAL_RANK' not in os.environ:
        raise RuntimeError('train-ddp must be launched with torchrun.')
    # Initialize process group
    local_rank = int(os.environ["LOCAL_RANK"])
    use_cuda = torch.device(args.device).type == 'cuda'
    backend = ('nccl' if use_cuda else 'gloo') if args.backend == 'auto' else args.backend
    if backend == 'nccl' and not use_cuda:
        raise ValueError('NCCL requires CUDA; use --backend gloo for CPU verification.')
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}' if use_cuda else 'cpu')
    args.backend = backend
    args.device = str(device)
    dist.init_process_group(backend=backend)

    seed_everything(args.seed)

    if local_rank == 0:
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print('{}'.format(args).replace(', ', ',\n'))
        print('training device: {}'.format(device))

    # Pathogenic dataset unsupervise learning pretraining
    # dataset_train = Pathogenic_Unsupervise(spectra_path=args.data_path+'/X_'+args.dataset+'.npy', transform=train_transform)
    dataset_train = Pathogenic_Unsupervise_h5(spectra_path=args.data_path, transform=train_transform)  # 加载基准测试集
    sampler = DistributedSampler(dataset_train, shuffle=True)
    data_loader_unsupervise = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler, pin_memory=True)

    # initalize model
    model = spectra_mae.__dict__[args.model]().to(device)

    start_epoch = 1
    best_loss = None
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if local_rank == 0:
            print(f'Resuming from checkpoint: {args.resume}')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict({key.removeprefix('module.'): value for key, value in checkpoint['model_state_dict'].items()})
            if local_rank == 0:
                print('Loaded full checkpoint (model + optimizer + scheduler + epoch)')
        else:
            model.load_state_dict(checkpoint)
            if local_rank == 0:
                print('Loaded model weights from checkpoint')

    model = DDP(model, device_ids=[local_rank], output_device=local_rank) if use_cuda else DDP(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    sheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)

    if checkpoint is not None:
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if local_rank == 0:
                print('Optimizer state restored.')
        if isinstance(checkpoint, dict) and 'scheduler_state_dict' in checkpoint:
            sheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if local_rank == 0:
                print('Scheduler state restored.')
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            if local_rank == 0:
                print(f'Resuming from epoch {start_epoch}')
        if isinstance(checkpoint, dict) and 'loss' in checkpoint:
            best_loss = checkpoint['loss']
    for group in optimizer.param_groups:
        group['weight_decay'] = args.wd
    if start_epoch > args.epoch:
        dist.destroy_process_group()
        raise ValueError(f'Resume starts at epoch {start_epoch}, beyond final epoch {args.epoch}.')

    if local_rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        print(f'Start training for {args.epoch} epochs (starting from epoch {start_epoch}): ')

    es = EarlyStop(patience=args.patience, mode='min')
    if best_loss is not None:
        es.val_score = best_loss
        es.best_score = -1. * best_loss
        if local_rank == 0:
            print(f'EarlyStop state restored with best loss: {best_loss:.6f}')
    if checkpoint and 'early_stopping' in checkpoint:
        for key, value in checkpoint['early_stopping'].items():
            setattr(es, key, value)

    from torch.utils.tensorboard import SummaryWriter

    log_dir = args.log_dir
    if args.resume:
        log_dir = log_dir.rstrip('/') + f'_resumed_from_epoch{start_epoch}'
        if local_rank == 0:
            print(f'Resume log dir: {log_dir}')
    args.log_dir = log_dir
    writer = SummaryWriter(log_dir=log_dir) if local_rank == 0 else None
    if local_rank == 0:
        context = {**(run_context or {}), 'world_size': dist.get_world_size(), 'backend': backend}
        snapshot = save_run_config(args, args.save_path, 'train-ddp', context)

    for epoch in range(start_epoch, args.epoch+1):
        sampler.set_epoch(epoch)
        loss_avg = Engine.train_unsupervise(epoch, data_loader_unsupervise, model, optimizer, sheduler, device, args=args, writer=None)
        totals = torch.tensor([loss_avg * len(sampler), len(sampler)], dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        loss_avg = (totals[0] / totals[1]).item()

        if local_rank == 0:
            ep_savename = os.path.join(args.save_path, f"epoch_{epoch}.pth")
            es(loss_avg, model.module, ep_savename, epoch)
            writer.add_scalar('train_loss', loss_avg, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            # Save full checkpoint alongside for future resume.
            full_ckpt_path = os.path.join(args.save_path, f"epoch_{epoch}_full.pth")
            state = training_state(model.module, optimizer, sheduler, epoch, loss_avg, args)
            state['early_stopping'] = {
                'best_score': float(es.best_score), 'val_score': float(es.val_score),
                'counter': es.counter, 'early_stop': bool(es.early_stop),
            }
            torch.save(state, full_ckpt_path)
            torch.save(state, os.path.join(args.save_path, 'last.pth'))

        stopping = torch.tensor(int(bool(es.early_stop)) if local_rank == 0 else 0, device=device)
        dist.broadcast(stopping, src=0)
        if stopping.item():
            break

    if local_rank == 0:
        save_name = os.path.join(args.save_path, f"epoch_{epoch}_final.pth")
        torch.save(state, save_name)
        save_summary(args.save_path, {
            'last_checkpoint': os.path.join(args.save_path, 'last.pth'),
            'final_weights': save_name, 'epochs_completed': epoch, 'loss': loss_avg,
            'weight_decay': [group['weight_decay'] for group in optimizer.param_groups],
            'config': str(snapshot), 'log_dir': log_dir, 'world_size': dist.get_world_size(),
        })

    if writer:
        writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
