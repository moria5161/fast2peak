import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ExponentialLR

from OtherMethods.smae.utils.data import Pathogenic_Unsupervise,Pathogenic_Unsupervise_h5, train_transform
from OtherMethods.smae.model import Spectra_MAE as spectra_mae
from OtherMethods.smae.utils.Engine import Engine, EarlyStop, seed_everything


def get_args_parser():
    parser = argparse.ArgumentParser('Raman MAE pre-training', add_help=False)

    parser.add_argument('--data_path', default='/home/RenPengju/codes/Raman_spectra/Dataset/bacteria_ID', type=str)
    parser.add_argument('--dataset', default='reference', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epoch', default=800, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)

    parser.add_argument('--model', default='spectraMAE_base_patch100', type=str)
    parser.add_argument('--mask_ratio', default=0.3, type=float)
    parser.add_argument('--save_path', default='param/', type=str)
    parser.add_argument('--patience', default=800, type=int)

    # DDP parameters
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--resume', default='', type=str,
                        help='Path to checkpoint for resuming training')

    return parser


def main(args):
    # Initialize process group
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    seed_everything(615)

    if local_rank == 0:
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print('{}'.format(args).replace(', ', ',\n'))
        print('training device: {}'.format(device))

    # Pathogenic dataset unsupervise learning pretraining
    # dataset_train = Pathogenic_Unsupervise(spectra_path=args.data_path+'/X_'+args.dataset+'.npy', transform=train_transform)
    dataset_train = Pathogenic_Unsupervise_h5(spectra_path=args.data_path, transform=train_transform)  # 加载基准测试集
    sampler = DistributedSampler(dataset_train, shuffle=True)
    data_loader_unsupervise = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=1, sampler=sampler, pin_memory=True)

    # initalize model
    model = spectra_mae.__dict__[args.model]().to(device)

    start_epoch = 1
    best_loss = None
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if local_rank == 0:
            print(f'Resuming from checkpoint: {args.resume}')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if local_rank == 0:
                print('Loaded full checkpoint (model + optimizer + scheduler + epoch)')
        else:
            model.load_state_dict(checkpoint)
            if local_rank == 0:
                print('Loaded model weights from checkpoint')

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
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

    if local_rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        print(f'Start training for {args.epoch} epochs (starting from epoch {start_epoch}): ')

    es = EarlyStop(patience=args.patience, mode='min')
    if best_loss is not None:
        es.val_score = best_loss
        es.best_score = -1. * best_loss
        if local_rank == 0:
            print(f'EarlyStop state restored with best loss: {best_loss:.6f}')

    from torch.utils.tensorboard import SummaryWriter

    log_dir = os.path.join('logs', args.save_path)
    if args.resume and local_rank == 0:
        import datetime
        log_dir = log_dir.rstrip('/') + f'_resumed_from_epoch{start_epoch}'
        print(f'Resume log dir: {log_dir}')
    writer = SummaryWriter(log_dir=log_dir) if local_rank == 0 else None

    for epoch in range(start_epoch, args.epoch+1):
        sampler.set_epoch(epoch)
        loss_avg = Engine.train_unsupervise(epoch, data_loader_unsupervise, model, optimizer, sheduler, device, args=args, writer=writer)

        if local_rank == 0:
            ep_savename = os.path.join(args.save_path, f"epoch_{epoch}.pth")
            es(loss_avg, model.module, ep_savename, epoch)
            # Save full checkpoint alongside for future resume
            full_ckpt_path = os.path.join(args.save_path, f"epoch_{epoch}_full.pth")
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': sheduler.state_dict(),
                'epoch': epoch,
                'loss': loss_avg,
            }, full_ckpt_path)
        else:
            es(loss_avg, model.module, os.devnull, epoch)

        if es.early_stop:
            break

    if local_rank == 0:
        save_name = os.path.join(args.save_path, f"epoch_{epoch}_final.pth")
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': sheduler.state_dict(),
            'epoch': epoch,
            'loss': loss_avg,
        }, save_name)

    dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
