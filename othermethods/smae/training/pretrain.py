import argparse
import os
import torch
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
    parser.add_argument('--num_workers', '--num-workers', default=12, type=int)
    parser.add_argument('--patience', default=800, type=int)
    parser.add_argument('--resume', default='', type=str)

    return parser


def main(args, run_context=None):

    normalize_paths(args)
    validate_training(args)
    seed_everything(args.seed)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print('training device: {}'.format(device))

    # Pathogenic dataset unsupervise learning pretraining
    # dataset_train = Pathogenic_Unsupervise(spectra_path=args.data_path+'/X_'+args.dataset+'.npy', transform=train_transform)
    dataset_train = Pathogenic_Unsupervise_h5(spectra_path=args.data_path, transform=train_transform)  # 加载基准测试集
    data_loader_unsupervise = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)

    # initalize model
    model = spectra_mae.__dict__[args.model]().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    sheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)

    start_epoch = 1
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            sheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = int(checkpoint.get('epoch', 0)) + 1
        for group in optimizer.param_groups:
            group['weight_decay'] = args.wd
    if start_epoch > args.epoch:
        raise ValueError(f'Resume starts at epoch {start_epoch}, beyond requested final epoch {args.epoch}.')

    os.makedirs(args.save_path, exist_ok=True)

    print(f'Start training for {args.epoch} epochs: ')
    es = EarlyStop(patience=args.patience, mode='min')
    if checkpoint and 'early_stopping' in checkpoint:
        for key, value in checkpoint['early_stopping'].items():
            setattr(es, key, value)
    snapshot = save_run_config(args, args.save_path, 'train', run_context)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=args.log_dir)

    try:
        for epoch in range(start_epoch, args.epoch+1):
            loss_avg = Engine.train_unsupervise(epoch, data_loader_unsupervise, model, optimizer, sheduler, device, args=args, writer=writer)
            ep_savename = os.path.join(args.save_path, f"epoch_{epoch}.pth")
            es(loss_avg, model, ep_savename, epoch)
            state = training_state(model, optimizer, sheduler, epoch, loss_avg, args)
            state['early_stopping'] = {
                'best_score': float(es.best_score), 'val_score': float(es.val_score),
                'counter': es.counter, 'early_stop': bool(es.early_stop),
            }
            torch.save(state, os.path.join(args.save_path, 'last.pth'))
            if es.early_stop:
                break
    finally:
        writer.close()

    save_name = os.path.join(args.save_path, f"epoch_{epoch}_final.pth")
    torch.save(model.state_dict(), save_name)
    summary = {
        'last_checkpoint': os.path.join(args.save_path, 'last.pth'),
        'final_weights': save_name, 'epochs_completed': epoch, 'loss': float(loss_avg),
        'weight_decay': [group['weight_decay'] for group in optimizer.param_groups],
        'config': str(snapshot), 'log_dir': args.log_dir,
    }
    save_summary(args.save_path, summary)
    return summary


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
