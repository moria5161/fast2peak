"""Numerical evaluation of project-adapted SMAE checkpoints; no plotting."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from othermethods.smae.models import Spectra_MAE as spectra_mae
from othermethods.smae.utils.Engine import seed_everything
from othermethods.smae.utils.metrics import calculate_metrics
from othermethods.smae.utils.runtime import save_run_config, save_summary, file_sha256


def prepare_model(chkpt_dir, arch=None, device='cuda:0'):
    checkpoint = torch.load(chkpt_dir, map_location=device, weights_only=True)
    config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    arch = arch or config.get('model')
    if not arch:
        raise ValueError('This checkpoint has no model metadata; specify --arch for historical bare weights.')
    model = getattr(spectra_mae, arch)()
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    state = {key.removeprefix('module.'): value for key, value in checkpoint.items()}
    model.load_state_dict(state, strict=True)
    model._smae_checkpoint_config = config
    model._smae_arch = arch
    return model.to(device).eval()


@torch.inference_mode()
def model_inference_batch(y, model_mae, batch_size=64, mask_ratio=0.50):
    y = np.asarray(y)
    expected_length = model_mae.num_patches * model_mae.patch_size
    if y.ndim != 2 or len(y) == 0 or y.shape[1] != expected_length:
        raise ValueError(f'Expected nonempty [N, {expected_length}] spectra, got {y.shape}; no implicit interpolation is applied.')
    if batch_size <= 0 or not 0 < mask_ratio <= 1:
        raise ValueError('batch_size must be positive and mask_ratio must be in (0, 1].')
    if not np.isfinite(y).all():
        raise ValueError('Input spectra contain NaN or Inf.')
    device = next(model_mae.parameters()).device
    spectra_tensor = torch.from_numpy(y).float().to(device).unsqueeze(1)
    predictions = []
    for start in range(0, len(y), batch_size):
        batch = spectra_tensor[start:start + batch_size]
        _, pred, _ = model_mae(batch, mask_ratio=mask_ratio)
        pred = model_mae.unpatchify(pred).detach().cpu().numpy().squeeze(1)
        predictions.append(pred)
    return np.vstack(predictions)


def model_inference(y, model_mae, mask_ratio=0.50):
    return model_inference_batch(np.asarray(y).reshape(1, -1), model_mae, 1, mask_ratio)[0]


def test(input_path, output_dir, chkpt_dir, arch=None,
         batch_size=64, seed=615, mask_ratio=None, device='cuda:0', run_context=None):
    seed_everything(seed)
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if source.is_dir():
        files = sorted(file for file in source.iterdir() if file.is_file() and file.suffix.lower() in {'.h5', '.txt'})
    elif source.is_file() and source.suffix.lower() in {'.h5', '.txt'}:
        files = [source]
    else:
        raise ValueError(f'Expected an existing .h5/.txt file or input directory: {source}')
    if not files:
        raise ValueError(f'No supported input files found in {source}')
    checkpoint_path = Path(chkpt_dir).expanduser().resolve()
    targets = set()
    for file in files:
        out = destination / file.name
        if file == out or checkpoint_path == out:
            raise ValueError(f'Refusing to overwrite an input spectrum or checkpoint: {out}')
        candidates = [out, destination / f'{file.stem}_metrics_details.csv', destination / f'{file.stem}_eval.log']
        if targets.intersection(candidates):
            raise ValueError('Input basenames would produce colliding output files.')
        targets.update(candidates)

    model = prepare_model(checkpoint_path, arch=arch, device=device)
    arch = model._smae_arch
    if mask_ratio is None:
        mask_ratio = model._smae_checkpoint_config.get('mask_ratio')
    if mask_ratio is None:
        raise ValueError('This checkpoint has no masking metadata; specify --mask-ratio for historical weights.')
    if batch_size <= 0 or not 0 < mask_ratio <= 1:
        raise ValueError('batch_size must be positive and mask_ratio must be in (0, 1].')
    destination.mkdir(parents=True, exist_ok=True)
    effective = dict(input_path=str(source), output_dir=str(destination), chkpt_dir=str(checkpoint_path),
                     arch=arch, batch_size=batch_size, seed=seed, mask_ratio=mask_ratio, device=device)
    checkpoint_hash = file_sha256(checkpoint_path)
    context = {**(run_context or {}), 'checkpoint_sha256': checkpoint_hash}
    save_run_config(effective, destination, 'evaluate', context)
    records = []
    for file in files:
        axis = None
        gt = None
        if file.suffix.lower() == '.h5':
            with h5py.File(file, 'r') as handle:
                y = handle['noisy'][:]
                gt = handle['gt'][:] if 'gt' in handle else None
                if 'wavenumbers' in handle:
                    axis = handle['wavenumbers'][:]
        else:
            data = np.loadtxt(file)
            if data.ndim == 2 and data.shape[1] == 2:
                axis = data[:, 0]
                y = data[:, 1].astype(np.float32).reshape(1, -1)
            elif data.ndim == 1:
                axis = np.arange(len(data))
                y = data.astype(np.float32).reshape(1, -1)
            else:
                raise ValueError(f'{file}: TXT must contain one intensity column or two axis/intensity columns.')
        if gt is not None and gt.shape != y.shape:
            raise ValueError(f'{file}: noisy and gt shapes differ: {y.shape} / {gt.shape}')
        started = time.perf_counter()
        prediction = model_inference_batch(y, model, batch_size, mask_ratio)
        elapsed = time.perf_counter() - started
        out = destination / file.name
        if file.suffix.lower() == '.h5':
            with h5py.File(out, 'w') as handle:
                handle.create_dataset('denoised', data=prediction)
                if axis is not None:
                    handle.create_dataset('wavenumbers', data=axis)
        else:
            np.savetxt(out, np.column_stack((axis, prediction[0])), fmt='%.6f', delimiter='\t')
        print(f'{file.name}: {len(prediction)} spectra, {elapsed:.4f} s; saved {out}')
        record = {'input': str(file), 'prediction': str(out), 'spectra': len(prediction)}
        if gt is not None:
            metrics = np.array([calculate_metrics(prediction[i], gt[i]) for i in range(len(y))])
            np.savetxt(destination / f'{file.stem}_metrics_details.csv', metrics,
                       delimiter=',', header='SNR,RMSE,SSS', comments='')
            mean = np.mean(metrics, axis=0)
            record['mean_metrics'] = {
                key: float(value) if np.isfinite(value) else None
                for key, value in zip(('SNR_dB', 'RMSE', 'SSS'), mean)
            }
            with (destination / f'{file.stem}_eval.log').open('w', encoding='utf-8') as stream:
                stream.write(f'Model: {arch}\nCheckpoint: {checkpoint_path}\n')
                stream.write(f'File: {file.name}\nSpectra: {len(y)}\n')
                stream.write(f'Seed: {seed}\nMask ratio: {mask_ratio}\nDevice: {device}\n')
                stream.write(f'Inference time (s): {elapsed:.6f}\nSNR/dB\tRMSE\tSSS\n')
                stream.write(f'{mean[0]:.2f}\t{mean[1]:.4f}\t{mean[2]:.4f}\n')
        records.append(record)
    summary = {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': checkpoint_hash,
               'arch': arch, 'mask_ratio': mask_ratio, 'files': records}
    save_summary(destination, summary)
    return summary


def get_args_parser():
    parser = argparse.ArgumentParser('SMAE numerical evaluation', add_help=False)
    parser.add_argument('--input_path', '--input-path', default='datasets/test.h5')
    parser.add_argument('--output_dir', '--output-dir', default='outputs/evaluation')
    parser.add_argument('--chkpt_dir', '--checkpoint', required=False, default='outputs/pretrained/epoch_500_final.pth')
    parser.add_argument('--arch', default=None)
    parser.add_argument('--batch_size', '--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=615)
    parser.add_argument('--mask_ratio', '--mask-ratio', type=float, default=None)
    parser.add_argument('--device', default='cuda:0')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    test(**vars(parser.parse_args()))
