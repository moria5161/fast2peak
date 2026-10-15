"""Persist effective run settings and training state for reproducible execution."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import yaml


def arguments(options):
    values = vars(options) if hasattr(options, '__dict__') else options
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in values.items() if not key.startswith('_')}


def save_run_config(options, output_dir, command, context=None, filename='config.yaml'):
    """Save the resolved settings, plus an immutable snapshot for each invocation."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    record = {
        'command': command,
        'started_at_utc': now.isoformat(),
        'arguments': arguments(options),
        'context': context or {},
        'environment': {
            'python': sys.version.split()[0],
            'torch': str(torch.__version__),
            'cuda_available': torch.cuda.is_available(),
        },
    }
    text = yaml.safe_dump(record, sort_keys=False, allow_unicode=True)
    unique = f"{Path(filename).stem}_{now.strftime('%Y%m%dT%H%M%S%f')}_{os.getpid()}.yaml"
    (destination / unique).write_text(text, encoding='utf-8')
    (destination / filename).write_text(text, encoding='utf-8')
    return destination / filename


def save_summary(output_dir, summary, filename='summary.json'):
    path = Path(output_dir) / filename
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def training_state(model, optimizer, scheduler, epoch, loss, options):
    return {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': int(epoch),
        'loss': float(loss),
        'config': arguments(options),
    }


def validate_training(options):
    if options.epoch < 1 or options.batch_size < 1 or options.num_workers < 0:
        raise ValueError('epochs/batch_size must be positive and num_workers nonnegative.')
    if not 0 < options.mask_ratio <= 1 or options.wd < 0 or options.lr <= 0:
        raise ValueError('Require mask_ratio in (0, 1], wd >= 0 and lr > 0.')
    if not Path(options.data_path).is_file():
        raise FileNotFoundError(f'Training data not found: {options.data_path}')
    if (Path(options.save_path) / 'last.pth').exists() and not getattr(options, 'resume', ''):
        raise FileExistsError('This training directory already contains last.pth; use --resume or a new --output-dir.')
