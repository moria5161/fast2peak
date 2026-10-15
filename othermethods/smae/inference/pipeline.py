"""Train and evaluate the exact checkpoint produced by the same invocation."""

import argparse
from pathlib import Path

from othermethods.smae.training import pretrain
from othermethods.smae.inference.evaluate import test
from othermethods.smae.utils.runtime import save_run_config, save_summary


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False, parents=[pretrain.get_args_parser()])
    parser.add_argument('--input_path', '--input-path', default='datasets/test/Simu.h5')
    parser.add_argument('--evaluation_dir', '--evaluation-dir', default='outputs/evaluation/default')
    parser.add_argument('--evaluation_batch_size', '--evaluation-batch-size', type=int, default=24)
    return parser


def main(args, run_context=None):
    if not Path(args.input_path).is_file() and not Path(args.input_path).is_dir():
        raise FileNotFoundError(f'Evaluation input not found: {args.input_path}')
    if args.evaluation_batch_size <= 0:
        raise ValueError('evaluation_batch_size must be positive.')
    train_keys = {action.dest for action in pretrain.get_args_parser()._actions}
    training = argparse.Namespace(**{key: value for key, value in vars(args).items() if key in train_keys})
    result = pretrain.main(training, run_context=run_context)
    save_run_config(args, args.save_path, 'run', run_context, filename='pipeline_config.yaml')
    evaluation = test(
        input_path=args.input_path, output_dir=args.evaluation_dir,
        chkpt_dir=result['last_checkpoint'], arch=None,
        batch_size=args.evaluation_batch_size, seed=args.seed,
        mask_ratio=None, device=args.device, run_context=run_context,
    )
    summary = {'training': result, 'evaluation': evaluation}
    save_summary(args.save_path, summary, filename='pipeline_summary.json')
    return summary
