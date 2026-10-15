#!/usr/bin/env python3
"""Run the project-adapted SMAE training and numerical evaluation workflows."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path = [str(REPO_ROOT)] + [
    entry for entry in sys.path
    if Path(entry or ".").resolve() not in (REPO_ROOT, PROJECT_ROOT)
]

MODULES = {
    "train": "othermethods.smae.training.pretrain",
    "train-ddp": "othermethods.smae.training.pretrain_ddp",
    "evaluate": "othermethods.smae.inference.evaluate",
    "run": "othermethods.smae.inference.pipeline",
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    root_parser = argparse.ArgumentParser(description=__doc__)
    root_parser.add_argument("command", choices=MODULES)
    command = root_parser.parse_args(argv[:1]).command

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "benchmark.yaml"
    )
    config_parser.add_argument("--print-config", action="store_true")
    shared, overrides = config_parser.parse_known_args(argv[1:])
    module = importlib.import_module(MODULES[command])
    parser = argparse.ArgumentParser(
        description=f"SMAE {command}; YAML defaults with explicit CLI overrides.",
        parents=[module.get_args_parser()],
    )
    if "--help" in overrides or "-h" in overrides:
        parser.parse_args(overrides)
        return

    from othermethods.smae.utils.config import load_defaults, normalize_paths

    defaults = load_defaults(shared.config, "evaluate" if command == "evaluate" else "train")
    if command == "run":
        evaluation = load_defaults(shared.config, "evaluate")
        defaults.update(input_path=evaluation['input_path'],
                        evaluation_dir=evaluation['output_dir'],
                        evaluation_batch_size=evaluation.get('batch_size', defaults['batch_size']))
    known = {action.dest for action in parser._actions}
    unknown = set(defaults) - known
    if unknown:
        parser.error(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    parser.set_defaults(**defaults)
    options = normalize_paths(parser.parse_args(overrides))
    if shared.print_config:
        print(json.dumps(vars(options), indent=2))
        return
    context = {'source_config': str(shared.config.expanduser().resolve()), 'argv': argv}
    if command == "evaluate":
        module.test(**vars(options), run_context=context)
    else:
        module.main(options, run_context=context)


if __name__ == "__main__":
    main()
