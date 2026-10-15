#!/usr/bin/env python3
"""Unified entry point for the final Maybridge 1H NMR benchmark."""

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
sys.path = [str(REPO_ROOT)] + [entry for entry in sys.path
    if Path(entry or ".").resolve() not in (REPO_ROOT, PROJECT_ROOT)]

from otherbench.NMR.utils.config import DEFAULT_CONFIG, load_config, command_arguments


def main(argv=None):
    modules = {"source": "utils.source", "split": "utils.splits", "prepare": "utils.preprocess",
               "train-f2p": "training.f2p", "train-dn-unet": "training.dn_unet", "train-ld-net": "training.ld_net",
               "evaluate": "inference.evaluate", "assemble": "inference.assemble", "export": "inference.export"}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=tuple(modules))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args, overrides = parser.parse_known_args(argv)
    path = args.config.resolve()
    config = load_config(path)
    module = importlib.import_module("otherbench.NMR." + modules[args.command])
    module.main(command_arguments(args.command, config, path) + overrides)


if __name__ == "__main__":
    main()
