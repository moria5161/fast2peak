#!/usr/bin/env python3
"""Unified entry point for the IR denoising benchmark."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


# Prevent local regular packages from shadowing shared namespace-package utils.
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path = [str(REPO_ROOT)] + [
    entry for entry in sys.path if Path(entry or ".").resolve() not in (REPO_ROOT, PROJECT_ROOT)
]

from otherbench.IR.utils.config import DEFAULT_CONFIG, load_config, command_arguments


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "export", "pack"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args, overrides = parser.parse_known_args(argv)
    config = load_config(args.config.resolve())
    modules = {
        "prepare": "otherbench.IR.utils.preprocess",
        "train": "otherbench.IR.training.train",
        "evaluate": "otherbench.IR.inference.evaluate",
        "export": "otherbench.IR.inference.export",
        "pack": "otherbench.IR.utils.pack_atr",
    }
    module = importlib.import_module(modules[args.command])
    # Explicit command-line options follow YAML defaults, so argparse takes
    # their final occurrence as the override.
    module.main(command_arguments(args.command, config) + overrides)


if __name__ == "__main__":
    main()
