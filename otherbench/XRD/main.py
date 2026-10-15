#!/usr/bin/env python3
"""Unified entry point for PXRD data preparation, evaluation and metric export."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Exclude the script directory: its regular utils package would otherwise
# shadow the shared repository's namespace-package utils despite root priority.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path = [str(REPO_ROOT)] + [
    entry for entry in sys.path if Path(entry or ".").resolve() not in (REPO_ROOT, PROJECT_ROOT)
]

from otherbench.XRD.utils.config import DEFAULT_CONFIG, load_config, command_arguments


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "export"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args, overrides = parser.parse_known_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    modules = {
        "prepare": "otherbench.XRD.utils.preprocess",
        "evaluate": "otherbench.XRD.inference.evaluate",
        "export": "otherbench.XRD.inference.export",
    }
    module = importlib.import_module(modules[args.command])
    module.main(command_arguments(args.command, config, config_path) + overrides)


if __name__ == "__main__":
    main()
