"""Load and normalize portable experiment defaults."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_OPTIONS = {"data_path", "save_path", "log_dir", "input_path", "output_dir", "chkpt_dir", "resume", "evaluation_dir"}


def load_defaults(path: Path, section: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get(section), dict):
        raise ValueError(f"{path}: expected a '{section}' mapping")
    options = dict(config[section])
    for key in PATH_OPTIONS & options.keys():
        if options[key]:
            value = Path(options[key]).expanduser()
            options[key] = str(value if value.is_absolute() else (PROJECT_ROOT / value).resolve())
    return options


def normalize_paths(options):
    """YAML paths are already absolute; resolve explicit CLI paths against cwd."""
    for key in PATH_OPTIONS & vars(options).keys():
        if getattr(options, key):
            setattr(options, key, str(Path(getattr(options, key)).expanduser().resolve()))
    return options
