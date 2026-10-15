"""Configuration and paths scoped to the IR subproject."""

from pathlib import Path
import yaml


IR_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = IR_ROOT.parent.parent
DEFAULT_CONFIG = IR_ROOT / "configs" / "benchmark.yaml"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return config


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else IR_ROOT / path).resolve()


def command_arguments(command: str, config: dict) -> list[str]:
    paths = {key: resolve_path(value) for key, value in config["paths"].items()}
    if command == "prepare":
        options = {
            "output-dir": paths["paired_data"],
            "reference-dir": paths["paired_data"],
            **{key.replace("_", "-"): value for key, value in config["data"].items()},
        }
    elif command == "train":
        options = {
            "data-dir": paths["paired_data"],
            "output-dir": paths["autoencoder_output"],
            **{key.replace("_", "-"): value for key, value in config["training"].items()},
            "savgol-window": config["savgol"]["window_length"],
            "savgol-order": config["savgol"]["polyorder"],
        }
    elif command == "evaluate":
        options = {
            "data-dir": paths["paired_data"],
            "paper-nn-dir": paths["autoencoder_output"],
            "output-dir": paths["evaluation_output"],
            "checkpoint": paths["f2p_checkpoint"],
            **{key.replace("_", "-"): value for key, value in config["inference"].items()},
            "savgol-window": config["savgol"]["window_length"],
            "savgol-order": config["savgol"]["polyorder"],
        }
    elif command == "export":
        options = {"evaluation-dir": paths["evaluation_output"], "output-dir": paths["public_docs"]}
    elif command == "pack":
        return []
    else:
        raise ValueError(command)
    return [item for key, value in options.items() for item in (f"--{key}", str(value))]
