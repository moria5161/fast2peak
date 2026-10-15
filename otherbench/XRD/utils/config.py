"""YAML configuration with paths scoped to the XRD subproject."""

from pathlib import Path
import yaml

XRD_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = XRD_ROOT / "configs" / "benchmark.yaml"


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else XRD_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    config["f2p"]["checkpoint"] = str(resolve_path(config["f2p"]["checkpoint"]))
    return config


def command_arguments(command: str, config: dict, config_path: Path) -> list[str]:
    paths = {key: resolve_path(value) for key, value in config["paths"].items()}
    if command == "prepare":
        options = {"config": config_path, "input-dir": paths["raw_data"], "output-dir": paths["benchmark_data"]}
    elif command == "evaluate":
        options = {
            "config": config_path,
            "data": paths["benchmark_data"] / "benchmark.h5",
            "output-dir": paths["evaluation_output"],
            "device": config["inference"]["device"],
        }
    elif command == "export":
        options = {"evaluation-dir": paths["evaluation_output"], "data-dir": paths["benchmark_data"], "output-dir": paths["public_docs"]}
    else:
        raise ValueError(command)
    return [item for key, value in options.items() for item in (f"--{key}", str(value))]
