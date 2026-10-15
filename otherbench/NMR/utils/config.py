"""Configuration and paths for the final Maybridge benchmark."""

from pathlib import Path
import yaml

NMR_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = NMR_ROOT / "configs/benchmark.yaml"


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else NMR_ROOT / path).resolve()


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    for section, keys in {"f2p": ["checkpoint"],
                          "dn_unet": ["converted_checkpoint", "trained_checkpoint"],
                          "ld_net": ["checkpoint", "trained_checkpoint"]}.items():
        for key in keys:
            config[section][key] = str(resolve_path(config[section][key]))
    return config


def flatten(options):
    result = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif value is not None:
            result.extend([flag, *map(str, value)] if isinstance(value, list) else [flag, str(value)])
    return result


def command_arguments(command, config, path):
    p = {key: resolve_path(value) for key, value in config["paths"].items()}
    levels = [float(row["sigma"]) for row in config["noise_conditions"]]
    if command == "source":
        options = {"spectra_txt": p["raw_spectra"], "molecule_sdf": p["molecule_sdf"], "output_dir": p["source_data"]}
    elif command == "split":
        options = {"source_h5": p["source_data"] / "clean.h5", "output_dir": p["splits"].parent, "seed": config["seed"]}
    elif command == "prepare":
        options = {"splits": p["splits"], "output": p["benchmark_data"],
                   "validation_output": p["validation_data"], "seed": config["seed"], "noise_levels": levels}
    elif command == "evaluate":
        options = {"config": path, "data": p["benchmark_data"], "output_dir": p["evaluation_output"],
                   "device": config["inference"]["device"]}
    elif command == "export":
        options = {"evaluation_dir": p["evaluation_output"], "output_dir": p["public_docs"]}
    elif command == "assemble":
        options = {}
    elif command.startswith("train-"):
        model = command[6:].replace("-", "_")
        options = {"splits": p["splits"], **config["training"][model]}
        for key in ("output_dir", "pretrained"):
            if key in options:
                options[key] = resolve_path(options[key])
        if model != "f2p":
            options["validation_data"] = p["validation_data"]
    else:
        raise ValueError(command)
    return flatten(options)
