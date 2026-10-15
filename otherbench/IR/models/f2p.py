"""Adapter for the shared repository Fast2PeakRPE architecture."""

from pathlib import Path
import torch
import yaml


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    from models import build_model, model_registry

    config_path = checkpoint_path.parent / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = ("model_name", "spectrum_length", "patch_size", "embed_dim", "depth", "heads")
    missing_config = [key for key in required if key not in config]
    if missing_config:
        raise KeyError(f"Missing model configuration keys: {missing_config}")

    build_config = dict(config)
    # The shared repository renamed Fast2PeakRPE to F2P without changing its
    # relative-attention architecture or state-dict keys. Preserve historical
    # checkpoint metadata and adapt the name locally, with strict weight loading.
    if build_config["model_name"] == "Fast2PeakRPE" and "Fast2PeakRPE" not in model_registry:
        build_config["model_name"] = "F2P"
    model = build_model(**build_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    model.eval()
    return model, config, checkpoint
