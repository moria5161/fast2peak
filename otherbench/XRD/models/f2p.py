"""Strict loader for the shared pretrained F2P architecture."""

from pathlib import Path
import torch
import yaml


def load_f2p(config: dict, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    from models import build_model, model_registry

    checkpoint_path = Path(config["f2p"]["checkpoint"])
    checkpoint_config = yaml.safe_load(
        (checkpoint_path.parent / "config.yaml").read_text(encoding="utf-8")
    )
    if int(checkpoint_config["spectrum_length"]) != int(config["crop"]["length"]):
        raise ValueError(
            f"F2P expects {checkpoint_config['spectrum_length']} points but crop has {config['crop']['length']}"
        )
    build_config = dict(checkpoint_config)
    # Preserve historical metadata while handling the shared architecture rename.
    if build_config["model_name"] == "Fast2PeakRPE" and "Fast2PeakRPE" not in model_registry:
        build_config["model_name"] = "F2P"
    model = build_model(**build_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint_config, checkpoint
