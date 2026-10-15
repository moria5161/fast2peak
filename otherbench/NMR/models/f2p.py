"""Load the shared relative-attention F2P with historical-name compatibility."""

from pathlib import Path
import torch
import yaml


def build_model(**config):
    from models import build_model as shared_build, model_registry
    current = dict(config)
    if current["model_name"] == "Fast2PeakRPE" and "Fast2PeakRPE" not in model_registry:
        current["model_name"] = "F2P"
    return shared_build(**current)


def load_f2p(config: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
    # Keep these imports local.  Besides avoiding unnecessary model imports for
    # classical-only scoring, this lets other evaluators reuse the metric
    # functions in this module without claiming the generic ``models`` and
    # ``utils`` module names first.

    checkpoint_path = Path(config["f2p"]["checkpoint"])
    checkpoint_config = yaml.safe_load(
        (checkpoint_path.parent / "config.yaml").read_text(encoding="utf-8")
    )
    native_length = int(
        config["f2p"].get("native_length", checkpoint_config["spectrum_length"])
    )
    if int(checkpoint_config["spectrum_length"]) != native_length:
        raise ValueError(
            f"F2P expects {checkpoint_config['spectrum_length']} points, "
            f"configured native window has {native_length}"
        )
    model = build_model(**checkpoint_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    model._f2p_asinh_scale = (
        float(checkpoint_config.get("asinh_scale", 0.0))
        if bool(checkpoint_config.get("asinh", False))
        else 0.0
    )
    return model, checkpoint
