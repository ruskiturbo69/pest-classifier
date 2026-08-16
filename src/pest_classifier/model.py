from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    """Create an EfficientNet-V2-S classifier for ``num_classes`` labels."""
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = efficientnet_v2_s(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _state_dict_from_checkpoint(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError("Checkpoint must contain a model state dict.")

    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model_state_dict is not a mapping.")

    # Support checkpoints saved from DataParallel/DDP.
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_model(
    checkpoint: str | Path,
    num_classes: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load both the current checkpoint format and the original raw state dict."""
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    payload = torch.load(path, map_location=device)
    state_dict = _state_dict_from_checkpoint(payload)
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return model, metadata
