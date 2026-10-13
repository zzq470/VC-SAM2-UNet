"""Builders for the four trained segmentation configurations."""
from __future__ import annotations

import torch.nn as nn

from train.protocol_utils import D10SAM2UNet
from models.full.metadata_film_global_dpbifpn import D10MetadataFiLMGlobalDPBiFPN
from models.component_ablation.sam2unet_component_ablation import build_variant


METHODS = {
    "A0": "SAM2-UNet baseline",
    "A1": "Metadata-FiLM only",
    "A2": "Selective DPBiFPN only",
    "A3": "Full VC-SAM2-UNet",
}


def build_model(method: str, pretrained: str) -> nn.Module:
    if method == "A0":
        model = D10SAM2UNet(pretrained)
        model.stage_a_needs_view = False
        return model
    if method == "A1":
        return build_variant(pretrained, "A1_film_only")
    if method == "A2":
        return build_variant(pretrained, "A2_dpbifpn_only")
    if method == "A3":
        model = D10MetadataFiLMGlobalDPBiFPN(pretrained)
        model.stage_a_needs_view = True
        return model
    raise ValueError(f"Unknown method: {method}; expected one of {tuple(METHODS)}")
