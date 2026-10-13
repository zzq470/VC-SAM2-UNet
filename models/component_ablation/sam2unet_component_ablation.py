"""Metadata-FiLM-only and Selective-DPBiFPN-only paper ablations."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_MODEL_DIR = REPO_ROOT / "models" / "full"
if str(FULL_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(FULL_MODEL_DIR))

from models.full.SAM2UNet import ViewConditioner
from models.full.selective_dpbifpn_common import SelectiveDualPathBiFPN
from models.baseline.SAM2UNet import SAM2UNet as BaselineSAM2UNet


class SAM2UNetComponentAblation(nn.Module):
    """SAM2-UNet with independently switchable RMF and GSDB components."""

    def __init__(
        self,
        checkpoint_path: str,
        use_metadata_film: bool,
        use_selective_dpbifpn: bool,
        view_embedding_dim: int = 16,
    ):
        super().__init__()
        self.use_metadata_film = bool(use_metadata_film)
        self.use_selective_dpbifpn = bool(use_selective_dpbifpn)
        self.stage_a_needs_view = self.use_metadata_film

        self.base = BaselineSAM2UNet(checkpoint_path)
        self.base.side1 = nn.Conv2d(64, 14, 1)
        self.base.side2 = nn.Conv2d(64, 14, 1)
        self.base.head = nn.Conv2d(64, 14, 1)

        if self.use_metadata_film:
            self.view_conditioner = ViewConditioner(
                num_views=4, embedding_dim=view_embedding_dim, channels=64, levels=4
            )
        if self.use_selective_dpbifpn:
            self.selective_dpbifpn = SelectiveDualPathBiFPN(
                channels=(64, 64, 64, 64), out_channels=64, view_conditioned=False
            )

    def forward(self, image: torch.Tensor, view_id: torch.Tensor | None = None):
        x1, x2, x3, x4 = self.base.encoder(image)
        features = tuple(
            layer(value)
            for layer, value in zip(
                (self.base.rfb1, self.base.rfb2, self.base.rfb3, self.base.rfb4),
                (x1, x2, x3, x4),
            )
        )
        if self.use_metadata_film:
            if view_id is None:
                raise ValueError("Metadata-FiLM requires a four-class view input")
            features = self.view_conditioner(features, view_id)
        elif view_id is not None:
            raise ValueError("The DPBiFPN-only ablation must not receive a view input")

        if self.use_selective_dpbifpn:
            features = self.selective_dpbifpn(features)

        x1, x2, x3, x4 = features
        decoded = self.base.up1(x4, x3)
        side1 = F.interpolate(self.base.side1(decoded), scale_factor=16, mode="bilinear")
        decoded = self.base.up2(decoded, x2)
        side2 = F.interpolate(self.base.side2(decoded), scale_factor=8, mode="bilinear")
        decoded = self.base.up3(decoded, x1)
        final = F.interpolate(self.base.head(decoded), scale_factor=4, mode="bilinear")
        return final, [side1, side2]


def build_variant(checkpoint_path: str, variant: str) -> SAM2UNetComponentAblation:
    settings = {
        "A1_film_only": (True, False),
        "A2_dpbifpn_only": (False, True),
    }
    if variant not in settings:
        raise ValueError(f"Unknown component-ablation variant: {variant}")
    return SAM2UNetComponentAblation(checkpoint_path, *settings[variant])
