"""Inference helpers for discrete and probability-weighted view conditioning."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_conditioned_forward(model, image: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Run the A3 decoder with a probability-weighted view embedding.

    This reproduces the frozen Auto-Soft inference path without changing the
    model's state dict or adding trainable parameters.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError("probabilities must have shape Bx4")
    if probabilities.shape[0] != image.shape[0]:
        raise ValueError("image and view-probability batch sizes differ")

    base = model.base
    x1, x2, x3, x4 = base.encoder(image)
    features = tuple(
        layer(value)
        for layer, value in zip((base.rfb1, base.rfb2, base.rfb3, base.rfb4), (x1, x2, x3, x4))
    )
    conditioner = base.view_conditioner
    hidden = conditioner.mlp(probabilities @ conditioner.embedding.weight)
    conditioned = []
    for feature, projection in zip(features, conditioner.projections):
        gamma, beta = projection(hidden).chunk(2, dim=1)
        conditioned.append(feature * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None])
    x1, x2, x3, x4 = base.selective_dpbifpn(tuple(conditioned))
    decoded = base.up1(x4, x3)
    decoded = base.up2(decoded, x2)
    decoded = base.up3(decoded, x1)
    return F.interpolate(base.head(decoded), scale_factor=4, mode="bilinear")
