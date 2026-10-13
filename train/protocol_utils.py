"""Shared protocol helpers used by the paper training entrypoints."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler

from models.baseline.SAM2UNet import SAM2UNet

LABELS: list[str] = []
VIEWS: dict[str, set[str]] = {}


class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def shared_epoch_plan(dataset, seed: int, epoch: int):
    order = list(range(len(dataset)))
    random.Random(f"{seed}:shared-order:{epoch}").shuffle(order)
    flip_plan = {
        index: random.Random(
            f"{seed}:shared-flip:{epoch}:{dataset.rows[index]['sample_id']}"
        ).random() < 0.5
        for index in order
    }
    return order, flip_plan


def seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class D10SAM2UNet(nn.Module):
    def __init__(self, pretrained: str):
        super().__init__()
        self.base = SAM2UNet(pretrained)
        self.base.side1 = nn.Conv2d(64, 14, 1)
        self.base.side2 = nn.Conv2d(64, 14, 1)
        self.base.head = nn.Conv2d(64, 14, 1)

    def forward(self, image):
        main, side1, side2 = self.base(image)
        return main, [side1, side2]


def masked_losses(logits, auxiliary, target, valid_channels):
    valid4 = valid_channels[:, :, None, None]

    def component(output):
        if output.shape[-2:] != target.shape[-2:]:
            output = F.interpolate(
                output, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        raw = F.binary_cross_entropy_with_logits(output, target, reduction="none")
        bce = (raw * valid4).sum() / (
            valid4.sum() * target.shape[-1] * target.shape[-2]
        ).clamp_min(1)
        probs = torch.sigmoid(output)
        intersection = (probs * target).sum((2, 3))
        denominator = probs.sum((2, 3)) + target.sum((2, 3))
        eligible = (target.sum((2, 3)) > 0) & (valid_channels > 0)
        dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
        dice_loss = ((1 - dice) * eligible).sum() / eligible.sum().clamp_min(1)
        return bce + dice_loss, bce, dice_loss

    values = [component(logits), *(component(x) for x in auxiliary)]
    total = torch.stack([x[0] for x in values]).mean()
    bce = torch.stack([x[1] for x in values]).mean()
    dice_loss = torch.stack([x[2] for x in values]).mean()
    details = {
        "bce": float(bce.detach()),
        "dice_loss": float(dice_loss.detach()),
        "final_loss": float(values[0][0].detach()),
        "side1_loss": float(values[1][0].detach()),
        "side2_loss": float(values[2][0].detach()),
    }
    return total, details
