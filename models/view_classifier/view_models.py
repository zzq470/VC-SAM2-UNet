#!/usr/bin/env python3
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)


class ResNet18View(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 4):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


class EfficientNetB0View(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 4):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


def _sononet_block(in_channels: int, out_channels: int, count: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(count):
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
        in_channels = out_channels
    return nn.Sequential(*layers)


class SonoNet64View(nn.Module):
    """SonoNet-64, following the official VGG-like PyTorch configuration.

    The official repository applies softmax in forward; this implementation
    deliberately returns logits because training uses CrossEntropyLoss.
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            _sononet_block(1, 64, 2), nn.MaxPool2d(2, 2),
            _sononet_block(64, 128, 2), nn.MaxPool2d(2, 2),
            _sononet_block(128, 256, 3), nn.MaxPool2d(2, 2),
            _sononet_block(256, 512, 3), nn.MaxPool2d(2, 2),
            _sononet_block(512, 512, 3),
        )
        self.adaptation = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.features(image)
        logits_map = self.adaptation(feature)
        return logits_map.mean(dim=(2, 3))


def build_model(method: str, pretrained: bool = True) -> nn.Module:
    if method == "V0_resnet18":
        return ResNet18View(pretrained=pretrained)
    if method == "V1_efficientnet_b0":
        return EfficientNetB0View(pretrained=pretrained)
    if method == "V2_sononet64":
        return SonoNet64View()
    raise ValueError(f"Unknown method: {method}")

