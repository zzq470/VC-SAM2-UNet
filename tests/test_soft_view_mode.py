from __future__ import annotations

import unittest

import torch
from torch import nn

from models.full.view_modes import soft_conditioned_forward


class FakeEncoder(nn.Module):
    def forward(self, image):
        return (image, image, image, image)


class FakeUp(nn.Module):
    def forward(self, high, skip):
        return skip


class FakeConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 2)
        self.mlp = nn.Identity()
        self.projections = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])
        for layer in self.projections:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FakeEncoder()
        self.rfb1 = self.rfb2 = self.rfb3 = self.rfb4 = nn.Identity()
        self.view_conditioner = FakeConditioner()
        self.selective_dpbifpn = nn.Identity()
        self.up1 = self.up2 = self.up3 = FakeUp()
        self.head = nn.Conv2d(1, 14, 1)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = FakeBase()


class SoftViewModeTests(unittest.TestCase):
    def test_output_shape_and_identity_initialization(self):
        model = FakeModel().eval()
        image = torch.randn(2, 1, 8, 8)
        probabilities = torch.tensor([[0.25] * 4, [0.1, 0.2, 0.3, 0.4]])
        with torch.no_grad():
            result = soft_conditioned_forward(model, image, probabilities)
        self.assertEqual(tuple(result.shape), (2, 14, 32, 32))
        self.assertTrue(torch.isfinite(result).all())

    def test_rejects_wrong_probability_shape(self):
        with self.assertRaises(ValueError):
            soft_conditioned_forward(FakeModel(), torch.zeros(1, 1, 8, 8), torch.zeros(1, 3))


if __name__ == "__main__":
    unittest.main()
