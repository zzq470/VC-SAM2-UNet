from __future__ import annotations

import unittest

import torch

from models.full.SAM2UNet import ViewConditioner


class MetadataFiLMIdentityTests(unittest.TestCase):
    def test_zero_initialized_residual_projections_are_identity(self):
        conditioner = ViewConditioner(num_views=4, embedding_dim=16, channels=64, levels=4)
        features = tuple(torch.randn(2, 64, size, size) for size in (44, 22, 11, 6))
        views = torch.tensor([0, 3])
        result = conditioner(features, views)
        for original, conditioned in zip(features, result):
            self.assertTrue(torch.equal(original, conditioned))
        for projection in conditioner.projections:
            self.assertEqual(float(projection.weight.abs().sum()), 0.0)
            self.assertEqual(float(projection.bias.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
