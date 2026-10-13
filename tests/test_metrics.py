from __future__ import annotations

import unittest

import torch

from eval.metrics import accumulate_counts, metrics_from_counts


class PartialLabelMetricTests(unittest.TestCase):
    def test_invalid_channel_is_excluded(self):
        logits = torch.tensor([[[[8.0]], [[8.0]]]])
        target = torch.tensor([[[[1.0]], [[0.0]]]])
        valid = torch.tensor([[1.0, 0.0]])
        tp, fp, fn = accumulate_counts(logits, target, valid)
        self.assertEqual(tp.tolist(), [1, 0])
        self.assertEqual(fp.tolist(), [0, 0])
        self.assertEqual(fn.tolist(), [0, 0])
        metrics = metrics_from_counts(tp, fp, fn)
        self.assertAlmostEqual(metrics["macro_dice"], 1.0)
        self.assertAlmostEqual(metrics["macro_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
