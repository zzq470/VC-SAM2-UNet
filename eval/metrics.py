"""Partial-label-safe multilabel segmentation metrics used by the paper."""
from __future__ import annotations

import numpy as np
import torch


def accumulate_counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_channels: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError("logits and targets must have identical BxCxHxW shapes")
    if valid_channels.shape != logits.shape[:2]:
        raise ValueError("valid_channels must have shape BxC")
    prediction = torch.sigmoid(logits) >= threshold
    truth = target.bool()
    valid = valid_channels.bool()[:, :, None, None]
    tp = (prediction & truth & valid).sum((0, 2, 3)).cpu().numpy().astype(np.int64)
    fp = (prediction & ~truth & valid).sum((0, 2, 3)).cpu().numpy().astype(np.int64)
    fn = (~prediction & truth & valid).sum((0, 2, 3)).cpu().numpy().astype(np.int64)
    return tp, fp, fn


def metrics_from_counts(tp, fp, fn, epsilon: float = 1e-6) -> dict[str, np.ndarray | float]:
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    return {
        "dice_per_class": dice,
        "iou_per_class": iou,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "macro_dice": float(dice.mean()),
        "macro_iou": float(iou.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
    }
