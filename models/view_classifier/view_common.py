#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


DATA_ROOT: Path | None = None
SPLIT_ROOT: Path | None = None
PROJECT_ROOT = Path(__file__).resolve().parents[2] / "results" / "view_classifier"
VIEW_TO_ID = {"4CHV": 0, "LVOT": 1, "RVOT": 2, "3VTV": 3}
ID_TO_VIEW = {v: k for k, v in VIEW_TO_ID.items()}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224


def configure_paths(data_root: Path, split_root: Path, project_root: Path | None = None) -> None:
    global DATA_ROOT, SPLIT_ROOT, PROJECT_ROOT
    DATA_ROOT = Path(data_root).resolve()
    SPLIT_ROOT = Path(split_root).resolve()
    if project_root is not None:
        PROJECT_ROOT = Path(project_root).resolve()


def read_split(name: str) -> list[dict[str, str]]:
    if SPLIT_ROOT is None:
        raise RuntimeError("Configure --split-root before reading a split")
    if name not in {"train", "val", "test"}:
        raise ValueError(name)
    with (SPLIT_ROOT / f"{name}.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stable_seed(seed: int, epoch: int, sample_id: str) -> int:
    payload = f"{seed}|{epoch}|{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fixed_mask(image: Image.Image) -> Image.Image:
    """D10 v3 is already grayscale/top-watermark processed; preserve all pixels."""
    return image.convert("RGB")


def resize_pad(image: Image.Image, size: int = INPUT_SIZE) -> tuple[Image.Image, dict[str, int]]:
    width, height = image.size
    scale = min(size / width, size / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = TF.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR, antialias=True)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    right = size - new_w - left
    bottom = size - new_h - top
    padded = TF.pad(resized, [left, top, right, bottom], fill=0)
    return padded, {"left": left, "top": top, "right": right, "bottom": bottom, "new_w": new_w, "new_h": new_h}


def sample_augmentation(seed: int, epoch: int, sample_id: str, width: int, height: int) -> dict[str, Any]:
    rng = random.Random(stable_seed(seed, epoch, sample_id))
    params = {
        "rotation_applied": rng.random() < 0.5,
        "translation_applied": rng.random() < 0.3,
        "scale_applied": rng.random() < 0.3,
        "brightness_applied": rng.random() < 0.3,
        "contrast_applied": rng.random() < 0.3,
        "gamma_applied": rng.random() < 0.2,
        "noise_applied": rng.random() < 0.15,
        "blur_applied": rng.random() < 0.1,
    }
    params.update({
        "angle": rng.uniform(-10.0, 10.0) if params["rotation_applied"] else 0.0,
        "translate_x": round(rng.uniform(-0.05, 0.05) * width) if params["translation_applied"] else 0,
        "translate_y": round(rng.uniform(-0.05, 0.05) * height) if params["translation_applied"] else 0,
        "scale": rng.uniform(0.95, 1.05) if params["scale_applied"] else 1.0,
        "brightness": rng.uniform(0.9, 1.1) if params["brightness_applied"] else 1.0,
        "contrast": rng.uniform(0.9, 1.1) if params["contrast_applied"] else 1.0,
        "gamma": rng.uniform(0.9, 1.1) if params["gamma_applied"] else 1.0,
        "noise_sigma": rng.uniform(0.0, 0.02) if params["noise_applied"] else 0.0,
        "blur_sigma": rng.uniform(0.1, 0.8) if params["blur_applied"] else 0.0,
        "horizontal_flip": False, "vertical_flip": False, "mixup": False,
        "cutmix": False, "random_erasing": False, "randaugment": False,
    })
    return params


def apply_augmentation(image: Image.Image, params: dict[str, Any]) -> Image.Image:
    image = TF.affine(
        image, angle=float(params["angle"]),
        translate=[int(params["translate_x"]), int(params["translate_y"])],
        scale=float(params["scale"]), shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR, fill=0,
    )
    if params["brightness_applied"]:
        image = ImageEnhance.Brightness(image).enhance(float(params["brightness"]))
    if params["contrast_applied"]:
        image = ImageEnhance.Contrast(image).enhance(float(params["contrast"]))
    if params["gamma_applied"]:
        image = TF.adjust_gamma(image, float(params["gamma"]))
    if params["blur_applied"]:
        image = image.filter(ImageFilter.GaussianBlur(float(params["blur_sigma"])))
    return image


def image_to_tensor(
    image: Image.Image,
    method: str,
    training: bool,
    seed: int,
    epoch: int,
    sample_id: str,
    grayscale_mean: float,
    grayscale_std: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    image = fixed_mask(image)
    params = sample_augmentation(seed, epoch, sample_id, *image.size) if training else {
        "angle": 0.0, "translate_x": 0, "translate_y": 0, "scale": 1.0,
        "brightness": 1.0, "contrast": 1.0, "gamma": 1.0, "noise_sigma": 0.0, "blur_sigma": 0.0,
        "rotation_applied": False, "translation_applied": False, "scale_applied": False,
        "brightness_applied": False, "contrast_applied": False, "gamma_applied": False,
        "noise_applied": False, "blur_applied": False, "horizontal_flip": False,
        "vertical_flip": False, "mixup": False, "cutmix": False,
        "random_erasing": False, "randaugment": False,
    }
    if training:
        image = apply_augmentation(image, params)
    image, pad = resize_pad(image)
    tensor = TF.to_tensor(image)
    if training and float(params["noise_sigma"]) > 0:
        generator = torch.Generator().manual_seed(stable_seed(seed, epoch, sample_id) + 17)
        noise = torch.randn(tensor.shape, generator=generator) * float(params["noise_sigma"])
        tensor = (tensor + noise).clamp(0, 1)
    if method == "V2_sononet64":
        tensor = TF.rgb_to_grayscale(tensor, num_output_channels=1)
        tensor = TF.normalize(tensor, [grayscale_mean], [grayscale_std])
    else:
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    params["pad"] = pad
    return tensor, params


class ViewDataset(Dataset):
    def __init__(self, split: str, method: str, training: bool, seed: int, grayscale_mean: float, grayscale_std: float):
        if split == "test":
            raise RuntimeError("Stage 2 forbids constructing a test image dataset")
        self.rows = read_split(split)
        self.split = split
        self.method = method
        self.training = training
        self.seed = seed
        self.epoch = 0
        self.grayscale_mean = grayscale_mean
        self.grayscale_std = grayscale_std

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if DATA_ROOT is None:
            raise RuntimeError("Configure --data-root before loading images")
        path = DATA_ROOT / row["image_relative_path"]
        with Image.open(path) as im:
            image, params = image_to_tensor(
                im.convert("RGB"), self.method, self.training, self.seed, self.epoch,
                row["sample_id"], self.grayscale_mean, self.grayscale_std,
            )
        return {
            "image": image,
            "target": torch.tensor(VIEW_TO_ID[row["view"]], dtype=torch.long),
            "sample_id": row["sample_id"],
            "audit_json": json.dumps(params, sort_keys=True),
        }


def compute_train_grayscale_stats() -> tuple[float, float, int]:
    total = 0
    s1 = 0.0
    s2 = 0.0
    for row in read_split("train"):
        with Image.open(DATA_ROOT / row["image_relative_path"]) as im:
            image, _ = resize_pad(fixed_mask(im.convert("RGB")))
            x = TF.rgb_to_grayscale(TF.to_tensor(image), num_output_channels=1).double()
        total += x.numel()
        s1 += float(x.sum())
        s2 += float((x * x).sum())
    mean = s1 / total
    variance = max(s2 / total - mean * mean, 1e-12)
    return mean, math.sqrt(variance), total


def confusion_matrix(targets: np.ndarray, preds: np.ndarray, classes: int = 4) -> np.ndarray:
    cm = np.zeros((classes, classes), dtype=np.int64)
    for y, p in zip(targets, preds):
        cm[int(y), int(p)] += 1
    return cm


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == targets
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def classification_metrics(logits: np.ndarray, targets: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]], np.ndarray]:
    logits_t = torch.tensor(logits, dtype=torch.float64)
    probs = torch.softmax(logits_t, dim=1).numpy()
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(targets, preds)
    per_class = []
    precisions, recalls, f1s, supports = [], [], [], []
    for c in range(4):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        support = int(cm[c, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class_id": c, "view": ID_TO_VIEW[c], "support": support, "precision": precision, "recall": recall, "f1": f1})
        precisions.append(precision); recalls.append(recall); f1s.append(f1); supports.append(support)
    n = len(targets)
    correct = int((preds == targets).sum())
    onehot = np.eye(4)[targets]
    nll = float(F.cross_entropy(torch.tensor(logits), torch.tensor(targets)).item())
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1)), axis=1)
    metrics = {
        "correct_count": correct, "error_count": n - correct, "accuracy": correct / n,
        "balanced_accuracy": float(np.mean(recalls)), "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)), "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.average(f1s, weights=supports)), "nll": nll, "brier": brier,
        "ece": expected_calibration_error(probs, targets), "mean_max_probability": float(probs.max(axis=1).mean()),
        "mean_predictive_entropy": float(entropy.mean()),
    }
    return metrics, per_class, cm


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    x = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(targets, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).item())


def lr_for_epoch(epoch: int, epochs: int, base_lr: float, eta_min: float = 1e-6) -> float:
    if epoch < 5:
        return base_lr * (0.1 + 0.9 * epoch / 4.0)
    progress = (epoch - 4) / max(1, epochs - 5)
    return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * progress))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
