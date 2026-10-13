#!/usr/bin/env python3
"""Run the fixed 179-image D10 test protocol for one trained configuration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "data_processing", REPO_ROOT / "models/view_classifier"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_processing.dataset_d10_multilabel import D10MultiLabelDataset
from eval.metrics import accumulate_counts, metrics_from_counts
from models.model_factory import METHODS, build_model
from models.full.view_modes import soft_conditioned_forward
from models.view_classifier.view_models import build_model as build_view_model
import models.view_classifier.view_common as view_common


VIEW_TO_ID = {"4CHV": 0, "LVOT": 1, "RVOT": 2, "3VTV": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classifier_batch(rows, data_root: Path, device: torch.device):
    images = []
    for row in rows:
        with Image.open(data_root / row["image_relative_path"]) as source:
            image = view_common.fixed_mask(source.convert("RGB"))
            image, _ = view_common.resize_pad(image, 224)
        images.append(TF.normalize(TF.to_tensor(image), view_common.IMAGENET_MEAN, view_common.IMAGENET_STD))
    return torch.stack(images).to(device, non_blocking=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args) -> dict:
    if args.view_mode != "none" and args.method != "A3":
        raise ValueError("GT/Auto view modes are defined for the full A3 model only")
    if args.view_mode.startswith("auto-") and args.view_checkpoint is None:
        raise ValueError("Auto view evaluation requires the frozen ResNet-18 checkpoint")
    if args.checkpoint.name != "best_checkpoint.pt":
        raise ValueError("Only the validation-selected best_checkpoint.pt is accepted")
    if args.checkpoint.name == "last_checkpoint.pt":
        raise ValueError("last checkpoints are not supported")
    if not args.checkpoint.is_file() or not args.pretrained.is_file():
        raise FileNotFoundError("Checkpoint or SAM2 Hiera initialization file not found")

    with args.label_map.open(encoding="utf-8") as handle:
        labels = [row["label_name"] for row in sorted(json.load(handle), key=lambda x: int(x["channel_index"]))]
    with args.view_rules.open(encoding="utf-8") as handle:
        view_rules = {key: set(values) for key, values in json.load(handle).items()}
    if len(labels) != 14:
        raise ValueError("The fixed evaluation requires 14 output channels")

    view_common.configure_paths(args.data_root, args.split_root)
    dataset = D10MultiLabelDataset(
        args.data_root, args.test_split, args.label_map, args.view_rules,
        img_size=352, training=False,
    )
    if len(dataset) != 179:
        raise ValueError(f"Expected 179 fixed test images; found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

    device = torch.device(args.device)
    model = build_model(args.method, str(args.pretrained)).to(device)
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("model_state", raw.get("model"))
    if state is None:
        raise ValueError("Checkpoint has no supported model_state/model key")
    model.load_state_dict(state, strict=True)
    model.eval()

    classifier = None
    classifier_sha = None
    if args.view_mode.startswith("auto-"):
        if args.view_checkpoint.name != "best_checkpoint.pt":
            raise ValueError("Only the validation-best View classifier checkpoint is accepted")
        classifier_raw = torch.load(args.view_checkpoint, map_location="cpu", weights_only=False)
        classifier_state = classifier_raw.get("model", classifier_raw.get("model_state"))
        if classifier_state is None:
            raise ValueError("View checkpoint has no supported model/model_state key")
        classifier = build_view_model("V0_resnet18", pretrained=False).to(device)
        classifier.load_state_dict(classifier_state, strict=True)
        classifier.eval()
        classifier_sha = sha256(args.view_checkpoint)

    tp = np.zeros(14, dtype=np.int64)
    fp = np.zeros(14, dtype=np.int64)
    fn = np.zeros(14, dtype=np.int64)
    per_view = {view: [np.zeros(14, dtype=np.int64) for _ in range(3)] for view in VIEW_TO_ID}
    per_image_rows = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device)
            valid = batch["valid_channels"].to(device)
            gt_view_ids = torch.tensor([VIEW_TO_ID[value] for value in batch["view"]], device=device)
            probabilities = None
            if classifier is not None:
                current_rows = dataset.rows[seen:seen + len(batch["view"])]
                probabilities = torch.softmax(classifier(classifier_batch(current_rows, args.data_root, device)), dim=1)

            if args.view_mode == "gt":
                logits, _ = model(image, gt_view_ids)
            elif args.view_mode == "auto-hard":
                predicted_ids = probabilities.argmax(dim=1)
                logits, _ = model(image, predicted_ids)
            elif args.view_mode == "auto-soft":
                logits = soft_conditioned_forward(model, image, probabilities)
            else:
                logits, _ = model(image)

            batch_tp, batch_fp, batch_fn = accumulate_counts(logits, target, valid, 0.5)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            predictions = torch.sigmoid(logits) >= 0.5
            truths = target.bool()
            for index, view in enumerate(batch["view"]):
                view_mask = [label in view_rules[view] for label in labels]
                for channel, allowed in enumerate(view_mask):
                    if not allowed or not bool(valid[index, channel]):
                        continue
                    pred, truth = predictions[index, channel], truths[index, channel]
                    per_view[view][0][channel] += int((pred & truth).sum())
                    per_view[view][1][channel] += int((pred & ~truth).sum())
                    per_view[view][2][channel] += int((~pred & truth).sum())
                if args.write_per_image:
                    image_metrics = metrics_from_counts(*accumulate_counts(
                        logits[index:index + 1], target[index:index + 1], valid[index:index + 1], 0.5
                    ))
                    per_image_rows.append({
                        "test_ordinal": seen + index,
                        "view": view,
                        "macro_dice": image_metrics["macro_dice"],
                        "macro_iou": image_metrics["macro_iou"],
                    })
            seen += len(batch["view"])

    metrics = metrics_from_counts(tp, fp, fn)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "method": args.method,
        "view_mode": args.view_mode,
        "test_images_count": len(dataset),
        "test_images_read": seen,
        "test_macro_dice": metrics["macro_dice"],
        "test_macro_iou": metrics["macro_iou"],
        "mean_precision": metrics["macro_precision"],
        "mean_recall": metrics["macro_recall"],
        "input_size": 352,
        "activation": "14 independent sigmoid channels",
        "threshold": 0.5,
        "tta": False,
        "vap": False,
        "postprocessing": False,
        "calibration": False,
        "threshold_tuning": False,
        "training_performed": False,
        "optimizer_steps": 0,
        "loaded_last_checkpoint": False,
    }
    (args.output_dir / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    structure_rows = []
    for index, label in enumerate(labels):
        structure_rows.append({
            "structure": label,
            "dice": float(metrics["dice_per_class"][index]),
            "iou": float(metrics["iou_per_class"][index]),
            "precision": float(metrics["precision_per_class"][index]),
            "recall": float(metrics["recall_per_class"][index]),
            "tp": int(tp[index]), "fp": int(fp[index]), "fn": int(fn[index]),
        })
    write_csv(args.output_dir / "per_structure.csv", structure_rows)
    view_rows = []
    for view, counts in per_view.items():
        view_metrics = metrics_from_counts(*counts)
        eligible = [
            i for i, label in enumerate(labels)
            if label in view_rules[view] and int(sum(count[i] for count in counts)) > 0
        ]
        view_rows.append({
            "view": view,
            "macro_dice": float(np.mean(view_metrics["dice_per_class"][eligible])),
            "macro_iou": float(np.mean(view_metrics["iou_per_class"][eligible])),
        })
    write_csv(args.output_dir / "per_view.csv", view_rows)
    if args.write_per_image:
        write_csv(args.output_dir / "per_image_anonymized.csv", per_image_rows)

    audit = {
        "checkpoint_type": "best_checkpoint.pt",
        "checkpoint_sha256": sha256(args.checkpoint),
        "strict_load": True,
        "loaded_last_checkpoint": False,
        "view_classifier_checkpoint_sha256": classifier_sha,
        "test_images_read": seen,
        "shuffle": False,
        "drop_last": False,
        "model_eval": True,
    }
    (args.output_dir / "checkpoint_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--view-mode", choices=("none", "gt", "auto-hard", "auto-soft"), default="none")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--view-checkpoint", type=Path)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, default=REPO_ROOT / "configs/label_map.json")
    parser.add_argument("--view-rules", type=Path, default=REPO_ROOT / "configs/view_label_rules.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--write-per-image", action="store_true", help="write ordinal-only per-image metrics")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
