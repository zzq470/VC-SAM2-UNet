#!/usr/bin/env python3
"""Paired D10 v3 Golden baseline/P1 validation-only training entrypoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT
GOLDEN = ROOT / "train"
C_GOLDEN = ROOT / "models/full"
P1_MODELS = ROOT / "models/full"
DATA_ROOT: Path | None = None
SPLIT_ROOT: Path | None = None
PRETRAINED: Path | None = None
VIEW_TO_ID = {"4CHV": 0, "LVOT": 1, "RVOT": 2, "3VTV": 3}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_processing"))
sys.path.insert(0, str(ROOT / "models/baseline"))
sys.path.insert(0, str(P1_MODELS))
sys.path.insert(0, str(GOLDEN))
import protocol_utils as golden
from dataset_d10_multilabel import D10MultiLabelDataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def call_model(model, batch, device):
    image = batch["image"].to(device, non_blocking=True)
    if getattr(model, "stage_a_needs_view", False):
        view_id = torch.tensor([VIEW_TO_ID[value] for value in batch["view"]], device=device)
        return model(image, view_id)
    return model(image)


@torch.no_grad()
def validate(model, loader, device, labels, views):
    model.eval()
    totals = {name: np.zeros(14, dtype=np.int64) for name in ("tp", "fp", "fn", "support", "pixels")}
    view_counts = {view: {name: np.zeros(14, dtype=np.int64) for name in ("tp", "fp", "fn")} for view in views}
    for batch in loader:
        logits, _ = call_model(model, batch, device)
        target = batch["target"].to(device).bool()
        valid = batch["valid_channels"].to(device).bool()
        pred = torch.sigmoid(logits) >= 0.5
        for index, view in enumerate(batch["view"]):
            for channel, label in enumerate(labels):
                if not valid[index, channel]:
                    continue
                p, g = pred[index, channel], target[index, channel]
                tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
                totals["tp"][channel] += tp; totals["fp"][channel] += fp; totals["fn"][channel] += fn
                totals["support"][channel] += int(g.any()); totals["pixels"][channel] += int(g.sum())
                if label in views[view]:
                    view_counts[view]["tp"][channel] += tp
                    view_counts[view]["fp"][channel] += fp
                    view_counts[view]["fn"][channel] += fn
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    per_view = {}
    for view, counts in view_counts.items():
        values = []
        for channel, label in enumerate(labels):
            denom = 2 * counts["tp"][channel] + counts["fp"][channel] + counts["fn"][channel]
            if label in views[view] and denom > 0:
                values.append((2 * counts["tp"][channel] + 1e-6) / (denom + 1e-6))
        per_view[view] = float(np.mean(values))
    return {
        "macro_dice": float(dice.mean()), "macro_iou": float(iou.mean()),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "dice": dice, "iou": iou, "precision": precision, "recall": recall,
        "support": totals["support"], "pixels": totals["pixels"], "per_view": per_view,
    }


def p1_diagnostics(model, batch, device):
    model.eval()
    image = batch["image"].to(device)
    view_id = torch.tensor([VIEW_TO_ID[value] for value in batch["view"]], device=device)
    with torch.no_grad():
        base = model.base
        enc = base.encoder(image)
        rfb = tuple(layer(value) for layer, value in zip((base.rfb1, base.rfb2, base.rfb3, base.rfb4), enc))
        conditioned = base.view_conditioner(rfb, view_id)
        film_ratio = float(np.mean([((b - a).norm() / a.norm().clamp_min(1e-8)).item() for a, b in zip(rfb, conditioned)]))
        full_logits, _ = model(image, view_id)
        branch = base.selective_dpbifpn
        residual_ratios = dict(branch.last_residual_ratios)
        saved = {key: value.detach().clone() for key, value in branch.residual.state_dict().items()}
        for projection in branch.residual.values():
            projection.weight.zero_(); projection.bias.zero_()
        disabled_logits, _ = model(image, view_id)
        branch.residual.load_state_dict(saved, strict=True)
        changed = int(((full_logits >= 0) != (disabled_logits >= 0)).sum())
        contribution = float((full_logits - disabled_logits).norm() / disabled_logits.norm().clamp_min(1e-8))
        normalized = branch.normalized_weights(None)
        weights = {node: [float(x) for x in value.detach().cpu().flatten()] for node, value in normalized.items()}
    return {
        "film_modulation_norm_ratio": film_ratio,
        "residual_contribution": contribution,
        "changed_pixels": changed,
        "residual_ratio_by_scale": residual_ratios,
        "normalized_fusion_weights": weights,
    }


def build_model(method: str, seed: int, device):
    golden.seed_all(seed)
    if method == "baseline":
        return golden.D10SAM2UNet(str(PRETRAINED)).to(device), {"shared_identity_keys": "all baseline keys"}
    control = golden.D10SAM2UNet(str(PRETRAINED)).to(device)
    shared = {key: value.detach().clone() for key, value in control.state_dict().items()}
    del control
    torch.cuda.empty_cache()
    # P1's model file expects the C/Metadata-FiLM SAM2UNet module, while the
    # training protocol helpers above intentionally come from Golden A.
    sys.modules.pop("SAM2UNet", None)
    sys.modules.pop("metadata_film_global_dpbifpn", None)
    sys.path.insert(0, str(C_GOLDEN))
    try:
        from metadata_film_global_dpbifpn import D10MetadataFiLMGlobalDPBiFPN
        model = D10MetadataFiLMGlobalDPBiFPN(str(PRETRAINED)).to(device)
    finally:
        sys.path.remove(str(C_GOLDEN))
    model.stage_a_needs_view = True
    own = model.state_dict()
    copied = []
    for key, value in shared.items():
        if key in own and own[key].shape == value.shape:
            own[key] = value
            copied.append(key)
    model.load_state_dict(own, strict=True)
    return model, {"shared_identity_key_count": len(copied), "new_p1_keys": len(own) - len(copied)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("baseline", "p1_gt_view"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--label-map", type=Path)
    parser.add_argument("--view-rules", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    global DATA_ROOT, SPLIT_ROOT, PRETRAINED
    DATA_ROOT = args.data_root.resolve()
    SPLIT_ROOT = args.split_root.resolve()
    PRETRAINED = args.pretrained.resolve()
    if PRETRAINED.name != "sam2_hiera_large.pt":
        raise RuntimeError("Initialization must use sam2_hiera_large.pt")
    label_map_path = (args.label_map or DATA_ROOT / "canonical/metadata/label_map.json").resolve()
    view_rules_path = (args.view_rules or DATA_ROOT / "canonical/metadata/view_label_rules.json").resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["label_name"] for row in sorted(json.loads(label_map_path.read_text()), key=lambda row: row["channel_index"])]
    views = {key: set(value) for key, value in json.loads(view_rules_path.read_text()).items()}
    golden.LABELS, golden.VIEWS = labels, views
    train = D10MultiLabelDataset(DATA_ROOT, SPLIT_ROOT / "train.csv", label_map_path, view_rules_path, 352, training=True)
    val = D10MultiLabelDataset(DATA_ROOT, SPLIT_ROOT / "val.csv", label_map_path, view_rules_path, 352, training=False)
    if (len(train), len(val)) != (625, 89):
        raise RuntimeError(f"Unexpected split sizes: train={len(train)} val={len(val)}")
    device = torch.device("cuda")
    model, identity = build_model(args.method, args.seed, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)
    val_loader = DataLoader(val, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    fixed_batch = next(iter(val_loader))
    config = {
        "method": args.method, "seed": args.seed, "epochs": args.epochs, "batch_size": 2,
        "input_size": 352, "optimizer": "AdamW", "lr": 5e-4, "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR", "eta_min": 5e-6,
        "loss": "partial-label masked multilabel BCE+Dice; final+side1+side2",
        "activation": "14 independent sigmoid channels", "threshold": 0.5,
        "augmentation": "Golden deterministic horizontal flip plan", "early_stopping": False,
        "dataset_root": "user-provided", "split_root": "user-provided",
        "train_split_sha256": sha256(SPLIT_ROOT / "train.csv"), "val_split_sha256": sha256(SPLIT_ROOT / "val.csv"),
        "test_split_sha256": "not-read-during-training", "checkpoint": PRETRAINED.name,
        "checkpoint_sha256": sha256(PRETRAINED), "loaded_best_checkpoint": False,
        "loaded_last_checkpoint": False, "test_images_read": 0, **identity,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    golden.seed_all(args.seed)
    first = next(iter(DataLoader(train, batch_size=2, shuffle=False, num_workers=0)))
    model.train(); logits, aux = call_model(model, first, device)
    loss, _ = golden.masked_losses(logits, aux, first["target"].to(device), first["valid_channels"].to(device))
    optimizer.zero_grad(set_to_none=True); loss.backward()
    finite = bool(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    if not finite or logits.shape[1] != 14:
        raise RuntimeError("Segmentation forward/backward preflight failed")
    optimizer.zero_grad(set_to_none=True)
    if args.preflight_only:
        print(json.dumps({"status": "PASS", "method": args.method, "loss": float(loss), "test_images_read": 0}))
        return
    model, identity = build_model(args.method, args.seed, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)
    logs, all_structure, all_view, sequence, activation = [], [], [], [], []
    best_dice, best_epoch = -1.0, -1
    for epoch in range(1, args.epochs + 1):
        started = time.time(); model.train(); losses = []
        order, flip_plan = golden.shared_epoch_plan(train, args.seed, epoch)
        train.set_flip_plan(flip_plan)
        loader = DataLoader(train, batch_size=2, sampler=golden.FixedOrderSampler(order), num_workers=0, pin_memory=True)
        for batch_index, batch in enumerate(loader):
            sequence.append({"epoch": epoch, "batch_index": batch_index,
                             "flip_flags": ";".join(str(int(flip_plan[order[batch_index * 2 + j]])) for j in range(len(batch["sample_id"])))})
            logits, aux = call_model(model, batch, device)
            loss, parts = golden.masked_losses(logits, aux, batch["target"].to(device), batch["valid_channels"].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss epoch={epoch} batch={batch_index}")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError(f"non-finite gradient epoch={epoch} batch={batch_index}")
            optimizer.step(); losses.append(float(loss.detach()))
        metrics = validate(model, val_loader, device, labels, views)
        scheduler.step()
        row = {"epoch": epoch, "train_total_loss": float(np.mean(losses)), "val_dice": metrics["macro_dice"],
               "val_iou": metrics["macro_iou"], "val_precision": metrics["macro_precision"],
               "val_recall": metrics["macro_recall"], "best_val_dice": max(best_dice, metrics["macro_dice"]),
               "lr": optimizer.param_groups[0]["lr"], "has_nan": False, "skipped_batches": 0,
               "elapsed_seconds": time.time() - started, **{f"val_{view}_dice": value for view, value in metrics["per_view"].items()}}
        logs.append(row)
        for c, label in enumerate(labels):
            all_structure.append({"epoch": epoch, "structure": label, "dice": metrics["dice"][c], "iou": metrics["iou"][c],
                                  "precision": metrics["precision"][c], "recall": metrics["recall"][c],
                                  "support": metrics["support"][c], "pixels": metrics["pixels"][c]})
        for view, value in metrics["per_view"].items():
            all_view.append({"epoch": epoch, "view": view, "dice": value})
        if args.method == "p1_gt_view" and (epoch == 1 or epoch % 10 == 0):
            activation.append({"epoch": epoch, **p1_diagnostics(model, fixed_batch, device)})
        if metrics["macro_dice"] > best_dice:
            best_dice, best_epoch = metrics["macro_dice"], epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "best_val_dice": best_dice, "config": config}, args.output_dir / "best_checkpoint.pt")
        write_csv(args.output_dir / "train_log.csv", logs)
        print(json.dumps({"method": args.method, "seed": args.seed, "epoch": epoch, "val_dice": metrics["macro_dice"], "best": best_dice, "test_images_read": 0}), flush=True)
    torch.save({"model_state": model.state_dict(), "epoch": args.epochs, "config": config}, args.output_dir / "last_checkpoint.pt")
    best_struct = [row for row in all_structure if row["epoch"] == best_epoch]
    best_views = [row for row in all_view if row["epoch"] == best_epoch]
    write_csv(args.output_dir / "validation_log.csv", logs)
    write_csv(args.output_dir / "validation_per_structure.csv", best_struct)
    write_csv(args.output_dir / "validation_per_view.csv", best_views)
    write_csv(args.output_dir / "training_sequence.csv", sequence)
    if activation:
        (args.output_dir / "p1_activation_every10.json").write_text(json.dumps(activation, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "training_summary.md").write_text(
        f"# {args.method} seed {args.seed}\n\nBest epoch: {best_epoch}\n\nBest validation Dice: {best_dice:.9f}\n\n"
        "Initialization: sam2_hiera_large.pt only. Test images read: 0.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
