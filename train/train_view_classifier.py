#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models/view_classifier"))
import view_common as view_data
from view_common import (
    ID_TO_VIEW,
    PROJECT_ROOT,
    ViewDataset,
    classification_metrics,
    compute_train_grayscale_stats,
    fit_temperature,
    lr_for_epoch,
    read_split,
    set_global_seed,
    write_csv,
)
from view_models import build_model


METHODS = {
    "V0_resnet18": {"epochs": 50, "lr": 3e-4},
    "V1_efficientnet_b0": {"epochs": 50, "lr": 2e-4},
    "V2_sononet64": {"epochs": 75, "lr": 3e-4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([row["image"] for row in batch]),
        "target": torch.stack([row["target"] for row in batch]),
        "sample_id": [row["sample_id"] for row in batch],
        "audit_json": [row["audit_json"] for row in batch],
    }


def make_loader(dataset: ViewDataset, batch_size: int, workers: int, seed: int, epoch: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed + epoch * 1009)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, persistent_workers=False, generator=generator, collate_fn=collate,
    )


@torch.no_grad()
def infer(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    logits_all, targets_all, ids = [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        logits = model(image)
        if logits.shape != (image.shape[0], 4) or not torch.isfinite(logits).all():
            raise RuntimeError(f"Invalid validation logits: {tuple(logits.shape)}")
        logits_all.append(logits.float().cpu().numpy())
        targets_all.append(batch["target"].numpy())
        ids.extend(batch["sample_id"])
    return np.concatenate(logits_all), np.concatenate(targets_all), ids


def save_confusion(path: Path, cm: np.ndarray) -> None:
    rows = []
    for truth in range(4):
        row = {"true_class": ID_TO_VIEW[truth]}
        row.update({f"pred_{ID_TO_VIEW[pred]}": int(cm[truth, pred]) for pred in range(4)})
        rows.append(row)
    write_csv(path, rows)


def save_evaluation(run_dir: Path, prefix: str, logits: np.ndarray, targets: np.ndarray, ids: list[str], temperature: float = 1.0) -> dict:
    scaled = logits / temperature
    metrics, per_class, cm = classification_metrics(scaled, targets)
    probs = torch.softmax(torch.tensor(scaled), dim=1).numpy()
    preds = probs.argmax(axis=1)
    pred_rows = []
    for i, sample_id in enumerate(ids):
        row = {
            "sample_id": sample_id, "target_id": int(targets[i]), "target_view": ID_TO_VIEW[int(targets[i])],
            "pred_id": int(preds[i]), "pred_view": ID_TO_VIEW[int(preds[i])], "correct": int(preds[i] == targets[i]),
            "max_probability": float(probs[i].max()),
            "entropy": float(-(probs[i] * np.log(np.clip(probs[i], 1e-12, 1))).sum()),
        }
        for cls in range(4):
            row[f"logit_{ID_TO_VIEW[cls]}"] = float(scaled[i, cls])
            row[f"prob_{ID_TO_VIEW[cls]}"] = float(probs[i, cls])
        pred_rows.append(row)
    write_csv(run_dir / f"{prefix}_predictions.csv", pred_rows)
    write_csv(run_dir / f"{prefix}_per_class_metrics.csv", per_class)
    save_confusion(run_dir / f"{prefix}_confusion_matrix.csv", cm)
    with (run_dir / f"{prefix}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    args = parse_args()
    view_data.configure_paths(args.data_root, args.split_root, args.run_dir.parent)
    spec = METHODS[args.method]
    epochs = args.epochs or spec["epochs"]
    run_dir = args.run_dir or PROJECT_ROOT / "runs" / f"{args.method}_seed{args.seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    set_global_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    device = torch.device("cuda")

    gray_mean, gray_std, gray_pixels = compute_train_grayscale_stats()
    train_ds = ViewDataset("train", args.method, True, args.seed, gray_mean, gray_std)
    val_ds = ViewDataset("val", args.method, False, args.seed, gray_mean, gray_std)

    model = build_model(args.method, pretrained=args.pretrained).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # FP32 is frozen for all three architectures. AMP overflowed on a valid
    # preflight batch; silently skipping that optimizer step would violate the
    # no-skipped-batch fairness gate.
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {
        "method": args.method, "seed": args.seed, "epochs": epochs, "batch_size": args.batch_size,
        "optimizer": "AdamW", "base_lr": spec["lr"], "weight_decay": 1e-4,
        "loss": "CrossEntropyLoss", "label_smoothing": 0.05, "class_weights": None,
        "warmup_epochs": 5, "scheduler": "linear_warmup_then_cosine", "eta_min": 1e-6,
        "checkpoint_metric": "val_macro_f1", "input_size": 224, "pretrained": args.pretrained,
        "grayscale_train_mean": gray_mean, "grayscale_train_std": gray_std, "grayscale_pixels": gray_pixels,
        "train_count": len(train_ds), "val_count": len(val_ds), "test_split_read_during_training": False,
        "test_images_read": 0, "patient_level_independence_unverified": True,
        "dataset": "D10_v3_topwm_gray_625_89_179",
        "preprocessing": "full grayscale image; aspect-ratio resize/pad 224; no crop/OCR/mask",
        "forbidden_forward_inputs": ["view", "filepath", "filename", "sample_id", "mask", "structure_count"],
        "model_forward_inputs": ["image"], "parameter_count": parameter_count,
        "torch": torch.__version__, "python": sys.version, "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0), "amp": use_amp, "amp_reason": "FP32 frozen after strict AMP overflow preflight",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False, "matmul_tf32": False, "cudnn_tf32": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""), "preflight": args.preflight,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    shutil.copy2(run_dir / "config.json", run_dir / "run_config.json")

    log_rows, sequence_rows = [], []
    best_f1, best_epoch = -1.0, -1
    started = time.time()
    optimizer_steps = 0
    for epoch in range(epochs):
        train_ds.set_epoch(epoch)
        lr = lr_for_epoch(epoch, epochs, spec["lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        loader = make_loader(train_ds, args.batch_size, args.workers, args.seed, epoch, True)
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for batch_index, batch in enumerate(loader):
            if epoch < 5:
                for position, (sample_id, audit_json) in enumerate(zip(batch["sample_id"], batch["audit_json"])):
                    sequence_rows.append({
                        "method": args.method, "seed": args.seed, "epoch": epoch + 1,
                        "batch_index": batch_index, "position": position,
                        "sample_id": sample_id, "augmentation_json": audit_json,
                    })
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(image)
                loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss epoch={epoch + 1} batch={batch_index}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Non-finite gradient {name} epoch={epoch + 1} batch={batch_index}")
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            loss_sum += float(loss) * image.shape[0]
            correct += int((logits.argmax(1) == target).sum())
            seen += image.shape[0]

        val_loader = make_loader(val_ds, args.batch_size, args.workers, args.seed, epoch, False)
        val_logits, val_targets, val_ids = infer(model, val_loader, device)
        val_metrics, _, _ = classification_metrics(val_logits, val_targets)
        row = {
            "epoch": epoch + 1, "lr": lr, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
            **{f"val_{k}": v for k, v in val_metrics.items()}, "best_val_macro_f1": max(best_f1, val_metrics["macro_f1"]),
            "best_epoch": best_epoch, "elapsed_seconds": time.time() - started, "has_nan": False,
        }
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            row["best_val_macro_f1"] = best_f1
            row["best_epoch"] = best_epoch
            torch.save({
                "model": model.state_dict(), "method": args.method, "seed": args.seed, "epoch": best_epoch,
                "val_macro_f1": best_f1, "config": config,
            }, run_dir / "best_model.pth")
            shutil.copy2(run_dir / "best_model.pth", run_dir / "best_checkpoint.pt")
        log_rows.append(row)
        write_csv(run_dir / "train_log.csv", log_rows)
        print(json.dumps({"epoch": epoch + 1, "train_loss": row["train_loss"], "train_acc": row["train_accuracy"],
                          "val_acc": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
                          "best_f1": best_f1, "best_epoch": best_epoch}), flush=True)

    torch.save({"model": model.state_dict(), "method": args.method, "seed": args.seed, "epoch": epochs, "config": config}, run_dir / "last_model.pth")
    shutil.copy2(run_dir / "last_model.pth", run_dir / "last_checkpoint.pt")
    write_csv(run_dir / "training_sequence_first5epochs.csv", sequence_rows)

    checkpoint = torch.load(run_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    val_loader = make_loader(val_ds, args.batch_size, args.workers, args.seed, 0, False)
    torch.cuda.synchronize()
    inference_started = time.time()
    logits, targets, ids = infer(model, val_loader, device)
    torch.cuda.synchronize()
    validation_inference_seconds = time.time() - inference_started
    raw_metrics = save_evaluation(run_dir, "val", logits, targets, ids)
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    np.save(run_dir / "val_probabilities.npy", probabilities)
    shutil.copy2(run_dir / "val_confusion_matrix.csv", run_dir / "confusion_matrix.csv")
    shutil.copy2(run_dir / "val_per_class_metrics.csv", run_dir / "per_class_metrics.csv")
    temperature = fit_temperature(logits, targets)
    calibrated_metrics = save_evaluation(run_dir, "val_calibrated", logits, targets, ids, temperature)
    unchanged = bool(np.array_equal(logits.argmax(1), (logits / temperature).argmax(1)))
    calibration = {
        "temperature": temperature, "fit_split": "validation", "argmax_predictions_unchanged": unchanged,
        "raw_nll": raw_metrics["nll"], "calibrated_nll": calibrated_metrics["nll"],
        "raw_ece": raw_metrics["ece"], "calibrated_ece": calibrated_metrics["ece"], "test_images_read": 0,
    }
    with (run_dir / "calibration.json").open("w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    with (run_dir / "temperature.json").open("w", encoding="utf-8") as f:
        json.dump({"temperature": temperature, "fit_split": "validation", "argmax_predictions_unchanged": unchanged}, f, indent=2)
    write_csv(run_dir / "calibration_metrics.csv", [
        {"state": "raw", **raw_metrics},
        {"state": "temperature_scaled", **calibrated_metrics},
    ])
    final = {
        "method": args.method, "seed": args.seed, "best_epoch": best_epoch,
        **{f"val_{k}": v for k, v in raw_metrics.items()},
        "temperature": temperature, "calibrated_ece": calibrated_metrics["ece"],
        "calibrated_nll": calibrated_metrics["nll"], "test_images_read": 0,
        "optimizer_steps": optimizer_steps, "nonfinite_batches": 0, "skipped_batches": 0,
        "parameter_count": parameter_count, "validation_inference_seconds": validation_inference_seconds,
        "validation_images_per_second": len(val_ds) / validation_inference_seconds,
        "segmentation_masks_used_as_input": False, "structure_counts_used_as_input": False,
        "P1_model_modified": False, "status": "completed", "run_dir": str(run_dir),
    }
    with (run_dir / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    (run_dir / "training_summary.md").write_text(
        f"# {args.method} seed {args.seed}\n\n"
        f"- best epoch: {best_epoch}\n- validation accuracy: {raw_metrics['accuracy']:.6f}\n"
        f"- validation macro F1: {raw_metrics['macro_f1']:.6f}\n- validation errors: {raw_metrics['error_count']} / {len(val_ds)}\n"
        f"- optimizer steps: {optimizer_steps}\n- skipped batches: 0\n- test_images_read: 0\n"
        "- segmentation_masks_used_as_input: false\n- structure_counts_used_as_input: false\n- P1_model_modified: false\n",
        encoding="utf-8",
    )
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
