#!/usr/bin/env python3
"""Plot the published aggregate tables; does not load images or predictions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (ROOT / "results/paper_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    names = ["Baseline", "RMF only", "GSDB only", "Full"]
    dice = [float(row["test_dice"]) for row in rows]
    iou = [float(row["test_iou"]) for row in rows]
    x = range(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor="white")
    for axis, values, title, color in zip(
        axes, (dice, iou), ("Fixed-test macro Dice", "Fixed-test macro IoU"), ("#2f6b9a", "#3a8d78")
    ):
        axis.set_facecolor("white")
        bars = axis.bar(x, values, color=color, width=0.64)
        axis.set_xticks(list(x), names, rotation=18, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
