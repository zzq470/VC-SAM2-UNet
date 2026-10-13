"""Validate the private dataset contract without printing sample identifiers."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_SPLIT_COLUMNS = {"sample_id", "image_relative_path", "view"}
REQUIRED_METADATA_COLUMNS = {
    "sample_id", "image_relative_path", "masks_relative_directory",
    "view", "annotation_complete", "present_labels",
    "missing_expected_labels", "unexpected_labels",
}


def read_header(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return set(reader.fieldnames or ())


def validate(data_root: Path, split_root: Path, label_map: Path, view_rules: Path) -> dict:
    with label_map.open(encoding="utf-8") as handle:
        labels = json.load(handle)
    with view_rules.open(encoding="utf-8") as handle:
        views = json.load(handle)
    if sorted(int(row["channel_index"]) for row in labels) != list(range(14)):
        raise ValueError("label map must define exactly 14 channels indexed 0..13")
    expected_views = {"4CHV", "LVOT", "RVOT", "3VTV"}
    if set(views) != expected_views:
        raise ValueError("view map must define 4CHV, LVOT, RVOT, and 3VTV")
    metadata = data_root / "canonical/metadata/all_1000_samples.csv"
    if not REQUIRED_METADATA_COLUMNS.issubset(read_header(metadata)):
        raise ValueError("private metadata manifest does not match the documented schema")
    counts = {}
    identifiers: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = split_root / f"{split}.csv"
        if not REQUIRED_SPLIT_COLUMNS.issubset(read_header(path)):
            raise ValueError(f"{split}.csv does not match the split schema")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        identifiers[split] = {row["sample_id"] for row in rows}
        if len(identifiers[split]) != len(rows):
            raise ValueError(f"{split}.csv contains duplicate sample identifiers")
        counts[split] = len(rows)
    if counts != {"train": 625, "val": 89, "test": 179}:
        raise ValueError(f"expected the paper split sizes, observed {counts}")
    if (identifiers["train"] & identifiers["val"] or
            identifiers["train"] & identifiers["test"] or
            identifiers["val"] & identifiers["test"]):
        raise ValueError("split sample identifiers overlap; identifiers are not printed")
    return {"split_counts": counts, "split_ids_disjoint": True, "labels": 14, "views": sorted(views)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, default=Path("configs/label_map.json"))
    parser.add_argument("--view-rules", type=Path, default=Path("configs/view_label_rules.json"))
    args = parser.parse_args()
    print(json.dumps(validate(args.data_root, args.split_root, args.label_map, args.view_rules), indent=2))


if __name__ == "__main__":
    main()
