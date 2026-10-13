"""D10 full-image multi-label dataset with channel-level partial supervision."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class D10MultiLabelDataset(Dataset):
    def __init__(self, data_root, split_csv, label_map, view_rules, img_size=352, training=False):
        self.root = Path(data_root)
        split_rows = list(csv.DictReader(Path(split_csv).open(encoding="utf-8-sig")))
        metadata_rows = {r["sample_id"]: r for r in csv.DictReader((self.root / "canonical" / "metadata" / "all_1000_samples.csv").open(encoding="utf-8-sig"))}
        # Split CSVs deliberately remain portable and compact. Merge the
        # authoritative per-sample annotation fields required for masking loss.
        self.rows = [{**metadata_rows[r["sample_id"]], **r} for r in split_rows]
        self.label_map = json.loads(Path(label_map).read_text(encoding="utf-8"))
        self.view_rules = {k: set(v) for k, v in json.loads(Path(view_rules).read_text(encoding="utf-8")).items()}
        self.img_size = int(img_size)
        self.training = bool(training)
        self.flip_plan = None
        if sorted(x["channel_index"] for x in self.label_map) != list(range(14)):
            raise ValueError("D10 label_map must define channel indices 0..13")

    def __len__(self): return len(self.rows)

    def set_flip_plan(self, flip_plan):
        self.flip_plan = flip_plan

    @staticmethod
    def _labels(value): return set(filter(None, value.split(";")))

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.root / row["image_relative_path"]).convert("RGB")
        original_size = (image.height, image.width)
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        flip = self.training and (bool(self.flip_plan[index]) if self.flip_plan is not None else random.random() < 0.5)
        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)
        image_t = (image_t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        masks = torch.zeros((14, self.img_size, self.img_size), dtype=torch.float32)
        mask_dir = self.root / row["masks_relative_directory"]
        for item in self.label_map:
            pattern = f"{int(item['semantic_id']):02d}_*.png"
            found = list(mask_dir.glob(pattern))
            if found:
                if len(found) != 1: raise ValueError(f"ambiguous mask for {row['sample_id']} channel {item['channel_index']}")
                mask = Image.open(found[0]).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
                if flip:
                    mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                masks[item["channel_index"]] = torch.from_numpy((np.asarray(mask) > 0).astype(np.float32))

        present, missing, unexpected = self._labels(row["present_labels"]), self._labels(row["missing_expected_labels"]), self._labels(row["unexpected_labels"])
        complete = row["annotation_complete"] == "True"
        valid = torch.ones(14, dtype=torch.float32)
        if not complete:
            for item in self.label_map:
                label, channel = item["label_name"], item["channel_index"]
                if label in missing or label in unexpected: valid[channel] = 0.0
        if not torch.all((masks == 0) | (masks == 1)): raise ValueError("D10 masks must be binary")
        return {"image": image_t, "target": masks, "valid_channels": valid, "sample_id": row["sample_id"], "view": row["view"], "annotation_complete": complete, "present_labels": row["present_labels"], "missing_expected_labels": row["missing_expected_labels"], "unexpected_labels": row["unexpected_labels"], "original_size": original_size}
