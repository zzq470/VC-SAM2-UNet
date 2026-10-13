from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProtocolContractTests(unittest.TestCase):
    def test_split_counts_and_independent_channels(self):
        config = json.loads((ROOT / "configs/paper_protocol.json").read_text(encoding="utf-8"))
        dataset = config["dataset"]
        self.assertEqual(dataset["images_total"], 893)
        self.assertEqual(dataset["split_counts"], {"train": 625, "validation": 89, "test": 179})
        self.assertEqual(sum(dataset["split_counts"].values()), dataset["images_total"])
        self.assertEqual(dataset["outputs"], 14)
        self.assertEqual(dataset["label_mode"], "independent overlapping binary masks")

    def test_label_and_view_orders(self):
        labels = json.loads((ROOT / "configs/label_map.json").read_text(encoding="utf-8"))
        labels = sorted(labels, key=lambda item: int(item["channel_index"]))
        self.assertEqual([int(item["channel_index"]) for item in labels], list(range(14)))
        views = json.loads((ROOT / "configs/view_label_rules.json").read_text(encoding="utf-8"))
        self.assertEqual(list(views), ["4CHV", "LVOT", "RVOT", "3VTV"])

    def test_published_tables_are_source_tables(self):
        with (ROOT / "results/paper_results.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual([int(row["validation_seed_count"]) for row in rows], [10] * 4)
        self.assertAlmostEqual(float(rows[0]["test_iou"]), 0.661626, places=6)
        self.assertAlmostEqual(float(rows[3]["test_iou"]), 0.702146, places=6)


if __name__ == "__main__":
    unittest.main()
