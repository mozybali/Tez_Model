from __future__ import annotations

import unittest
from pathlib import Path

from Preprocessing.dataset import AlanKidneyDataset, load_records, split_records


class AlanDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_records(
            info_csv=Path("ALAN/info.csv"),
            volumes_dir=Path("ALAN/alan"),
            metadata_csv=Path("ALAN/metadata.csv"),
            summary_json=Path("ALAN/summary.json"),
        )

    def test_record_count_matches_csv(self) -> None:
        self.assertEqual(len(self.records), 1584)

    def test_subset_counts_match_expected_split(self) -> None:
        splits = split_records(self.records)
        self.assertEqual(len(splits["train"]), 1188)
        self.assertEqual(len(splits["val"]), 98)
        self.assertEqual(len(splits["test"]), 298)

    def test_no_patient_level_leakage(self) -> None:
        """Both L and R for each patient must be in the same split."""
        patient_splits: dict[str, set[str]] = {}
        for record in self.records:
            patient_id = record.roi_id.rsplit("_", 1)[0]
            patient_splits.setdefault(patient_id, set()).add(record.subset)
        leaking = {pid: subs for pid, subs in patient_splits.items() if len(subs) > 1}
        self.assertEqual(len(leaking), 0, f"Patient-level leakage: {leaking}")

    def test_dataset_returns_expected_tensor_shape(self) -> None:
        dataset = AlanKidneyDataset(
            records=self.records[:2],
            target_shape=(64, 64, 64),
            use_bbox_crop=True,
            bbox_margin=8,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["volume"].shape), (1, 64, 64, 64))
        self.assertIn(float(sample["label"].item()), {0.0, 1.0})


if __name__ == "__main__":
    unittest.main()

