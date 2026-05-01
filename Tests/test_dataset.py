from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from Preprocessing.dataset import AlanKidneyDataset, AlanRecord, load_records, split_records


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

    def test_disk_cache_writes_preprocessed_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            volume = np.zeros((10, 12, 14), dtype=np.float32)
            volume[2:8, 3:9, 4:10] = 1.0
            path = tmp / "fake.npy"
            np.save(path, volume)
            record = AlanRecord(
                roi_id="CACHE001_L",
                subset="ZS-train",
                label=1,
                side="L",
                volume_path=path,
                voxel_count=int(volume.sum()),
                bbox_min=(2, 3, 4),
                bbox_max=(7, 8, 9),
            )
            cache_dir = tmp / "cache"
            dataset = AlanKidneyDataset(
                records=[record],
                target_shape=(8, 8, 8),
                use_bbox_crop=True,
                bbox_margin=1,
                cache_mode="disk",
                cache_dir=cache_dir,
            )

            first = dataset[0]["volume"]
            second = dataset[0]["volume"]

            self.assertEqual(tuple(first.shape), (1, 8, 8, 8))
            self.assertTrue(cache_dir.exists())
            self.assertEqual(len(list(cache_dir.glob("*.npy"))), 1)
            np.testing.assert_allclose(first.numpy(), second.numpy())

    def test_memory_cache_returns_clone(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            volume = np.zeros((10, 12, 14), dtype=np.float32)
            volume[2:8, 3:9, 4:10] = 1.0
            path = tmp / "fake.npy"
            np.save(path, volume)
            record = AlanRecord(
                roi_id="CACHE002_L",
                subset="ZS-dev",
                label=0,
                side="L",
                volume_path=path,
                voxel_count=int(volume.sum()),
                bbox_min=(2, 3, 4),
                bbox_max=(7, 8, 9),
            )
            dataset = AlanKidneyDataset(
                records=[record],
                target_shape=(8, 8, 8),
                use_bbox_crop=True,
                bbox_margin=1,
                cache_mode="memory",
            )
            expected = dataset._preprocess_uncached(record)

            first = dataset[0]["volume"]
            first.zero_()
            second = dataset[0]["volume"]

            np.testing.assert_allclose(second.numpy(), expected.numpy())

    def test_disk_cache_ignores_replace_race_when_target_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            dataset = AlanKidneyDataset(records=[], cache_mode="disk", cache_dir=cache_dir)
            key = "race"
            target = cache_dir / f"{key}.npy"

            def fake_replace(src: Path, dst: Path) -> None:
                np.save(dst, np.ones((1, 2, 2, 2), dtype=np.float32))
                raise PermissionError("target is busy")

            with mock.patch("Preprocessing.dataset.os.replace", side_effect=fake_replace):
                dataset._save_disk_cache(key, torch.zeros((1, 2, 2, 2), dtype=torch.float32))

            self.assertTrue(target.exists())
            self.assertEqual(list(cache_dir.glob(f"{key}.*.tmp.npy")), [])


if __name__ == "__main__":
    unittest.main()
