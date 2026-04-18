from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Preprocessing.dataset import crop_to_bbox, pad_to_cube, resize_volume, apply_nan_strategy
from Utils.metrics import compute_binary_classification_metrics, select_model_score


class PreprocessingFunctionsTest(unittest.TestCase):
    def test_crop_to_bbox(self) -> None:
        volume = np.zeros((20, 30, 40), dtype=np.float32)
        volume[5:10, 10:20, 15:25] = 1.0
        cropped = crop_to_bbox(volume, bbox_min=(5, 10, 15), bbox_max=(9, 19, 24), margin=2)
        self.assertEqual(cropped.shape[0], 4 + 2 * 2 + 1)   # (bbox_max - bbox_min) + 2*margin + 1
        self.assertEqual(cropped.shape[1], 9 + 2 * 2 + 1)
        self.assertEqual(cropped.shape[2], 9 + 2 * 2 + 1)

    def test_pad_to_cube(self) -> None:
        volume = np.ones((10, 20, 15), dtype=np.float32)
        cubed = pad_to_cube(volume)
        self.assertEqual(cubed.shape[0], 20)
        self.assertEqual(cubed.shape[1], 20)
        self.assertEqual(cubed.shape[2], 20)

    def test_resize_volume(self) -> None:
        volume = np.random.rand(16, 16, 16).astype(np.float32)
        tensor = resize_volume(volume, target_shape=(32, 32, 32))
        self.assertEqual(tuple(tensor.shape), (1, 32, 32, 32))


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]
        m = compute_binary_classification_metrics(y_true, y_prob)
        self.assertAlmostEqual(m["roc_auc"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)

    def test_select_model_score_fallback(self) -> None:
        metrics = {"roc_auc": float("nan"), "balanced_accuracy": 0.75}
        score = select_model_score(metrics, "roc_auc")
        self.assertAlmostEqual(score, 0.75)


class SmokeTest(unittest.TestCase):
    """Minimal end-to-end test with synthetic data (no ALAN files needed)."""

    def test_training_smoke(self) -> None:
        from Preprocessing.dataset import AlanRecord, AlanKidneyDataset, infer_positive_class_weight
        from Model.resnet3d import build_resnet3d
        from Utils.metrics import compute_binary_classification_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            records = []
            for i in range(6):
                vol = np.zeros((16, 16, 16), dtype=np.float32)
                vol[4:12, 4:12, 4:12] = float(i % 2)
                path = tmp / f"fake_{i}.npy"
                np.save(path, vol)
                records.append(AlanRecord(
                    roi_id=f"FAKE{i:03d}_L",
                    subset="ZS-train",
                    label=i % 2,
                    side="L",
                    volume_path=path,
                    voxel_count=int(vol.sum()),
                    bbox_min=(4, 4, 4),
                    bbox_max=(11, 11, 11),
                ))

            dataset = AlanKidneyDataset(records=records, target_shape=(16, 16, 16), bbox_margin=2)
            sample = dataset[0]
            self.assertEqual(tuple(sample["volume"].shape), (1, 16, 16, 16))

            model = build_resnet3d(depth=18, base_channels=8, dropout=0.0)
            model.eval()

            pw = infer_positive_class_weight(records)
            self.assertGreater(pw, 0)

            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
            volumes = torch.stack([dataset[i]["volume"] for i in range(len(dataset))])
            labels = torch.tensor([dataset[i]["label"] for i in range(len(dataset))])

            with torch.no_grad():
                logits = model(volumes)
                loss = criterion(logits, labels)

            self.assertFalse(torch.isnan(loss))
            probs = torch.sigmoid(logits).tolist()
            m = compute_binary_classification_metrics(labels.tolist(), probs)
            self.assertIn("roc_auc", m)

    def test_tabular_feature_builder(self) -> None:
        from Model.engine import build_tabular_features, compute_tabular_feature_stats
        from Preprocessing.dataset import AlanRecord

        records = [
            AlanRecord(
                roi_id="FAKE001_L", subset="ZS-train", label=0, side="L",
                volume_path=Path("fake_l.npy"), voxel_count=100,
                bbox_min=(0, 0, 0), bbox_max=(1, 1, 1),
            ),
            AlanRecord(
                roi_id="FAKE002_R", subset="ZS-train", label=1, side="R",
                volume_path=Path("fake_r.npy"), voxel_count=400,
                bbox_min=(0, 0, 0), bbox_max=(1, 1, 1),
            ),
        ]
        stats = compute_tabular_feature_stats(records)
        features = build_tabular_features(
            {"voxel_count": torch.tensor([100.0, 400.0]), "side": ["L", "R"]},
            stats,
            torch.device("cpu"),
        )

        self.assertEqual(tuple(features.shape), (2, 2))
        self.assertAlmostEqual(float(features[0, 1]), 1.0)
        self.assertAlmostEqual(float(features[1, 1]), 0.0)


class NanStrategyTest(unittest.TestCase):
    """Tests for NaN handling in the preprocessing pipeline."""

    def test_no_nan_preserves_volume(self) -> None:
        vol = np.ones((8, 8, 8), dtype=np.float32)
        result = apply_nan_strategy(vol, "none")
        np.testing.assert_array_equal(vol, result)

    def test_fill_zero_removes_nans(self) -> None:
        vol = np.ones((8, 8, 8), dtype=np.float32)
        vol[0, 0, 0] = np.nan
        result = apply_nan_strategy(vol, "fill_zero")
        self.assertFalse(np.isnan(result).any())
        self.assertEqual(result[0, 0, 0], 0.0)

    def test_fill_mean_correct_value(self) -> None:
        vol = np.full((4, 4, 4), 2.0, dtype=np.float32)
        vol[0, 0, 0] = np.nan
        result = apply_nan_strategy(vol, "fill_mean")
        self.assertFalse(np.isnan(result).any())
        self.assertAlmostEqual(result[0, 0, 0], 2.0)

    def test_fill_median_correct_value(self) -> None:
        vol = np.full((4, 4, 4), 3.0, dtype=np.float32)
        vol[0, 0, 0] = np.nan
        result = apply_nan_strategy(vol, "fill_median")
        self.assertFalse(np.isnan(result).any())
        self.assertAlmostEqual(result[0, 0, 0], 3.0)

    def test_fill_constant_uses_given_value(self) -> None:
        vol = np.ones((4, 4, 4), dtype=np.float32)
        vol[0, 0, 0] = np.nan
        result = apply_nan_strategy(vol, "fill_constant", fill_value=7.5)
        self.assertAlmostEqual(result[0, 0, 0], 7.5)

    def test_all_nan_volume_fallback(self) -> None:
        vol = np.full((4, 4, 4), np.nan, dtype=np.float32)
        for strategy in ("fill_mean", "fill_median"):
            result = apply_nan_strategy(vol, strategy)
            self.assertFalse(np.isnan(result).any())
            np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_none_strategy_leaves_nans(self) -> None:
        vol = np.ones((4, 4, 4), dtype=np.float32)
        vol[0, 0, 0] = np.nan
        result = apply_nan_strategy(vol, "none")
        self.assertTrue(np.isnan(result[0, 0, 0]))

    def test_dataset_with_nan_fill_zero(self) -> None:
        from Preprocessing.dataset import AlanRecord, AlanKidneyDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            vol = np.ones((16, 16, 16), dtype=np.float32)
            vol[2, 2, 2] = np.nan
            path = tmp / "nan_vol.npy"
            np.save(path, vol)
            record = AlanRecord(
                roi_id="NAN_TEST_L", subset="ZS-train", label=0, side="L",
                volume_path=path, voxel_count=int(np.nansum(vol)),
                bbox_min=(0, 0, 0), bbox_max=(15, 15, 15),
                nan_count=1, nan_ratio=1 / 16**3, has_nan=True,
            )
            dataset = AlanKidneyDataset(
                records=[record], target_shape=(8, 8, 8), bbox_margin=0,
                nan_strategy="fill_zero",
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["volume"].shape), (1, 8, 8, 8))
            self.assertFalse(torch.isnan(sample["volume"]).any())

    def test_drop_record_filters_correctly(self) -> None:
        from Preprocessing.dataset import AlanRecord

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            records = []
            for i in range(4):
                vol = np.ones((8, 8, 8), dtype=np.float32)
                has_nan = i < 2
                if has_nan:
                    vol[0, 0, 0] = np.nan
                path = tmp / f"drop_{i}.npy"
                np.save(path, vol)
                records.append(AlanRecord(
                    roi_id=f"DROP{i:03d}_L", subset="ZS-train", label=i % 2,
                    side="L", volume_path=path, voxel_count=int(np.nansum(vol)),
                    bbox_min=(0, 0, 0), bbox_max=(7, 7, 7),
                    nan_count=1 if has_nan else 0,
                    nan_ratio=(1 / 8**3) if has_nan else 0.0,
                    has_nan=has_nan,
                ))
            clean = [r for r in records if not r.has_nan]
            self.assertEqual(len(clean), 2)


if __name__ == "__main__":
    unittest.main()
