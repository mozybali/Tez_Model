from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Preprocessing.dataset import crop_to_bbox, pad_to_cube, resize_volume, apply_nan_strategy
from Utils.metrics import (
    compute_binary_classification_metrics,
    compute_per_class_report,
    select_model_score,
)


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
        # Perfect predictions ⇒ MCC = 1, kappa = 1, no false negatives / positives.
        self.assertAlmostEqual(m["mcc"], 1.0)
        self.assertAlmostEqual(m["cohen_kappa"], 1.0)
        self.assertEqual(m["fp"], 0.0)
        self.assertEqual(m["fn"], 0.0)
        self.assertAlmostEqual(m["fpr"], 0.0)
        self.assertAlmostEqual(m["fnr"], 0.0)

    def test_select_model_score_fallback(self) -> None:
        metrics = {"roc_auc": float("nan"), "balanced_accuracy": 0.75}
        score = select_model_score(metrics, "roc_auc")
        self.assertAlmostEqual(score, 0.75)


class ExtendedBinaryMetricsTest(unittest.TestCase):
    """Verify the metrics added in the audit against hand-computed values.

    Confusion matrix used below (threshold 0.5):
        y_true = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1]
        y_prob = [0.9, 0.8, 0.7, 0.2, 0.1, 0.6, 0.4, 0.3, 0.55, 0.85]
        y_pred = [1, 1, 1, 0, 0, 1, 0, 0, 1, 1]
        ⇒ TP=4, FN=1, FP=2, TN=3   (n=10, positives=5, negatives=5)
    """

    def setUp(self) -> None:
        self.y_true = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1]
        self.y_prob = [0.9, 0.8, 0.7, 0.2, 0.1, 0.6, 0.4, 0.3, 0.55, 0.85]
        self.metrics = compute_binary_classification_metrics(self.y_true, self.y_prob, threshold=0.5)

    def test_confusion_matrix_counts(self) -> None:
        self.assertEqual(self.metrics["tp"], 4.0)
        self.assertEqual(self.metrics["fn"], 1.0)
        self.assertEqual(self.metrics["fp"], 2.0)
        self.assertEqual(self.metrics["tn"], 3.0)
        self.assertEqual(self.metrics["support"], 10.0)
        self.assertEqual(self.metrics["support_positive"], 5.0)
        self.assertEqual(self.metrics["support_negative"], 5.0)

    def test_sensitivity_specificity_and_aliases(self) -> None:
        self.assertAlmostEqual(self.metrics["sensitivity"], 4 / 5)
        self.assertAlmostEqual(self.metrics["recall_positive"], 4 / 5)
        self.assertAlmostEqual(self.metrics["specificity"], 3 / 5)
        self.assertAlmostEqual(self.metrics["precision_positive"], 4 / 6)
        # f1_positive = 2 * P * R / (P + R)
        self.assertAlmostEqual(
            self.metrics["f1_positive"],
            (2 * (4 / 6) * (4 / 5)) / ((4 / 6) + (4 / 5)),
        )

    def test_negative_predictive_value_and_rates(self) -> None:
        self.assertAlmostEqual(self.metrics["npv"], 3 / 4)        # TN / (TN + FN)
        self.assertAlmostEqual(self.metrics["fpr"], 2 / 5)        # FP / (FP + TN)
        self.assertAlmostEqual(self.metrics["fnr"], 1 / 5)        # FN / (FN + TP)
        self.assertAlmostEqual(self.metrics["fdr"], 2 / 6)        # FP / (FP + TP)
        self.assertAlmostEqual(self.metrics["for"], 1 / 4)        # FN / (FN + TN)

    def test_mcc_and_cohen_kappa_match_hand_calculation(self) -> None:
        # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        expected_mcc = (4 * 3 - 2 * 1) / math.sqrt((4 + 2) * (4 + 1) * (3 + 2) * (3 + 1))
        self.assertAlmostEqual(self.metrics["mcc"], expected_mcc, places=6)
        # Cohen's kappa: po = 7/10; pe = (6*5 + 4*5)/100 = 0.5; kappa = (0.7 - 0.5)/(1 - 0.5)
        self.assertAlmostEqual(self.metrics["cohen_kappa"], (0.7 - 0.5) / (1 - 0.5), places=6)

    def test_macro_and_weighted_averages(self) -> None:
        # With n_pos = n_neg = 5, macro and weighted averages must match.
        self.assertAlmostEqual(self.metrics["macro_f1"], self.metrics["weighted_f1"])
        self.assertAlmostEqual(self.metrics["macro_precision"], self.metrics["weighted_precision"])
        self.assertAlmostEqual(self.metrics["macro_recall"], self.metrics["weighted_recall"])

    def test_existing_keys_preserved(self) -> None:
        # Backward-compatibility guard: every previously-emitted JSON key must remain.
        for key in (
            "threshold", "accuracy", "balanced_accuracy", "f1", "precision", "recall",
            "support", "support_positive", "support_negative",
            "roc_auc", "pr_auc", "tn", "fp", "fn", "tp",
        ):
            self.assertIn(key, self.metrics)

    def test_per_class_report_shape(self) -> None:
        report = compute_per_class_report(
            self.y_true, self.y_prob, threshold=0.5, class_names=("Normal", "Anomaly"),
        )
        self.assertIn("Normal", report)
        self.assertIn("Anomaly", report)
        self.assertEqual(report["Anomaly"]["support"], 5)
        self.assertAlmostEqual(report["Anomaly"]["recall"], 4 / 5)


class SingleClassRobustnessTest(unittest.TestCase):
    """ROC-AUC / PR-AUC must not crash when y_true contains a single class."""

    def test_all_negative_y_true_returns_nan_aucs(self) -> None:
        y_true = [0, 0, 0, 0]
        y_prob = [0.1, 0.4, 0.7, 0.9]
        m = compute_binary_classification_metrics(y_true, y_prob, threshold=0.5)
        self.assertTrue(math.isnan(m["roc_auc"]))
        self.assertTrue(math.isnan(m["pr_auc"]))
        # Confusion-matrix rates must still be finite.
        self.assertEqual(m["tp"], 0.0)
        self.assertEqual(m["fn"], 0.0)
        self.assertFalse(math.isnan(m["specificity"]))
        self.assertFalse(math.isnan(m["mcc"]))

    def test_all_positive_y_true_returns_nan_aucs(self) -> None:
        y_true = [1, 1, 1, 1]
        y_prob = [0.1, 0.4, 0.7, 0.9]
        m = compute_binary_classification_metrics(y_true, y_prob, threshold=0.5)
        self.assertTrue(math.isnan(m["roc_auc"]))
        self.assertTrue(math.isnan(m["pr_auc"]))
        self.assertEqual(m["tn"], 0.0)


class FinalEvaluationImportTest(unittest.TestCase):
    """The Optuna pipeline imports run_final_evaluation; guard the import surface."""

    def test_run_final_evaluation_importable_with_signature(self) -> None:
        import inspect

        from evaluate_final import run_final_evaluation

        signature = inspect.signature(run_final_evaluation)
        params = signature.parameters
        self.assertIn("run_dir", params)
        self.assertIn("output_dir", params)
        self.assertIn("use_saved_predictions", params)
        self.assertIn("threshold", params)
        # Default contract: saved predictions preferred, no threshold override.
        self.assertIs(params["output_dir"].default, None)
        self.assertEqual(params["use_saved_predictions"].default, True)
        self.assertIs(params["threshold"].default, None)


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
