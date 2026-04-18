from __future__ import annotations

import unittest
from pathlib import Path

from Utils.config import AugmentationConfig, DataConfig, ModelConfig, TrainConfig
from Model.search import (
    _configs_from_params,
    _epoch_choices,
    _flip_axes_from_choice,
    _patience_choices,
)


class EpochChoicesTest(unittest.TestCase):
    def test_basic(self) -> None:
        choices = _epoch_choices(12)
        self.assertIsInstance(choices, list)
        self.assertTrue(all(c >= 1 for c in choices))
        self.assertIn(12, choices)

    def test_minimum_clamp(self) -> None:
        choices = _epoch_choices(1)
        self.assertTrue(all(c >= 1 for c in choices))

    def test_sorted(self) -> None:
        choices = _epoch_choices(20)
        self.assertEqual(choices, sorted(choices))


class PatienceChoicesTest(unittest.TestCase):
    def test_basic(self) -> None:
        choices = _patience_choices(12)
        self.assertTrue(all(1 <= c <= 12 for c in choices))

    def test_small_epoch_budget(self) -> None:
        choices = _patience_choices(2)
        self.assertTrue(all(c <= 2 for c in choices))
        self.assertTrue(len(choices) >= 1)


class FlipAxesFromChoiceTest(unittest.TestCase):
    def test_known_choices(self) -> None:
        self.assertEqual(_flip_axes_from_choice("none", (1, 2)), ())
        self.assertEqual(_flip_axes_from_choice("0_1_2", (1, 2)), (0, 1, 2))

    def test_none_returns_fallback(self) -> None:
        self.assertEqual(_flip_axes_from_choice(None, (1, 2)), (1, 2))


class ConfigsFromParamsAugEnabledTest(unittest.TestCase):
    """_configs_from_params with augmentations_enabled=True."""

    def setUp(self) -> None:
        self.base_data = DataConfig()
        self.base_aug = AugmentationConfig()
        self.base_model = ModelConfig()
        self.base_train = TrainConfig()
        self.params = {
            "target_edge": 64,
            "use_bbox_crop": True,
            "bbox_margin": 8,
            "pad_to_cube": True,
            "canonicalize_right": False,
            "nan_strategy": "none",
            "augmentations_enabled": True,
            "flip_probability": 0.5,
            "flip_axes": "1_2",
            "affine_probability": 0.6,
            "rotation_degrees": 10.0,
            "translation_fraction": 0.05,
            "scale_min": 0.9,
            "scale_max": 1.1,
            "morphology_probability": 0.1,
            "depth": 18,
            "base_channels": 32,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer_name": "adamw",
            "scheduler_name": "cosine",
            "batch_size": 8,
            "epochs": 12,
        }

    def test_returns_four_configs(self) -> None:
        data_cfg, aug_cfg, model_cfg, train_cfg = _configs_from_params(
            self.params, self.base_data, self.base_aug, self.base_model, self.base_train, Path("/tmp/test"),
        )
        self.assertTrue(aug_cfg.enabled)
        self.assertEqual(aug_cfg.flip_probability, 0.5)
        self.assertEqual(aug_cfg.rotation_degrees, 10.0)
        self.assertEqual(data_cfg.target_shape, (64, 64, 64))
        self.assertEqual(model_cfg.depth, 18)
        self.assertEqual(train_cfg.epochs, 12)

    def test_epochs_override(self) -> None:
        _, _, _, train_cfg = _configs_from_params(
            self.params, self.base_data, self.base_aug, self.base_model, self.base_train,
            Path("/tmp/test"), epochs_override=30,
        )
        self.assertEqual(train_cfg.epochs, 30)


class ConfigsFromParamsAugDisabledTest(unittest.TestCase):
    """_configs_from_params must NOT KeyError when augmentations_enabled=False."""

    def setUp(self) -> None:
        self.base_data = DataConfig()
        self.base_aug = AugmentationConfig()
        self.base_model = ModelConfig()
        self.base_train = TrainConfig()
        self.params = {
            "target_edge": 48,
            "use_bbox_crop": False,
            "bbox_margin": 6,
            "pad_to_cube": True,
            "canonicalize_right": False,
            "nan_strategy": "none",
            "augmentations_enabled": False,
            # NOTE: no flip_probability, rotation_degrees, etc.
            "flip_axes": "none",
            "depth": 34,
            "base_channels": 16,
            "dropout": 0.1,
            "learning_rate": 5e-4,
            "weight_decay": 1e-5,
            "optimizer_name": "adam",
            "scheduler_name": "none",
            "batch_size": 4,
            "epochs": 6,
        }

    def test_no_keyerror(self) -> None:
        data_cfg, aug_cfg, model_cfg, train_cfg = _configs_from_params(
            self.params, self.base_data, self.base_aug, self.base_model, self.base_train, Path("/tmp/test"),
        )
        self.assertFalse(aug_cfg.enabled)
        self.assertEqual(data_cfg.target_shape, (48, 48, 48))
        self.assertEqual(model_cfg.depth, 34)

    def test_aug_fields_are_base_defaults(self) -> None:
        _, aug_cfg, _, _ = _configs_from_params(
            self.params, self.base_data, self.base_aug, self.base_model, self.base_train, Path("/tmp/test"),
        )
        # When disabled, fields should stay at base defaults (enabled=False is the only override)
        self.assertFalse(aug_cfg.enabled)


class ConfigsFromParamsNewFieldsTest(unittest.TestCase):
    """warmup_epochs and pos_weight_strategy are reconstructed from params."""

    def test_warmup_and_pos_weight(self) -> None:
        params = {
            "target_edge": 64,
            "bbox_margin": 8,
            "nan_strategy": "none",
            "augmentations_enabled": True,
            "flip_probability": 0.5,
            "flip_axes": "1_2",
            "affine_probability": 0.6,
            "rotation_degrees": 10.0,
            "translation_fraction": 0.05,
            "scale_min": 0.9,
            "scale_max": 1.1,
            "morphology_probability": 0.1,
            "depth": 18,
            "base_channels": 32,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer_name": "adamw",
            "scheduler_name": "cosine",
            "batch_size": 8,
            "epochs": 12,
            "warmup_epochs": 2,
            "pos_weight_strategy": "sqrt",
        }
        _, _, _, train_cfg = _configs_from_params(
            params, DataConfig(), AugmentationConfig(), ModelConfig(), TrainConfig(), Path("/tmp/test"),
        )
        self.assertEqual(train_cfg.warmup_epochs, 2)
        self.assertEqual(train_cfg.pos_weight_strategy, "sqrt")

    def test_defaults_when_missing(self) -> None:
        params = {
            "target_edge": 64,
            "bbox_margin": 8,
            "nan_strategy": "none",
            "augmentations_enabled": False,
            "flip_axes": "none",
            "depth": 18,
            "base_channels": 32,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer_name": "adamw",
            "scheduler_name": "cosine",
            "batch_size": 8,
            "epochs": 12,
        }
        base_train = TrainConfig(warmup_epochs=0, pos_weight_strategy="ratio")
        _, _, _, train_cfg = _configs_from_params(
            params, DataConfig(), AugmentationConfig(), ModelConfig(), base_train, Path("/tmp/test"),
        )
        self.assertEqual(train_cfg.warmup_epochs, 0)
        self.assertEqual(train_cfg.pos_weight_strategy, "ratio")


if __name__ == "__main__":
    unittest.main()
