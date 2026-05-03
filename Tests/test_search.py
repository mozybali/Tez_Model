from __future__ import annotations

import unittest
from pathlib import Path

import optuna

from Utils.config import AugmentationConfig, DataConfig, ModelConfig, TrainConfig
from Model.search import (
    _configs_from_params,
    _epoch_choices,
    _flip_axes_from_choice,
    _patience_choices,
    _resolve_flip_axes,
    _sample_threshold_config,
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


class ResolveFlipAxesTest(unittest.TestCase):
    """When canonicalize_right is True, the canonicalization axis must be excluded
    from augmentation flips — otherwise the random flip undoes the L→R alignment."""

    def test_removes_canonical_axis(self) -> None:
        self.assertEqual(_resolve_flip_axes("0_2", (1, 2), True, 2), (0,))

    def test_removes_canonical_axis_single(self) -> None:
        self.assertEqual(_resolve_flip_axes("2", (1, 2), True, 2), ())

    def test_keeps_all_axes_when_not_canonicalizing(self) -> None:
        self.assertEqual(_resolve_flip_axes("0_2", (1, 2), False, 2), (0, 2))

    def test_other_axes_untouched(self) -> None:
        self.assertEqual(_resolve_flip_axes("0_1_2", (1, 2), True, 1), (0, 2))

    def test_fallback_respected_when_choice_none(self) -> None:
        self.assertEqual(_resolve_flip_axes(None, (1, 2), True, 2), (1,))


class ThresholdConfigSearchTest(unittest.TestCase):
    def test_searches_all_threshold_selection_choices(self) -> None:
        captured: list[str] = []

        def objective(trial: optuna.Trial) -> float:
            selection, _ = _sample_threshold_config(trial, TrainConfig())
            captured.append(selection)
            return 0.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.GridSampler(
                {
                    "threshold_selection": ["youden", "f1", "fbeta"],
                    "threshold_fbeta": [2.0],
                }
            ),
        )
        study.optimize(objective, n_trials=3)

        self.assertEqual(set(captured), {"youden", "f1", "fbeta"})

    def test_fbeta_beta_is_only_sampled_for_fbeta_selection(self) -> None:
        captured: dict[str, tuple[str, float, bool]] = {}

        def objective(trial: optuna.Trial) -> float:
            selection, beta = _sample_threshold_config(trial, TrainConfig(threshold_fbeta=1.0))
            captured[selection] = (selection, beta, "threshold_fbeta" in trial.params)
            return 0.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.GridSampler(
                {
                    "threshold_selection": ["youden", "f1", "fbeta"],
                    "threshold_fbeta": [2.0],
                }
            ),
        )
        study.optimize(objective, n_trials=3)

        self.assertEqual(captured["youden"], ("youden", 1.0, False))
        self.assertEqual(captured["f1"], ("f1", 1.0, False))
        self.assertEqual(captured["fbeta"], ("fbeta", 2.0, True))

    def test_can_limit_threshold_selection_choices(self) -> None:
        captured: list[str] = []

        def objective(trial: optuna.Trial) -> float:
            selection, _ = _sample_threshold_config(
                trial,
                TrainConfig(),
                threshold_selection_choices=["youden", "f1"],
            )
            captured.append(selection)
            return 0.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.GridSampler(
                {"threshold_selection": ["youden", "f1"]}
            ),
        )
        study.optimize(objective, n_trials=2)

        self.assertEqual(set(captured), {"youden", "f1"})

    def test_resuming_old_single_fbeta_study_uses_v2_param(self) -> None:
        study = optuna.create_study(direction="maximize")

        def old_objective(trial: optuna.Trial) -> float:
            trial.suggest_categorical("threshold_selection", ["fbeta"])
            return 0.0

        study.optimize(old_objective, n_trials=1)

        trial = study.ask()
        selection, _ = _sample_threshold_config(trial, TrainConfig())

        self.assertIn(selection, {"youden", "f1", "fbeta"})
        self.assertIn("threshold_selection_v2", trial.params)
        self.assertNotIn("threshold_selection", trial.params)


class ConfigsFromParamsFlipAxisResolutionTest(unittest.TestCase):
    """Params reconstruction must also drop the canonicalization axis from flips."""

    def test_drops_canonical_axis_in_replay(self) -> None:
        params = {
            "target_edge": 64,
            "use_bbox_crop": True,
            "bbox_margin": 8,
            "pad_to_cube": True,
            "canonicalize_right": True,
            "right_flip_axis": 2,
            "nan_strategy": "none",
            "augmentations_enabled": True,
            "flip_probability": 0.5,
            "flip_axes": "0_2",
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
        _, aug_cfg, _, _ = _configs_from_params(
            params, DataConfig(), AugmentationConfig(), ModelConfig(), TrainConfig(), Path("/tmp/test"),
        )
        self.assertEqual(aug_cfg.flip_axes, (0,))


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


class ConfigsFromParamsArchitectureTest(unittest.TestCase):
    """Architecture field round-trips through _configs_from_params."""

    def _base_params(self) -> dict:
        return {
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

    def test_unet_architecture_is_reconstructed(self) -> None:
        params = self._base_params()
        params.update({
            "architecture": "unet3d",
            "unet_depth": 3,
            "unet_base_channels": 16,
            "unet_channel_multiplier": 2,
        })
        _, _, model_cfg, _ = _configs_from_params(
            params, DataConfig(), AugmentationConfig(), ModelConfig(), TrainConfig(), Path("/tmp/test"),
        )
        self.assertEqual(model_cfg.architecture, "unet3d")
        self.assertEqual(model_cfg.unet_depth, 3)
        self.assertEqual(model_cfg.unet_base_channels, 16)
        self.assertEqual(model_cfg.unet_channel_multiplier, 2)

    def test_missing_architecture_defaults_to_resnet3d(self) -> None:
        """Old param dicts (pre–U-Net integration) lack 'architecture'."""
        params = self._base_params()  # no architecture key
        _, _, model_cfg, _ = _configs_from_params(
            params, DataConfig(), AugmentationConfig(), ModelConfig(), TrainConfig(), Path("/tmp/test"),
        )
        self.assertEqual(model_cfg.architecture, "resnet3d")
        self.assertEqual(model_cfg.depth, 18)
        self.assertEqual(model_cfg.base_channels, 32)

    def test_pointnet_architecture_is_reconstructed(self) -> None:
        params = self._base_params()
        params.update({
            "architecture": "pointnet",
            "pointnet_num_points": 2048,
            "pointnet_global_dim": 256,
            "pointnet_mlp_variant": "large",
            "pointnet_head_hidden_dim": 64,
            "pointnet_point_features": 4,
            "pointnet_use_input_transform": True,
        })
        _, _, model_cfg, _ = _configs_from_params(
            params, DataConfig(), AugmentationConfig(), ModelConfig(), TrainConfig(), Path("/tmp/test"),
        )
        self.assertEqual(model_cfg.architecture, "pointnet")
        self.assertEqual(model_cfg.pointnet_num_points, 2048)
        self.assertEqual(model_cfg.pointnet_global_dim, 256)
        self.assertEqual(model_cfg.pointnet_mlp_channels, (64, 128, 256, 512))
        self.assertEqual(model_cfg.pointnet_head_hidden_dim, 64)
        self.assertEqual(model_cfg.pointnet_point_features, 4)
        self.assertTrue(model_cfg.pointnet_use_input_transform)

    def test_pointnet_missing_params_use_base_defaults(self) -> None:
        params = self._base_params()
        params.update({"architecture": "pointnet"})  # no pointnet_* keys
        base = ModelConfig(
            pointnet_num_points=777,
            pointnet_global_dim=333,
            pointnet_mlp_channels=(8, 16),
            pointnet_head_hidden_dim=11,
            pointnet_point_features=3,
            pointnet_use_input_transform=False,
        )
        _, _, model_cfg, _ = _configs_from_params(
            params, DataConfig(), AugmentationConfig(), base, TrainConfig(), Path("/tmp/test"),
        )
        self.assertEqual(model_cfg.architecture, "pointnet")
        self.assertEqual(model_cfg.pointnet_num_points, 777)
        self.assertEqual(model_cfg.pointnet_global_dim, 333)
        self.assertEqual(model_cfg.pointnet_mlp_channels, (8, 16))
        self.assertEqual(model_cfg.pointnet_head_hidden_dim, 11)
        self.assertEqual(model_cfg.pointnet_point_features, 3)
        self.assertFalse(model_cfg.pointnet_use_input_transform)


if __name__ == "__main__":
    unittest.main()
