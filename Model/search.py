from __future__ import annotations

import os

# Cap BLAS thread pools before numpy/torch are imported. Windows OpenBLAS can
# raise "Memory allocation still failed after 10 retries" under bursts of
# many short-lived BLAS calls (bootstrap CI + sklearn metrics); a single
# thread avoids the contention without measurably slowing down our sizes.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna

from Model.engine import release_gpu_memory, run_cross_validation, run_training, save_json, make_json_safe
from Utils.config import AugmentationConfig, DataConfig, ModelConfig, SearchConfig, TrainConfig, to_serializable


_FLIP_AXIS_CHOICES: dict[str, tuple[int, ...]] = {
    "none": (),
    "0": (0,),
    "1": (1,),
    "2": (2,),
    "0_1": (0, 1),
    "0_2": (0, 2),
    "1_2": (1, 2),
    "0_1_2": (0, 1, 2),
}

_NAN_STRATEGY_CHOICES = (
    "none",
    "drop_record",
    "fill_zero",
    "fill_mean",
    "fill_median",
    "fill_constant",
)


def _epoch_choices(max_epochs: int) -> list[int]:
    max_epochs = max(1, int(max_epochs))
    return sorted({max(1, int(round(max_epochs * fraction))) for fraction in (0.5, 0.75, 1.0)})


def _patience_choices(max_epochs: int) -> list[int]:
    """Patience budget: favor longer patience so training isn't cut short.

    We avoid patience < 5 because prior runs stopped at epoch 10 and still had
    room to improve. The lower bound scales with max_epochs.
    """
    max_epochs = max(1, int(max_epochs))
    lower = max(5, int(round(max_epochs * 0.25)))
    candidates = {lower, 8, 10, 12, 15, max(1, int(round(max_epochs * 0.5)))}
    return sorted(choice for choice in candidates if choice <= max_epochs) or [max_epochs]


def _flip_axes_from_choice(choice: str | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if choice is None:
        return fallback
    return _FLIP_AXIS_CHOICES[choice]


def _resolve_flip_axes(
    choice: str | None,
    fallback: tuple[int, ...],
    canonicalize_right: bool,
    right_flip_axis: int,
) -> tuple[int, ...]:
    axes = _flip_axes_from_choice(choice, fallback)
    if canonicalize_right:
        axes = tuple(axis for axis in axes if axis != right_flip_axis)
    return axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bayesian hyperparameter search with Optuna.")
    parser.add_argument("--info-csv", type=Path, default=Path("ALAN/info.csv"))
    parser.add_argument("--volumes-dir", type=Path, default=Path("ALAN/alan"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("ALAN/metadata.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("ALAN/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/optuna"))
    parser.add_argument("--study-name", default="alan_resnet3d")
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="Use k-fold CV for the HPO objective (mean primary metric across folds). "
             "Default 1 = single ZS-train/ZS-dev split (legacy behavior). "
             "Common choices: 3 for ResNet3D/UNet3D, 5 for PointNet. "
             "The final best_run retrain always uses the single ZS split, regardless of this flag.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum per-trial epoch budget; Optuna samples shorter budgets up to this value.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--batch-size-choices",
        type=int,
        nargs="+",
        default=None,
        help="Optuna batch_size search space. If omitted, defaults to [8,16,24,32]. "
             "Pass a single value (e.g. --batch-size-choices 48) to fix batch size.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4,
                        help="Minimum score improvement to reset patience counter.")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-tabular-features", action="store_true")
    parser.add_argument("--tabular-hidden-dim", type=int, default=16)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--flip-axes", nargs="+", type=int, default=(1, 2))
    parser.add_argument(
        "--architecture",
        choices=["resnet3d", "unet3d", "pointnet"],
        default="resnet3d",
        help="Backbone architecture to search over. Defaults to resnet3d to preserve prior behavior.",
    )
    parser.add_argument("--final-epochs", type=int, default=None,
                        help="Epochs for the best_run final retrain (defaults to the best trial's epoch budget).")
    parser.add_argument("--final-patience", type=int, default=None,
                        help="Early-stopping patience for the best_run final retrain. "
                             "Overrides the best trial's sampled patience so a longer final budget "
                             "(--final-epochs) isn't cut short. Defaults to the best trial's patience.")
    parser.add_argument("--primary-metric",
                        choices=["roc_auc", "pr_auc", "balanced_accuracy", "f1", "mcc", "constrained_f1"],
                        default="roc_auc",
                        help="Metric to maximize (fallback chain: pr_auc → balanced_accuracy → f1). "
                             "'mcc' and 'constrained_f1' guard against the recall-greedy collapse where "
                             "a fold picks threshold≈0.05 and reports high F1/PR-AUC at near-zero specificity.")
    parser.add_argument("--threshold-min-specificity", type=float, default=None,
                        help="Floor on val specificity when picking the F-beta threshold. "
                             "Without this, β>1 will pick threshold≈0.05 on small val sets.")
    parser.add_argument("--threshold-min-precision", type=float, default=None,
                        help="Floor on val precision for F-beta threshold selection.")
    parser.add_argument("--constrained-f1-min-specificity", type=float, default=0.80,
                        help="Specificity floor used when primary-metric=constrained_f1. "
                             "F1 collapses to 0 below this on any val/fold so HPO cannot reward "
                             "models that just spam positives.")
    parser.add_argument("--isotonic-min-samples", type=int, default=150,
                        help="Skip isotonic calibration when val n is below this. "
                             "Isotonic on tiny vals (<~150) overfits ECE to 0 and turns "
                             "calibrated probs into a coarse staircase.")
    parser.add_argument("--cv-score-std-penalty", type=float, default=0.0,
                        help="Penalize fold disagreement: HPO score = mean − penalty * std "
                             "of the primary metric across folds. 0.0 = legacy mean-only.")
    parser.add_argument("--nan-strategy",
                        choices=["none", "drop_record", "fill_zero", "fill_mean", "fill_median", "fill_constant"],
                        default="none")
    parser.add_argument("--nan-fill-value", type=float, default=0.0)
    parser.add_argument(
        "--force-augmentation",
        action="store_true",
        help="Force augmentations on for every trial (skip the on/off categorical and only search augmentation strength).",
    )
    parser.add_argument(
        "--force-amp",
        action="store_true",
        help="Force AMP (mixed precision) on for every trial; skips the amp on/off categorical.",
    )
    return parser.parse_args()


def _sample_trial_configs(
    trial: optuna.Trial,
    base_data: DataConfig,
    base_aug: AugmentationConfig,
    base_model: ModelConfig,
    base_train: TrainConfig,
    output_dir: Path,
    batch_size_choices: list[int] | None = None,
    architecture: str = "resnet3d",
    force_augmentation: bool = False,
    force_amp: bool = False,
) -> tuple[DataConfig, AugmentationConfig, ModelConfig, TrainConfig]:
    target_edge = trial.suggest_categorical("target_edge", [48, 64, 80])
    flip_axes_choice = trial.suggest_categorical("flip_axes", list(_FLIP_AXIS_CHOICES))
    use_tabular_features = (
        trial.suggest_categorical("use_tabular_features", [True, False])
        if base_model.use_tabular_features
        else False
    )
    train_epochs = trial.suggest_categorical("epochs", _epoch_choices(base_train.epochs))

    canonicalize_right = trial.suggest_categorical("canonicalize_right", [False, True])
    right_flip_axis = (
        trial.suggest_categorical("right_flip_axis", [0, 1, 2])
        if canonicalize_right
        else base_data.right_flip_axis
    )
    nan_strategy = trial.suggest_categorical("nan_strategy", list(_NAN_STRATEGY_CHOICES))
    nan_fill_value = (
        trial.suggest_categorical("nan_fill_value", [0.0, 0.5, 1.0])
        if nan_strategy == "fill_constant"
        else base_data.nan_fill_value
    )

    data_config = replace(
        base_data,
        target_shape=(target_edge, target_edge, target_edge),
        use_bbox_crop=trial.suggest_categorical("use_bbox_crop", [True, False]),
        bbox_margin=trial.suggest_int("bbox_margin", 4, 16, step=2),
        pad_to_cube=trial.suggest_categorical("pad_to_cube", [True, False]),
        canonicalize_right=canonicalize_right,
        right_flip_axis=right_flip_axis,
        nan_strategy=nan_strategy,
        nan_fill_value=nan_fill_value,
    )

    augmentations_enabled = (
        True
        if force_augmentation
        else trial.suggest_categorical("augmentations_enabled", [True, False])
    )
    if augmentations_enabled:
        aug_config = replace(
            base_aug,
            enabled=True,
            flip_probability=trial.suggest_float("flip_probability", 0.2, 0.7),
            flip_axes=_resolve_flip_axes(
                flip_axes_choice,
                base_aug.flip_axes,
                canonicalize_right,
                right_flip_axis,
            ),
            affine_probability=trial.suggest_float("affine_probability", 0.3, 0.8),
            rotation_degrees=trial.suggest_float("rotation_degrees", 4.0, 15.0),
            translation_fraction=trial.suggest_float("translation_fraction", 0.02, 0.10),
            scale_min=trial.suggest_float("scale_min", 0.85, 0.98),
            scale_max=trial.suggest_float("scale_max", 1.02, 1.15),
            morphology_probability=trial.suggest_float("morphology_probability", 0.0, 0.2),
        )
    else:
        aug_config = replace(base_aug, enabled=False)

    trial.set_user_attr("architecture", architecture)
    depth = base_model.depth
    base_channels_val = base_model.base_channels
    unet_depth = base_model.unet_depth
    unet_base_channels = base_model.unet_base_channels
    unet_channel_multiplier = base_model.unet_channel_multiplier
    unet_bottleneck_channels = base_model.unet_bottleneck_channels
    pointnet_num_points = base_model.pointnet_num_points
    pointnet_global_dim = base_model.pointnet_global_dim
    pointnet_mlp_channels = base_model.pointnet_mlp_channels
    pointnet_head_hidden_dim = base_model.pointnet_head_hidden_dim
    pointnet_point_features = base_model.pointnet_point_features
    pointnet_use_input_transform = base_model.pointnet_use_input_transform
    pointnet_binary_threshold = base_model.pointnet_binary_threshold

    if architecture == "resnet3d":
        depth = trial.suggest_categorical("depth", [18, 34])
        base_channels_val = trial.suggest_categorical("base_channels", [16, 24, 32])
    elif architecture == "unet3d":
        # Keep memory usage conservative — volumes can be 80^3, so cap U-Net size.
        unet_depth = trial.suggest_categorical("unet_depth", [3, 4])
        unet_base_channels = trial.suggest_categorical("unet_base_channels", [8, 16, 24, 32])
        unet_channel_multiplier = trial.suggest_categorical("unet_channel_multiplier", [2, 3])
        unet_bottleneck_choice = trial.suggest_categorical(
            "unet_bottleneck_choice", ["auto", "64", "128", "256"]
        )
        unet_bottleneck_channels = (
            None if unet_bottleneck_choice == "auto" else int(unet_bottleneck_choice)
        )
    else:  # pointnet — conservative memory use on small-batch GPUs
        pointnet_num_points = trial.suggest_categorical(
            "pointnet_num_points", [1024, 2048, 4096, 8192]
        )
        pointnet_global_dim = trial.suggest_categorical(
            "pointnet_global_dim", [256, 512, 1024]
        )
        pointnet_mlp_variant = trial.suggest_categorical(
            "pointnet_mlp_variant", ["small", "medium", "large"]
        )
        pointnet_mlp_channels = {
            "small": (64, 128),
            "medium": (64, 128, 256),
            "large": (64, 128, 256, 512),
        }[pointnet_mlp_variant]
        pointnet_head_hidden_dim = trial.suggest_categorical(
            "pointnet_head_hidden_dim", [0, 64, 128, 256]
        )
        # Fixed at 4 so PointNet always sees (xyz, intensity) — the 4th channel
        # now carries the real voxel intensity, not a constant occupancy=1.
        pointnet_point_features = trial.suggest_categorical(
            "pointnet_point_features", [4]
        )
        pointnet_use_input_transform = trial.suggest_categorical(
            "pointnet_use_input_transform", [False, True]
        )
        # Fixed mid-value: 0.3 was too permissive (background bleed), 0.7 too
        # restrictive on partial-volume voxels. Re-search later if needed.
        pointnet_binary_threshold = trial.suggest_categorical(
            "pointnet_binary_threshold", [0.5]
        )
    model_config = replace(
        base_model,
        architecture=architecture,
        depth=depth,
        base_channels=base_channels_val,
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        use_tabular_features=use_tabular_features,
        tabular_hidden_dim=(
            trial.suggest_categorical("tabular_hidden_dim", [8, 16, 32])
            if use_tabular_features
            else base_model.tabular_hidden_dim
        ),
        norm_type=(norm_type := trial.suggest_categorical("norm_type", ["batch", "group"])),
        group_norm_groups=(
            trial.suggest_categorical("group_norm_groups", [4, 8, 16])
            if norm_type == "group"
            else base_model.group_norm_groups
        ),
        unet_depth=unet_depth,
        unet_base_channels=unet_base_channels,
        unet_channel_multiplier=unet_channel_multiplier,
        unet_bottleneck_channels=unet_bottleneck_channels,
        pointnet_num_points=pointnet_num_points,
        pointnet_point_features=pointnet_point_features,
        pointnet_mlp_channels=pointnet_mlp_channels,
        pointnet_global_dim=pointnet_global_dim,
        pointnet_head_hidden_dim=pointnet_head_hidden_dim,
        pointnet_binary_threshold=pointnet_binary_threshold,
        pointnet_use_input_transform=pointnet_use_input_transform,
    )

    optimizer_name = trial.suggest_categorical("optimizer_name", ["adam", "adamw", "sgd"])
    sgd_momentum = (
        trial.suggest_float("sgd_momentum", 0.0, 0.95)
        if optimizer_name == "sgd"
        else base_train.sgd_momentum
    )
    scheduler_name = trial.suggest_categorical("scheduler_name", ["cosine", "none"])
    # Warmup lower bound of 2 (prior run suffered a big loss spike at epoch 2 with only 1 warmup epoch).
    warmup_upper = max(3, train_epochs // 3)
    warmup_epochs = (
        trial.suggest_int("warmup_epochs", 2, warmup_upper)
        if scheduler_name != "none"
        else 0
    )
    loss_type = trial.suggest_categorical("loss_type", ["bce", "focal"])
    # Narrowed lower bound: γ < 2 barely down-weights easy negatives, which is
    # the regime we want to escape on this 3:1-imbalanced anomaly split.
    focal_gamma = (
        trial.suggest_float("focal_gamma", 2.0, 3.0)
        if loss_type == "focal"
        else base_train.focal_gamma
    )
    # When the user explicitly optimizes F1 (--primary-metric f1), align the
    # threshold selector: β=1.0 → F1. Otherwise keep recall-priority β ∈ [1.5, 2.5]
    # (Youden/F1 don't penalize missed anomalies enough for the clinical use-case).
    threshold_selection = trial.suggest_categorical(
        "threshold_selection", ["fbeta"],
    )
    if base_train.primary_metric == "f1":
        threshold_fbeta = 1.0
    else:
        threshold_fbeta = trial.suggest_float("threshold_fbeta", 1.5, 2.5)
    # Narrowed strategy choices to the two that actually meaningfully reweight
    # the minority class on this 3:1-imbalanced split. Mutual exclusion below:
    # if the WeightedRandomSampler is on, batches are already balanced — adding
    # loss reweighting on top double-counts and inflates the gradient.
    sampled_pos_strategy = trial.suggest_categorical(
        "pos_weight_strategy", ["effective", "inverse"]
    )
    use_weighted_sampler = trial.suggest_categorical(
        "use_weighted_sampler", [False, True]
    )
    pos_weight_strategy = "none" if use_weighted_sampler else sampled_pos_strategy
    train_config = replace(
        base_train,
        output_dir=output_dir / f"trial_{trial.number:03d}",
        epochs=train_epochs,
        learning_rate=trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        optimizer_name=optimizer_name,
        sgd_momentum=sgd_momentum,
        scheduler_name=scheduler_name,
        warmup_epochs=warmup_epochs,
        pos_weight_strategy=pos_weight_strategy,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        batch_size=(
            batch_size_choices[0]
            if batch_size_choices and len(batch_size_choices) == 1
            else trial.suggest_categorical(
                "batch_size",
                batch_size_choices if batch_size_choices else [8, 16, 24, 32],
            )
        ),
        early_stopping_patience=trial.suggest_categorical(
            "early_stopping_patience",
            _patience_choices(base_train.epochs),
        ),
        # Tighter clip bounds — 5.0 was too loose and allowed the loss spike at epoch 2.
        gradient_clip_norm=trial.suggest_categorical("gradient_clip_norm", [0.25, 0.5, 1.0, 2.0]),
        use_weighted_sampler=use_weighted_sampler,
        amp=True if force_amp else trial.suggest_categorical("amp", [True, False]),
        threshold_selection=threshold_selection,
        threshold_fbeta=threshold_fbeta,
        calibration_method=trial.suggest_categorical(
            "calibration_method", ["temperature", "isotonic", "temperature+isotonic"],
        ),
        tta_enabled=trial.suggest_categorical("tta_enabled", [False, True]),
    )
    return data_config, aug_config, model_config, train_config


def _configs_from_params(
    params: dict,
    base_data: DataConfig,
    base_aug: AugmentationConfig,
    base_model: ModelConfig,
    base_train: TrainConfig,
    output_dir: Path,
    epochs_override: int | None = None,
    architecture_override: str | None = None,
    patience_override: int | None = None,
) -> tuple[DataConfig, AugmentationConfig, ModelConfig, TrainConfig]:
    """Rebuild configs from a finished trial's params dict (no suggest_ calls)."""
    edge = params["target_edge"]
    data_config = replace(
        base_data,
        target_shape=(edge, edge, edge),
        use_bbox_crop=params.get("use_bbox_crop", base_data.use_bbox_crop),
        bbox_margin=params["bbox_margin"],
        pad_to_cube=params.get("pad_to_cube", base_data.pad_to_cube),
        canonicalize_right=params.get("canonicalize_right", base_data.canonicalize_right),
        right_flip_axis=params.get("right_flip_axis", base_data.right_flip_axis),
        nan_strategy=params.get("nan_strategy", base_data.nan_strategy),
        nan_fill_value=params.get("nan_fill_value", base_data.nan_fill_value),
    )
    augmentations_enabled = params.get("augmentations_enabled", base_aug.enabled)
    if augmentations_enabled:
        aug_config = replace(
            base_aug,
            enabled=True,
            flip_probability=params["flip_probability"],
            flip_axes=_resolve_flip_axes(
                params.get("flip_axes"),
                base_aug.flip_axes,
                data_config.canonicalize_right,
                data_config.right_flip_axis,
            ),
            affine_probability=params["affine_probability"],
            rotation_degrees=params["rotation_degrees"],
            translation_fraction=params["translation_fraction"],
            scale_min=params["scale_min"],
            scale_max=params["scale_max"],
            morphology_probability=params["morphology_probability"],
        )
    else:
        aug_config = replace(base_aug, enabled=False)
    # Missing "architecture" key means this trial was produced before the U-Net
    # integration — fall back to ResNet3D for backward compatibility.
    architecture = architecture_override or params.get("architecture", "resnet3d")
    pointnet_mlp_variant = params.get("pointnet_mlp_variant")
    pointnet_mlp_channels = {
        "small": (64, 128),
        "medium": (64, 128, 256),
        "large": (64, 128, 256, 512),
    }.get(pointnet_mlp_variant, tuple(base_model.pointnet_mlp_channels))
    model_config = replace(
        base_model,
        architecture=architecture,
        depth=params.get("depth", base_model.depth),
        base_channels=params.get("base_channels", base_model.base_channels),
        dropout=params["dropout"],
        use_tabular_features=params.get("use_tabular_features", base_model.use_tabular_features),
        tabular_hidden_dim=params.get("tabular_hidden_dim", base_model.tabular_hidden_dim),
        norm_type=params.get("norm_type", base_model.norm_type),
        group_norm_groups=params.get("group_norm_groups", base_model.group_norm_groups),
        unet_depth=params.get("unet_depth", base_model.unet_depth),
        unet_base_channels=params.get("unet_base_channels", base_model.unet_base_channels),
        unet_channel_multiplier=params.get(
            "unet_channel_multiplier", base_model.unet_channel_multiplier
        ),
        unet_bottleneck_channels=(
            None
            if params.get("unet_bottleneck_choice") == "auto"
            else (
                int(params["unet_bottleneck_choice"])
                if "unet_bottleneck_choice" in params
                else params.get("unet_bottleneck_channels", base_model.unet_bottleneck_channels)
            )
        ),
        pointnet_num_points=params.get("pointnet_num_points", base_model.pointnet_num_points),
        pointnet_point_features=params.get(
            "pointnet_point_features", base_model.pointnet_point_features
        ),
        pointnet_mlp_channels=pointnet_mlp_channels,
        pointnet_global_dim=params.get("pointnet_global_dim", base_model.pointnet_global_dim),
        pointnet_head_hidden_dim=params.get(
            "pointnet_head_hidden_dim", base_model.pointnet_head_hidden_dim
        ),
        pointnet_binary_threshold=params.get(
            "pointnet_binary_threshold", base_model.pointnet_binary_threshold
        ),
        pointnet_use_input_transform=params.get(
            "pointnet_use_input_transform", base_model.pointnet_use_input_transform
        ),
    )
    # Re-apply the sampler ↔ pos_weight mutual exclusion so the final retrain
    # matches what the trial actually trained with.
    sampler_on = params.get("use_weighted_sampler", base_train.use_weighted_sampler)
    sampled_pos_strategy = params.get("pos_weight_strategy", base_train.pos_weight_strategy)
    pos_weight_strategy = "none" if sampler_on else sampled_pos_strategy
    train_config = replace(
        base_train,
        output_dir=output_dir,
        epochs=epochs_override if epochs_override is not None else params.get("epochs", base_train.epochs),
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        optimizer_name=params["optimizer_name"],
        sgd_momentum=params.get("sgd_momentum", base_train.sgd_momentum),
        scheduler_name=params["scheduler_name"],
        warmup_epochs=params.get("warmup_epochs", base_train.warmup_epochs),
        pos_weight_strategy=pos_weight_strategy,
        loss_type=params.get("loss_type", base_train.loss_type),
        focal_gamma=params.get("focal_gamma", base_train.focal_gamma),
        batch_size=params.get("batch_size", base_train.batch_size),
        early_stopping_patience=(
            patience_override
            if patience_override is not None
            else params.get("early_stopping_patience", base_train.early_stopping_patience)
        ),
        gradient_clip_norm=params.get("gradient_clip_norm", base_train.gradient_clip_norm),
        use_weighted_sampler=params.get("use_weighted_sampler", base_train.use_weighted_sampler),
        amp=params.get("amp", base_train.amp),
        threshold_selection=params.get("threshold_selection", base_train.threshold_selection),
        threshold_fbeta=params.get("threshold_fbeta", base_train.threshold_fbeta),
        calibration_method=params.get("calibration_method", base_train.calibration_method),
        tta_enabled=params.get("tta_enabled", base_train.tta_enabled),
    )
    return data_config, aug_config, model_config, train_config


def _nan_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    args = parse_args()
    final_epochs = args.final_epochs

    search_config = SearchConfig(
        study_name=args.study_name,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        timeout_seconds=args.timeout_seconds,
        sampler_seed=args.seed,
    )
    base_data = DataConfig(
        info_csv=args.info_csv,
        volumes_dir=args.volumes_dir,
        metadata_csv=args.metadata_csv,
        summary_json=args.summary_json,
        nan_strategy=args.nan_strategy,
        nan_fill_value=args.nan_fill_value,
    )
    base_aug = AugmentationConfig(flip_axes=tuple(args.flip_axes))
    base_model = ModelConfig(
        use_tabular_features=not args.disable_tabular_features,
        tabular_hidden_dim=args.tabular_hidden_dim,
    )
    base_train = TrainConfig(
        output_dir=args.output_dir / "base",
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        gradient_clip_norm=args.gradient_clip_norm,
        primary_metric=args.primary_metric,
        decision_threshold=args.decision_threshold,
        amp=not args.disable_amp,
        threshold_min_specificity=args.threshold_min_specificity,
        threshold_min_precision=args.threshold_min_precision,
        constrained_f1_min_specificity=args.constrained_f1_min_specificity,
        isotonic_min_samples=args.isotonic_min_samples,
        cv_score_std_penalty=args.cv_score_std_penalty,
    )

    root_dir = Path(search_config.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # Leaderboard file: appended after each trial
    leaderboard_path = root_dir / "leaderboard.json"
    leaderboard: list[dict] = []
    if leaderboard_path.exists():
        leaderboard = json.loads(leaderboard_path.read_text())

    storage_path = root_dir / "optuna_study.db"
    storage_url = f"sqlite:///{storage_path}"

    sampler = optuna.samplers.TPESampler(seed=search_config.sampler_seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=search_config.study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        print(f"\n{'='*60}")
        print(f"  TRIAL {trial.number:03d}")
        print(f"{'='*60}")
        data_cfg, aug_cfg, model_cfg, train_cfg = _sample_trial_configs(
            trial=trial,
            base_data=base_data,
            base_aug=base_aug,
            base_model=base_model,
            base_train=base_train,
            output_dir=root_dir,
            batch_size_choices=args.batch_size_choices,
            architecture=args.architecture,
            force_augmentation=args.force_augmentation,
            force_amp=args.force_amp,
        )
        use_cv = args.n_folds > 1
        try:
            if use_cv:
                cv_results = run_cross_validation(
                    data_config=data_cfg,
                    augmentation_config=aug_cfg,
                    model_config=model_cfg,
                    train_config=train_cfg,
                    n_folds=args.n_folds,
                    quiet=False,
                )
            else:
                results = run_training(
                    data_config=data_cfg,
                    augmentation_config=aug_cfg,
                    model_config=model_cfg,
                    train_config=train_cfg,
                    quiet=False,
                    skip_test=True,
                )
        except Exception as exc:
            print(f"  Trial {trial.number} failed: {exc}")
            release_gpu_memory()
            raise optuna.TrialPruned() from exc

        if use_cv:
            primary = train_cfg.primary_metric
            fold_scores = [
                fr["best_val_metrics"].get(primary)
                for fr in cv_results["fold_results"]
            ]
            fold_scores = [
                float(s) for s in fold_scores
                if s is not None and not (isinstance(s, float) and math.isnan(s))
            ]
            if not fold_scores:
                # All folds returned NaN for the primary metric — treat as failure.
                release_gpu_memory()
                raise optuna.TrialPruned("All folds returned NaN for primary metric.")
            cv_mean = float(sum(fold_scores) / len(fold_scores))
            cv_std = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0
            penalty = float(getattr(train_cfg, "cv_score_std_penalty", 0.0) or 0.0)
            # score = mean − penalty * std penalizes trials whose folds disagree.
            # On the run where trial_15 won, fold_03 collapsed (threshold≈0.05,
            # MCC=0); a non-zero penalty would have pushed the picker toward a
            # trial whose folds agree, even at slightly lower mean PR-AUC.
            score = cv_mean - penalty * cv_std
            agg = cv_results.get("aggregated_val_metrics", {})
            output_dir_str = str(train_cfg.output_dir)
            trial.set_user_attr("cv_fold_scores", fold_scores)
            trial.set_user_attr("cv_score_mean", cv_mean)
            trial.set_user_attr("cv_score_std", cv_std)
            trial.set_user_attr("cv_score_std_penalty", penalty)
            trial.set_user_attr("aggregated_val_metrics", agg)
            trial.set_user_attr("output_dir", output_dir_str)

            # Use the per-metric mean across folds as the "best_val_metrics"
            # surrogate so the leaderboard schema stays compatible.
            mean_val_metrics = {
                key: stats["mean"] for key, stats in agg.items()
                if isinstance(stats, dict) and "mean" in stats
            }
            best_epochs = [fr.get("best_epoch") for fr in cv_results["fold_results"]]
            trial_summary = {
                "trial_number": trial.number,
                "primary_metric": primary,
                "params": trial.params,
                "score": _nan_safe(score),
                "best_epoch": best_epochs,
                "n_folds": args.n_folds,
                "cv_fold_scores": fold_scores,
                "cv_score_mean": _nan_safe(cv_mean),
                "cv_score_std": _nan_safe(cv_std),
                "cv_score_std_penalty": penalty,
                "best_val_metrics": {k: _nan_safe(v) for k, v in mean_val_metrics.items()},
                "fold_results": cv_results["fold_results"],
            }
            save_json(trial_summary, Path(output_dir_str) / "trial_summary.json")
        else:
            trial.set_user_attr("best_val_metrics", results["best_val_metrics"])
            trial.set_user_attr("output_dir", results["output_dir"])

            score = float(results["best_score"])

            trial_summary = {
                "trial_number": trial.number,
                "primary_metric": train_cfg.primary_metric,
                "params": trial.params,
                "score": _nan_safe(score),
                "best_epoch": results["best_epoch"],
                "best_val_metrics": {k: _nan_safe(v) for k, v in results["best_val_metrics"].items()},
            }
            save_json(trial_summary, Path(results["output_dir"]) / "trial_summary.json")

        # Append to leaderboard
        leaderboard.append(trial_summary)
        leaderboard_path.write_text(json.dumps(make_json_safe(leaderboard), indent=2))
        print(f"  Trial {trial.number:03d} score = {score:.4f}")
        release_gpu_memory()
        return score

    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_trials = max(0, search_config.n_trials - completed_trials)
    if completed_trials > 0:
        print(f"  Resuming study: {completed_trials} trials already completed, {remaining_trials} remaining.")

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=search_config.timeout_seconds,
        )

    # ── Final re-train with best trial's params ──────────────────────
    best_trial = study.best_trial
    print(f"\n{'='*60}")
    print(f"  BEST TRIAL: {best_trial.number:03d}  (score={best_trial.value:.4f})")
    print(f"  Params: {json.dumps(best_trial.params, indent=4)}")
    final_epoch_budget = final_epochs if final_epochs is not None else best_trial.params.get("epochs", args.epochs)
    print(f"  Re-training with {final_epoch_budget} epochs into best_run/ ...")
    print(f"{'='*60}\n")

    best_run_dir = root_dir / "best_run"
    data_cfg, aug_cfg, model_cfg, train_cfg = _configs_from_params(
        params=best_trial.params,
        base_data=base_data,
        base_aug=base_aug,
        base_model=base_model,
        base_train=base_train,
        output_dir=best_run_dir,
        epochs_override=final_epochs,
        architecture_override=args.architecture,
        patience_override=args.final_patience,
    )
    final_results = run_training(
        data_config=data_cfg,
        augmentation_config=aug_cfg,
        model_config=model_cfg,
        train_config=train_cfg,
        quiet=False,
    )

    # Copy best model from best_run to root for convenience
    best_model_src = best_run_dir / "best_model.pt"
    if best_model_src.exists():
        shutil.copy2(best_model_src, root_dir / "best_model.pt")

    # ── Automatic final evaluation on the freshly retrained best_run ──
    # Prefer the saved test_predictions.json so we don't repeat inference; fall
    # back to fresh inference if the file is missing. Failures here must NOT
    # destroy training artifacts — we capture the error in study_summary.
    final_eval_dir = root_dir / "final_evaluation"
    final_eval_summary: dict[str, object] = {"output_dir": str(final_eval_dir)}
    try:
        from evaluate_final import run_final_evaluation

        saved_predictions_available = (best_run_dir / "test_predictions.json").exists()
        print(f"\n  Running final evaluation on {best_run_dir} "
              f"(use_saved_predictions={saved_predictions_available}) ...")
        eval_result = run_final_evaluation(
            run_dir=best_run_dir,
            output_dir=final_eval_dir,
            use_saved_predictions=saved_predictions_available,
        )
        tuned_metrics = eval_result.get("metrics_tuned_threshold", {}) or {}
        fixed_metrics = eval_result.get("metrics_fixed_threshold", {}) or {}
        final_eval_summary.update({
            "output_dir": eval_result.get("output_dir", str(final_eval_dir)),
            "source": eval_result.get("source"),
            "tuned_threshold": eval_result.get("tuned_threshold"),
            "fixed_threshold": eval_result.get("fixed_threshold"),
            "metrics_tuned_threshold": {k: _nan_safe(v) for k, v in tuned_metrics.items() if not isinstance(v, dict)},
            "metrics_fixed_threshold": {k: _nan_safe(v) for k, v in fixed_metrics.items() if not isinstance(v, dict)},
        })
    except Exception as exc:
        message = f"final_evaluation_failed: {type(exc).__name__}: {exc}"
        print(f"  Warning: {message}")
        final_eval_summary["error"] = message

    # Study-level summary
    study_summary = {
        "study_name": search_config.study_name,
        "primary_metric": args.primary_metric,
        "n_trials_completed": len(study.trials),
        "best_trial_number": best_trial.number,
        "best_value": _nan_safe(best_trial.value),
        "best_params": best_trial.params,
        "search_config": to_serializable(search_config),
        "final_results": {
            "output_dir": final_results["output_dir"],
            "best_epoch": final_results["best_epoch"],
            "best_score": _nan_safe(final_results["best_score"]),
            "best_val_metrics": {k: _nan_safe(v) for k, v in final_results["best_val_metrics"].items()},
            "test_metrics": {k: _nan_safe(v) for k, v in final_results["test_metrics"].items()},
        },
        "final_evaluation": final_eval_summary,
    }
    save_json(study_summary, root_dir / "study_summary.json")
    print(json.dumps(study_summary, indent=2))


if __name__ == "__main__":
    main()
