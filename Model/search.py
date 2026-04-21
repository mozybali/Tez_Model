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

import optuna

from Model.engine import run_training, save_json, make_json_safe
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
        "--epochs",
        type=int,
        default=30,
        help="Maximum per-trial epoch budget; Optuna samples shorter budgets up to this value.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
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
    parser.add_argument("--final-epochs", type=int, default=None,
                        help="Epochs for the best_run final retrain (defaults to the best trial's epoch budget).")
    parser.add_argument("--primary-metric",
                        choices=["roc_auc", "pr_auc", "balanced_accuracy", "f1"],
                        default="roc_auc",
                        help="Metric to maximize (fallback chain: pr_auc → balanced_accuracy → f1).")
    parser.add_argument("--nan-strategy",
                        choices=["none", "drop_record", "fill_zero", "fill_mean", "fill_median", "fill_constant"],
                        default="none")
    parser.add_argument("--nan-fill-value", type=float, default=0.0)
    return parser.parse_args()


def _sample_trial_configs(
    trial: optuna.Trial,
    base_data: DataConfig,
    base_aug: AugmentationConfig,
    base_model: ModelConfig,
    base_train: TrainConfig,
    output_dir: Path,
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

    augmentations_enabled = trial.suggest_categorical("augmentations_enabled", [True, False])
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

    architecture = "resnet3d"
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
            "pointnet_num_points", [512, 1024, 2048, 4096]
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
        pointnet_point_features = trial.suggest_categorical(
            "pointnet_point_features", [3, 4]
        )
        pointnet_use_input_transform = trial.suggest_categorical(
            "pointnet_use_input_transform", [False, True]
        )
        pointnet_binary_threshold = trial.suggest_categorical(
            "pointnet_binary_threshold", [0.3, 0.5, 0.7]
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
    focal_gamma = (
        trial.suggest_float("focal_gamma", 1.0, 3.0)
        if loss_type == "focal"
        else base_train.focal_gamma
    )
    threshold_selection = trial.suggest_categorical(
        "threshold_selection", ["youden", "f1", "fbeta"],
    )
    threshold_fbeta = (
        trial.suggest_float("threshold_fbeta", 1.0, 2.0)
        if threshold_selection == "fbeta"
        else base_train.threshold_fbeta
    )
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
        pos_weight_strategy=trial.suggest_categorical(
            "pos_weight_strategy", ["ratio", "sqrt", "log", "inverse", "effective", "none"]
        ),
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        batch_size=trial.suggest_categorical("batch_size", [8, 16, 24, 32]),
        early_stopping_patience=trial.suggest_categorical(
            "early_stopping_patience",
            _patience_choices(base_train.epochs),
        ),
        # Tighter clip bounds — 5.0 was too loose and allowed the loss spike at epoch 2.
        gradient_clip_norm=trial.suggest_categorical("gradient_clip_norm", [0.25, 0.5, 1.0, 2.0]),
        use_weighted_sampler=trial.suggest_categorical("use_weighted_sampler", [False, True]),
        amp=trial.suggest_categorical("amp", [True, False]),
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
    architecture = params.get("architecture", "resnet3d")
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
        pos_weight_strategy=params.get("pos_weight_strategy", base_train.pos_weight_strategy),
        loss_type=params.get("loss_type", base_train.loss_type),
        focal_gamma=params.get("focal_gamma", base_train.focal_gamma),
        batch_size=params["batch_size"],
        early_stopping_patience=params.get("early_stopping_patience", base_train.early_stopping_patience),
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
        )
        try:
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
            raise optuna.TrialPruned() from exc

        trial.set_user_attr("best_val_metrics", results["best_val_metrics"])
        trial.set_user_attr("output_dir", results["output_dir"])

        score = float(results["best_score"])

        # Save per-trial summary into the trial folder
        trial_summary = {
            "trial_number": trial.number,
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

    # Study-level summary
    study_summary = {
        "study_name": search_config.study_name,
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
    }
    save_json(study_summary, root_dir / "study_summary.json")
    print(json.dumps(study_summary, indent=2))


if __name__ == "__main__":
    main()
