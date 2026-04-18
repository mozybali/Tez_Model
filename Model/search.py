from __future__ import annotations

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
    max_epochs = max(1, int(max_epochs))
    candidates = {2, 3, 5, 6, 8, 10, max(1, int(round(max_epochs * 0.5)))}
    return sorted(choice for choice in candidates if choice <= max_epochs) or [max_epochs]


def _flip_axes_from_choice(choice: str | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if choice is None:
        return fallback
    return _FLIP_AXIS_CHOICES[choice]


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
        default=12,
        help="Maximum per-trial epoch budget; Optuna samples shorter budgets up to this value.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
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
            flip_axes=_flip_axes_from_choice(flip_axes_choice, base_aug.flip_axes),
            affine_probability=trial.suggest_float("affine_probability", 0.3, 0.8),
            rotation_degrees=trial.suggest_float("rotation_degrees", 4.0, 15.0),
            translation_fraction=trial.suggest_float("translation_fraction", 0.02, 0.10),
            scale_min=trial.suggest_float("scale_min", 0.85, 0.98),
            scale_max=trial.suggest_float("scale_max", 1.02, 1.15),
            morphology_probability=trial.suggest_float("morphology_probability", 0.0, 0.2),
        )
    else:
        aug_config = replace(base_aug, enabled=False)

    model_config = replace(
        base_model,
        depth=trial.suggest_categorical("depth", [18, 34]),
        base_channels=trial.suggest_categorical("base_channels", [16, 24, 32]),
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        use_tabular_features=use_tabular_features,
        tabular_hidden_dim=(
            trial.suggest_categorical("tabular_hidden_dim", [8, 16, 32])
            if use_tabular_features
            else base_model.tabular_hidden_dim
        ),
    )

    optimizer_name = trial.suggest_categorical("optimizer_name", ["adam", "adamw", "sgd"])
    sgd_momentum = (
        trial.suggest_float("sgd_momentum", 0.0, 0.95)
        if optimizer_name == "sgd"
        else base_train.sgd_momentum
    )
    scheduler_name = trial.suggest_categorical("scheduler_name", ["cosine", "none"])
    warmup_epochs = (
        trial.suggest_int("warmup_epochs", 0, max(1, train_epochs // 4))
        if scheduler_name != "none"
        else 0
    )
    loss_type = trial.suggest_categorical("loss_type", ["bce", "focal"])
    focal_gamma = (
        trial.suggest_float("focal_gamma", 1.0, 3.0)
        if loss_type == "focal"
        else base_train.focal_gamma
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
        batch_size=trial.suggest_categorical("batch_size", [4, 8, 16]),
        early_stopping_patience=trial.suggest_categorical(
            "early_stopping_patience",
            _patience_choices(base_train.epochs),
        ),
        gradient_clip_norm=trial.suggest_categorical("gradient_clip_norm", [0.5, 1.0, 2.0, 5.0]),
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
            flip_axes=_flip_axes_from_choice(params.get("flip_axes"), base_aug.flip_axes),
            affine_probability=params["affine_probability"],
            rotation_degrees=params["rotation_degrees"],
            translation_fraction=params["translation_fraction"],
            scale_min=params["scale_min"],
            scale_max=params["scale_max"],
            morphology_probability=params["morphology_probability"],
        )
    else:
        aug_config = replace(base_aug, enabled=False)
    model_config = replace(
        base_model,
        depth=params["depth"],
        base_channels=params["base_channels"],
        dropout=params["dropout"],
        use_tabular_features=params.get("use_tabular_features", base_model.use_tabular_features),
        tabular_hidden_dim=params.get("tabular_hidden_dim", base_model.tabular_hidden_dim),
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
