from __future__ import annotations

import argparse
import json
from pathlib import Path

from Model.engine import run_training, run_cross_validation, make_json_safe
from Utils.config import AugmentationConfig, DataConfig, ModelConfig, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet3D model on the ALAN kidney dataset.")
    parser.add_argument("--info-csv", type=Path, default=Path("ALAN/info.csv"))
    parser.add_argument("--volumes-dir", type=Path, default=Path("ALAN/alan"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("ALAN/metadata.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("ALAN/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--sgd-momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--depth", choices=[18, 34], type=int, default=18)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--disable-tabular-features", action="store_true")
    parser.add_argument("--tabular-hidden-dim", type=int, default=16)
    parser.add_argument("--target-shape", nargs=3, type=int, default=(64, 64, 64))
    parser.add_argument("--bbox-margin", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--norm-type", choices=["batch", "group"], default="batch",
                        help="batch: BatchNorm3D. group: GroupNorm (better for small batch_size).")
    parser.add_argument("--group-norm-groups", type=int, default=8)
    parser.add_argument("--use-weighted-sampler", action="store_true",
                        help="Use inverse-frequency WeightedRandomSampler for training.")
    parser.add_argument("--disable-calibration", action="store_true",
                        help="Disable post-hoc temperature scaling.")
    parser.add_argument("--threshold-selection", choices=["youden", "f1", "fbeta", "fixed"], default="youden",
                        help="How to pick the decision threshold on the validation set.")
    parser.add_argument("--threshold-fbeta", type=float, default=1.0,
                        help="Beta for fbeta threshold selection (>1 weights recall).")
    parser.add_argument("--calibration-method",
                        choices=["temperature", "isotonic", "temperature+isotonic"],
                        default="temperature",
                        help="Post-hoc probability calibration strategy.")
    parser.add_argument("--tta", action="store_true",
                        help="Enable flip-based test-time augmentation during evaluation.")
    parser.add_argument("--disable-augmentations", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-bbox-crop", action="store_true")
    parser.add_argument("--disable-pad-to-cube", action="store_true")
    parser.add_argument("--canonicalize-right", action="store_true")
    parser.add_argument("--right-flip-axis", type=int, default=0)
    parser.add_argument("--rotation-degrees", type=float, default=10.0)
    parser.add_argument("--translation-fraction", type=float, default=0.05)
    parser.add_argument("--scale-min", type=float, default=0.9)
    parser.add_argument("--scale-max", type=float, default=1.1)
    parser.add_argument("--flip-probability", type=float, default=0.5)
    parser.add_argument("--flip-axes", nargs="+", type=int, default=(1, 2))
    parser.add_argument("--affine-probability", type=float, default=0.6)
    parser.add_argument("--morphology-probability", type=float, default=0.1)
    parser.add_argument("--primary-metric",
                        choices=["roc_auc", "pr_auc", "balanced_accuracy", "f1"],
                        default="pr_auc",
                        help="Metric to maximize (fallback chain: pr_auc → balanced_accuracy → f1).")
    parser.add_argument("--nan-strategy",
                        choices=["none", "drop_record", "fill_zero", "fill_mean", "fill_median", "fill_constant"],
                        default="none")
    parser.add_argument("--nan-fill-value", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Number of linear warmup epochs before the main scheduler kicks in.")
    parser.add_argument("--pos-weight-strategy",
                        choices=["ratio", "sqrt", "log", "inverse", "effective", "none"],
                        default="inverse",
                        help="Strategy for computing positive class weight.")
    parser.add_argument("--loss-type", choices=["bce", "focal"], default="bce",
                        help="Loss function: bce (weighted BCE) or focal (sigmoid focal loss).")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focusing parameter for focal loss (ignored when --loss-type=bce).")
    parser.add_argument("--cv-folds", type=int, default=0,
                        help="If >1, run k-fold cross-validation instead of a single train/val/test split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_config = DataConfig(
        info_csv=args.info_csv,
        volumes_dir=args.volumes_dir,
        metadata_csv=args.metadata_csv,
        summary_json=args.summary_json,
        target_shape=tuple(args.target_shape),
        use_bbox_crop=not args.disable_bbox_crop,
        bbox_margin=args.bbox_margin,
        pad_to_cube=not args.disable_pad_to_cube,
        canonicalize_right=args.canonicalize_right,
        right_flip_axis=args.right_flip_axis,
        nan_strategy=args.nan_strategy,
        nan_fill_value=args.nan_fill_value,
    )
    augmentation_config = AugmentationConfig(
        enabled=not args.disable_augmentations,
        flip_probability=args.flip_probability,
        flip_axes=tuple(args.flip_axes),
        affine_probability=args.affine_probability,
        rotation_degrees=args.rotation_degrees,
        translation_fraction=args.translation_fraction,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        morphology_probability=args.morphology_probability,
    )
    model_config = ModelConfig(
        depth=args.depth,
        base_channels=args.base_channels,
        dropout=args.dropout,
        use_tabular_features=not args.disable_tabular_features,
        tabular_hidden_dim=args.tabular_hidden_dim,
        norm_type=args.norm_type,
        group_norm_groups=args.group_norm_groups,
    )
    train_config = TrainConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        sgd_momentum=args.sgd_momentum,
        scheduler_name=args.scheduler,
        device=args.device,
        seed=args.seed,
        early_stopping_patience=args.patience,
        gradient_clip_norm=args.gradient_clip_norm,
        decision_threshold=args.decision_threshold,
        primary_metric=args.primary_metric,
        amp=not args.disable_amp,
        warmup_epochs=args.warmup_epochs,
        pos_weight_strategy=args.pos_weight_strategy,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        cv_folds=args.cv_folds,
        use_weighted_sampler=args.use_weighted_sampler,
        calibrate_temperature=not args.disable_calibration,
        threshold_selection=args.threshold_selection,
        threshold_fbeta=args.threshold_fbeta,
        calibration_method=args.calibration_method,
        tta_enabled=args.tta,
    )

    if train_config.cv_folds > 1:
        results = run_cross_validation(
            data_config=data_config,
            augmentation_config=augmentation_config,
            model_config=model_config,
            train_config=train_config,
            n_folds=train_config.cv_folds,
        )
    else:
        results = run_training(
            data_config=data_config,
            augmentation_config=augmentation_config,
            model_config=model_config,
            train_config=train_config,
        )
    print(json.dumps(make_json_safe(results), indent=2))


if __name__ == "__main__":
    main()
