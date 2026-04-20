"""Soft-voting ensemble over the top-K Optuna trials.

Reads ``leaderboard.json`` from an Optuna study directory, picks the K trials
with the highest validation score, reloads each trial's best checkpoint, runs
inference on the test split, averages the (temperature-calibrated) probabilities,
and reports the ensemble's metrics and bootstrap confidence intervals.

Usage:
    python -m Model.ensemble --study-dir outputs/hpo_resnet3d_full --top-k 5

Assumes each trial folder under the study directory contains:
  - ``best_model.pt`` (checkpoint saved by Model/engine.py)
  - ``checkpoint_meta.json`` (config snapshot written alongside the checkpoint)
  - optionally ``calibration.json`` (for the temperature scalar)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from Model.engine import (
    TABULAR_FEATURE_NAMES,
    build_dataloaders,
    collect_predictions,
    resolve_device,
    save_json,
)
from Model.factory import build_model
from Utils.calibration import apply_temperature, logits_from_probs
from Utils.config import AugmentationConfig, DataConfig, ModelConfig, TrainConfig
from Utils.metrics import bootstrap_confidence_intervals, compute_binary_classification_metrics, optimize_threshold


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _reconstruct_configs(meta: dict) -> tuple[DataConfig, AugmentationConfig, ModelConfig, TrainConfig]:
    data_raw = meta.get("data_config", {})
    model_raw = meta.get("model_config", {})
    train_raw = meta.get("train_config", {})

    data_config = DataConfig(
        info_csv=Path(data_raw.get("info_csv", "ALAN/info.csv")),
        volumes_dir=Path(data_raw.get("volumes_dir", "ALAN/alan")),
        metadata_csv=Path(data_raw.get("metadata_csv", "ALAN/metadata.csv")),
        summary_json=Path(data_raw.get("summary_json", "ALAN/summary.json")),
        train_subset=data_raw.get("train_subset", "ZS-train"),
        val_subset=data_raw.get("val_subset", "ZS-dev"),
        test_subset=data_raw.get("test_subset", "ZS-test"),
        target_shape=tuple(data_raw.get("target_shape", (64, 64, 64))),
        use_bbox_crop=data_raw.get("use_bbox_crop", True),
        bbox_margin=data_raw.get("bbox_margin", 8),
        pad_to_cube=data_raw.get("pad_to_cube", True),
        canonicalize_right=data_raw.get("canonicalize_right", False),
        right_flip_axis=data_raw.get("right_flip_axis", 0),
        nan_strategy=data_raw.get("nan_strategy", "none"),
        nan_fill_value=data_raw.get("nan_fill_value", 0.0),
    )
    aug_config = AugmentationConfig(enabled=False)  # inference-only

    model_config = ModelConfig(
        # Older checkpoints predating U-Net integration lack "architecture";
        # default to ResNet3D so those runs keep reloading cleanly.
        architecture=model_raw.get("architecture", "resnet3d"),
        depth=model_raw.get("depth", 18),
        in_channels=model_raw.get("in_channels", 1),
        base_channels=model_raw.get("base_channels", 32),
        dropout=model_raw.get("dropout", 0.0),
        num_classes=model_raw.get("num_classes", 1),
        use_tabular_features=model_raw.get("use_tabular_features", False),
        tabular_hidden_dim=model_raw.get("tabular_hidden_dim", 16),
        norm_type=model_raw.get("norm_type", "batch"),
        group_norm_groups=model_raw.get("group_norm_groups", 8),
        unet_depth=model_raw.get("unet_depth", 4),
        unet_base_channels=model_raw.get("unet_base_channels", 16),
        unet_channel_multiplier=model_raw.get("unet_channel_multiplier", 2),
        unet_bottleneck_channels=model_raw.get("unet_bottleneck_channels", None),
        pointnet_num_points=model_raw.get("pointnet_num_points", 1024),
        pointnet_point_features=model_raw.get("pointnet_point_features", 3),
        pointnet_mlp_channels=tuple(model_raw.get("pointnet_mlp_channels", (64, 128, 256))),
        pointnet_global_dim=model_raw.get("pointnet_global_dim", 512),
        pointnet_head_hidden_dim=model_raw.get("pointnet_head_hidden_dim", 128),
        pointnet_binary_threshold=model_raw.get("pointnet_binary_threshold", 0.5),
        pointnet_use_input_transform=model_raw.get("pointnet_use_input_transform", False),
    )

    train_config = TrainConfig(
        output_dir=Path(train_raw.get("output_dir", "outputs/ensemble")),
        epochs=1,
        batch_size=train_raw.get("batch_size", 8),
        num_workers=train_raw.get("num_workers", 0),
        pin_memory=train_raw.get("pin_memory", True),
        device=train_raw.get("device", "auto"),
        seed=train_raw.get("seed", 42),
        amp=False,
    )
    return data_config, aug_config, model_config, train_config


def _run_trial_inference(trial_dir: Path, device: torch.device) -> dict:
    checkpoint_meta = _load_json(trial_dir / "checkpoint_meta.json")
    data_cfg, aug_cfg, model_cfg, train_cfg = _reconstruct_configs(checkpoint_meta)

    dataloaders, _ = build_dataloaders(
        data_config=data_cfg,
        augmentation_config=aug_cfg,
        train_config=train_cfg,
        device=device,
    )
    tabular_stats = checkpoint_meta.get("tabular_feature_stats")

    model = build_model(
        model_config=model_cfg,
        num_tabular_features=len(TABULAR_FEATURE_NAMES) if model_cfg.use_tabular_features else 0,
    ).to(device)
    checkpoint = torch.load(trial_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_preds = collect_predictions(model, dataloaders["test"], device, tabular_stats)
    val_preds = collect_predictions(model, dataloaders["val"], device, tabular_stats)

    temperature = 1.0
    calibration_file = trial_dir / "calibration.json"
    if calibration_file.exists():
        try:
            temperature = float(_load_json(calibration_file).get("temperature", 1.0))
        except Exception:
            temperature = 1.0

    test_logits = np.asarray(test_preds.get("y_logit") or logits_from_probs(np.asarray(test_preds["y_prob"])))
    val_logits = np.asarray(val_preds.get("y_logit") or logits_from_probs(np.asarray(val_preds["y_prob"])))
    return {
        "val_y_true": val_preds["y_true"],
        "val_y_prob": apply_temperature(val_logits, temperature).tolist(),
        "test_y_true": test_preds["y_true"],
        "test_y_prob": apply_temperature(test_logits, temperature).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-K soft-voting ensemble over Optuna trials.")
    parser.add_argument("--study-dir", type=Path, required=True,
                        help="Optuna study directory (contains leaderboard.json and trial_XXX folders).")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to save ensemble metrics (defaults to <study-dir>/ensemble_topk).")
    args = parser.parse_args()

    study_dir: Path = args.study_dir
    leaderboard = _load_json(study_dir / "leaderboard.json")
    leaderboard = [entry for entry in leaderboard if entry.get("score") is not None]
    leaderboard.sort(key=lambda entry: float(entry["score"]), reverse=True)
    top_k = leaderboard[: max(1, args.top_k)]

    output_dir: Path = args.output_dir or (study_dir / "ensemble_topk")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device("auto")
    print(f"  Ensembling top-{len(top_k)} trials into {output_dir}")

    test_prob_stack: list[np.ndarray] = []
    val_prob_stack: list[np.ndarray] = []
    test_y_true: list[float] | None = None
    val_y_true: list[float] | None = None
    used_trials: list[dict] = []

    for entry in top_k:
        trial_number = int(entry["trial_number"])
        trial_dir = study_dir / f"trial_{trial_number:03d}"
        if not (trial_dir / "best_model.pt").exists():
            print(f"  [skip] trial {trial_number:03d}: checkpoint missing")
            continue
        print(f"  [run]  trial {trial_number:03d}  score={entry['score']:.4f}")
        preds = _run_trial_inference(trial_dir, device)
        test_prob_stack.append(np.asarray(preds["test_y_prob"]))
        val_prob_stack.append(np.asarray(preds["val_y_prob"]))
        test_y_true = preds["test_y_true"]
        val_y_true = preds["val_y_true"]
        used_trials.append({"trial_number": trial_number, "score": entry["score"]})

    if not test_prob_stack or test_y_true is None or val_y_true is None:
        raise RuntimeError("No usable trial checkpoints found in the study directory.")

    ensemble_test_prob = np.mean(np.stack(test_prob_stack, axis=0), axis=0)
    ensemble_val_prob = np.mean(np.stack(val_prob_stack, axis=0), axis=0)

    tuned_threshold = optimize_threshold(val_y_true, ensemble_val_prob.tolist(), method="youden")
    fixed_threshold = 0.5

    test_metrics_tuned = compute_binary_classification_metrics(
        test_y_true, ensemble_test_prob.tolist(), threshold=tuned_threshold,
    )
    test_metrics_fixed = compute_binary_classification_metrics(
        test_y_true, ensemble_test_prob.tolist(), threshold=fixed_threshold,
    )
    test_ci = bootstrap_confidence_intervals(
        test_y_true, ensemble_test_prob.tolist(), threshold=tuned_threshold,
    )

    payload = {
        "trials": used_trials,
        "tuned_threshold": tuned_threshold,
        "fixed_threshold": fixed_threshold,
        "test_metrics_tuned_threshold": test_metrics_tuned,
        "test_metrics_fixed_threshold": test_metrics_fixed,
        "test_confidence_intervals": test_ci,
    }
    save_json(payload, output_dir / "ensemble_metrics.json")
    save_json(
        {
            "y_true": list(test_y_true),
            "y_prob_ensemble": ensemble_test_prob.tolist(),
            "val_y_true": list(val_y_true),
            "val_y_prob_ensemble": ensemble_val_prob.tolist(),
            "per_trial_test_probs": [p.tolist() for p in test_prob_stack],
        },
        output_dir / "ensemble_predictions.json",
    )
    print(f"  Ensemble tuned_thresh={tuned_threshold:.4f}  "
          f"ROC-AUC={test_metrics_tuned.get('roc_auc', float('nan')):.4f}  "
          f"PR-AUC={test_metrics_tuned.get('pr_auc', float('nan')):.4f}  "
          f"F1={test_metrics_tuned.get('f1', float('nan')):.4f}")


if __name__ == "__main__":
    main()
