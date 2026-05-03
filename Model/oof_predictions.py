from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from Model.engine import (
    TABULAR_FEATURE_NAMES,
    collect_predictions,
    compute_tabular_feature_stats,
    effective_pin_memory,
    release_gpu_memory,
    resolve_device,
    save_json,
)
from Model.factory import build_model
from Model.search import _configs_from_params
from Preprocessing.dataset import AlanKidneyDataset, AlanRecord, load_records, split_records
from Utils.calibration import (
    IsotonicResult,
    ThresholdBootstrapResult,
    apply_isotonic,
    apply_temperature,
    expected_calibration_error,
    fit_isotonic,
    fit_temperature,
    logits_from_probs,
    select_threshold_bootstrap,
)
from Utils.config import AugmentationConfig, DataConfig, ModelConfig, TrainConfig, to_serializable
from Utils.metrics import bootstrap_confidence_intervals, compute_binary_classification_metrics


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def trial_dir_for(study_dir: Path, trial_number: int) -> Path:
    return Path(study_dir) / f"trial_{int(trial_number):03d}"


def _study_seed(study_dir: Path) -> int:
    summary_path = Path(study_dir) / "study_summary.json"
    if not summary_path.exists():
        return 42
    try:
        summary = _load_json(summary_path)
        return int(summary.get("search_config", {}).get("sampler_seed", 42))
    except Exception:
        return 42


def reconstruct_trial_configs(
    study_dir: Path,
    trial_number: int,
    device_name: str = "auto",
    num_workers: int | None = None,
    batch_size: int | None = None,
) -> tuple[DataConfig, AugmentationConfig, ModelConfig, TrainConfig, dict[str, Any]]:
    """Rebuild a finished Optuna CV trial from trial_summary.json params."""
    study_dir = Path(study_dir)
    trial_dir = trial_dir_for(study_dir, trial_number)
    summary_path = trial_dir / "trial_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing trial summary: {summary_path}")

    trial_summary = _load_json(summary_path)
    params = trial_summary.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{summary_path} does not contain a params object.")

    base_train = TrainConfig(
        output_dir=trial_dir,
        seed=_study_seed(study_dir),
        device=device_name,
        primary_metric=str(trial_summary.get("primary_metric", "roc_auc")),
        cv_score_std_penalty=float(trial_summary.get("cv_score_std_penalty", 0.0) or 0.0),
    )
    data_config, aug_config, model_config, train_config = _configs_from_params(
        params=params,
        base_data=DataConfig(),
        base_aug=AugmentationConfig(),
        base_model=ModelConfig(),
        base_train=base_train,
        output_dir=trial_dir,
    )
    train_config = replace(train_config, output_dir=trial_dir, device=device_name)
    if num_workers is not None:
        train_config = replace(train_config, num_workers=int(num_workers))
    if batch_size is not None:
        train_config = replace(train_config, batch_size=int(batch_size))
    return data_config.resolved(), aug_config, model_config, train_config, trial_summary


def _common_dataset_kwargs(data_config: DataConfig) -> dict[str, Any]:
    return {
        "target_shape": data_config.target_shape,
        "use_bbox_crop": data_config.use_bbox_crop,
        "bbox_margin": data_config.bbox_margin,
        "pad_to_cube_input": data_config.pad_to_cube,
        "canonicalize_right": data_config.canonicalize_right,
        "right_flip_axis": data_config.right_flip_axis,
        "nan_strategy": data_config.nan_strategy if data_config.nan_strategy != "drop_record" else "none",
        "nan_fill_value": data_config.nan_fill_value,
        "cache_mode": data_config.cache_mode,
        "cache_dir": data_config.cache_dir,
    }


def build_inference_dataloader(
    records: list[AlanRecord],
    data_config: DataConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> DataLoader:
    dataset = AlanKidneyDataset(records=records, transform=None, **_common_dataset_kwargs(data_config))
    persistent = train_config.num_workers > 0
    return DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=effective_pin_memory(train_config.pin_memory, device),
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )


def _load_records_for_config(data_config: DataConfig) -> dict[str, list[AlanRecord]]:
    records = load_records(
        info_csv=data_config.info_csv,
        volumes_dir=data_config.volumes_dir,
        metadata_csv=data_config.metadata_csv,
        summary_json=data_config.summary_json,
    )
    return split_records(
        records,
        train_subset=data_config.train_subset,
        val_subset=data_config.val_subset,
        test_subset=data_config.test_subset,
    )


def reconstruct_cv_folds(
    data_config: DataConfig,
    train_config: TrainConfig,
    n_folds: int,
) -> tuple[list[AlanRecord], list[tuple[list[AlanRecord], list[AlanRecord]]]]:
    splits = _load_records_for_config(data_config)
    cv_records = splits["train"] + splits["val"]
    labels = [int(record.label) for record in cv_records]
    groups = [record.roi_id.rsplit("_", 1)[0] for record in cv_records]

    splitter = StratifiedGroupKFold(
        n_splits=int(n_folds),
        shuffle=True,
        random_state=train_config.seed,
    )
    folds: list[tuple[list[AlanRecord], list[AlanRecord]]] = []
    for train_indices, val_indices in splitter.split(cv_records, labels, groups):
        fold_train = [cv_records[int(index)] for index in train_indices]
        fold_val = [cv_records[int(index)] for index in val_indices]
        if data_config.nan_strategy == "drop_record":
            fold_train = [record for record in fold_train if not record.has_nan]
            fold_val = [record for record in fold_val if not record.has_nan]
            if not fold_train:
                raise ValueError("A CV fold train split was emptied by nan_strategy='drop_record'.")
            if not fold_val:
                raise ValueError("A CV fold val split was emptied by nan_strategy='drop_record'.")
        folds.append((fold_train, fold_val))
    return cv_records, folds


def reconstruct_test_records(data_config: DataConfig) -> list[AlanRecord]:
    splits = _load_records_for_config(data_config)
    test_records = list(splits["test"])
    if data_config.nan_strategy == "drop_record":
        test_records = [record for record in test_records if not record.has_nan]
    if not test_records:
        raise ValueError("The reconstructed test split is empty.")
    return test_records


def _load_fold_model(
    checkpoint_path: Path,
    model_config: ModelConfig,
    device: torch.device,
) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing fold checkpoint: {checkpoint_path}")
    model = build_model(
        model_config=model_config,
        num_tabular_features=len(TABULAR_FEATURE_NAMES) if model_config.use_tabular_features else 0,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _ids_from_predictions(preds: dict[str, Any], dataset: AlanKidneyDataset) -> list[str]:
    raw_ids = preds.get("id")
    if raw_ids is None:
        return [str(record.roi_id) for record in dataset.records]
    if isinstance(raw_ids, str):
        return [raw_ids]
    return [str(value) for value in list(raw_ids)]


def _validate_ids(ids: list[str], expected_count: int, context: str) -> None:
    if len(ids) != expected_count:
        raise ValueError(f"{context}: got {len(ids)} ids for {expected_count} predictions.")
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"{context}: duplicate ids detected: {preview}")


def _validate_prediction_lengths(ids: list[str], preds: dict[str, Any], context: str) -> None:
    lengths = {
        "id": len(ids),
        "y_true": len(preds.get("y_true", [])),
        "y_prob": len(preds.get("y_prob", [])),
        "y_logit": len(preds.get("y_logit", [])),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{context}: prediction lengths disagree: {lengths}")


def _validate_label_order(ids: list[str], y_true: list[float], records: list[AlanRecord], context: str) -> None:
    expected = {str(record.roi_id): int(record.label) for record in records}
    for sample_id, label in zip(ids, y_true):
        if int(label) != expected.get(str(sample_id)):
            raise ValueError(f"{context}: label mismatch for id {sample_id}.")


def _threshold_payload(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str,
    beta: float,
    min_specificity: float | None,
    min_precision: float | None,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    result = select_threshold_bootstrap(
        y_true,
        y_prob,
        method=method,
        n_bootstrap=n_bootstrap,
        seed=seed,
        beta=beta,
        min_specificity=min_specificity,
        min_precision=min_precision,
        return_distribution=True,
    )
    if not isinstance(result, ThresholdBootstrapResult):
        raise TypeError("select_threshold_bootstrap did not return ThresholdBootstrapResult.")
    threshold = float(result.threshold)
    metrics = compute_binary_classification_metrics(y_true, y_prob, threshold=threshold)
    try:
        ci = bootstrap_confidence_intervals(
            y_true,
            y_prob,
            threshold=threshold,
            n_bootstrap=min(int(n_bootstrap), 1000),
            seed=seed,
        )
    except Exception as exc:
        ci = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "name": name,
        "selected_threshold": threshold,
        "bootstrap_median_threshold": float(result.median),
        "bootstrap_threshold_ci": {
            "ci_lower": float(result.ci_lower),
            "ci_upper": float(result.ci_upper),
        },
        "valid_bootstrap_samples": int(result.valid_bootstrap_samples),
        "threshold_selection_method": method,
        "beta": float(beta),
        "min_specificity": min_specificity,
        "min_precision": min_precision,
        "seed": int(seed),
        "n_bootstrap": int(n_bootstrap),
        "oof_metrics_at_threshold": metrics,
        "oof_confidence_intervals_at_threshold": ci,
        "bootstrap_threshold_distribution": list(result.thresholds),
    }


def compute_oof_thresholds(
    y_true: np.ndarray | list[float],
    y_prob: np.ndarray | list[float],
    seed: int,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probs = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    return {
        "f1_threshold": _threshold_payload(
            name="f1_threshold",
            y_true=labels,
            y_prob=probs,
            method="f1",
            beta=1.0,
            min_specificity=None,
            min_precision=None,
            seed=seed,
            n_bootstrap=n_bootstrap,
        ),
        "clinical_threshold": _threshold_payload(
            name="clinical_threshold",
            y_true=labels,
            y_prob=probs,
            method="fbeta",
            beta=2.0,
            min_specificity=0.85,
            min_precision=None,
            seed=seed,
            n_bootstrap=n_bootstrap,
        ),
    }


def _isotonic_from_payload(payload: dict[str, Any]) -> IsotonicResult | None:
    iso = payload.get("metadata", {}).get("pooled_isotonic") or payload.get("pooled_isotonic")
    if not iso:
        return None
    x = iso.get("x") or iso.get("isotonic_x")
    y = iso.get("y") or iso.get("isotonic_y")
    if not x or not y:
        return None
    return IsotonicResult(
        x=tuple(float(value) for value in x),
        y=tuple(float(value) for value in y),
        ece_before=float(iso.get("ece_before", float("nan"))),
        ece_after=float(iso.get("ece_after", float("nan"))),
    )


def apply_trial_calibration(
    logits: np.ndarray | list[float],
    temperature: float,
    oof_payload: dict[str, Any],
) -> np.ndarray:
    probs = apply_temperature(logits, float(temperature))
    isotonic = _isotonic_from_payload(oof_payload)
    if isotonic is not None:
        probs = apply_isotonic(probs, isotonic)
    return np.clip(probs, 0.0, 1.0)


def load_oof_predictions(study_dir: Path, trial_number: int) -> dict[str, Any]:
    path = trial_dir_for(study_dir, trial_number) / "oof_predictions.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing OOF predictions: {path}")
    return _load_json(path)


def _canonical_oof_path(study_dir: Path, trial_number: int) -> Path:
    return trial_dir_for(study_dir, trial_number) / "oof_predictions.json"


def _prediction_logits(preds: dict[str, Any]) -> np.ndarray:
    raw_logits = preds.get("y_logit")
    if raw_logits is not None:
        logits = np.asarray(raw_logits, dtype=np.float64).reshape(-1)
        if logits.size > 0:
            return logits
    return logits_from_probs(np.asarray(preds["y_prob"], dtype=np.float64))


def generate_oof_predictions(
    study_dir: Path,
    trial_number: int,
    output_path: Path | None = None,
    force: bool = False,
    reuse_existing: bool = True,
    device_name: str = "auto",
    num_workers: int | None = None,
    batch_size: int | None = None,
    n_bootstrap: int = 1000,
    quiet: bool = False,
) -> dict[str, Any]:
    study_dir = Path(study_dir)
    trial_dir = trial_dir_for(study_dir, trial_number)
    canonical_path = _canonical_oof_path(study_dir, trial_number)
    if canonical_path.exists() and not force:
        if not reuse_existing:
            raise FileExistsError(f"OOF predictions already exist: {canonical_path}")
        payload = _load_json(canonical_path)
        if output_path is not None and Path(output_path) != canonical_path:
            save_json(payload, Path(output_path))
        if not quiet:
            print(f"  Loading existing OOF predictions: {canonical_path}")
        return payload

    data_config, _aug_config, model_config, train_config, trial_summary = reconstruct_trial_configs(
        study_dir=study_dir,
        trial_number=trial_number,
        device_name=device_name,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    device = resolve_device(train_config.device)
    n_folds = int(trial_summary.get("n_folds", 3))
    cv_records, folds = reconstruct_cv_folds(data_config, train_config, n_folds=n_folds)
    expected_cv_records = (
        [record for record in cv_records if not record.has_nan]
        if data_config.nan_strategy == "drop_record"
        else list(cv_records)
    )
    expected_cv_ids = {str(record.roi_id) for record in expected_cv_records}

    samples: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    pooled_ids: set[str] = set()

    for fold_idx, (fold_train, fold_val) in enumerate(folds, start=1):
        checkpoint_path = trial_dir / f"fold_{fold_idx:02d}" / "best_model.pt"
        if not quiet:
            print(f"  OOF trial {int(trial_number):03d} fold {fold_idx:02d}: {len(fold_val)} validation samples")
        tabular_stats = (
            compute_tabular_feature_stats(fold_train)
            if model_config.use_tabular_features
            else None
        )
        dataloader = build_inference_dataloader(fold_val, data_config, train_config, device)
        dataset = dataloader.dataset
        if not isinstance(dataset, AlanKidneyDataset):
            raise TypeError("Expected AlanKidneyDataset for validation inference.")
        model = _load_fold_model(checkpoint_path, model_config, device)
        preds = collect_predictions(
            model=model,
            dataloader=dataloader,
            device=device,
            tabular_feature_stats=tabular_stats,
            tta_enabled=bool(getattr(train_config, "tta_enabled", False)),
        )
        ids = _ids_from_predictions(preds, dataset)
        _validate_prediction_lengths(ids, preds, f"fold {fold_idx:02d}")
        _validate_ids(ids, len(preds["y_prob"]), f"fold {fold_idx:02d}")
        dataset_ids = [str(record.roi_id) for record in dataset.records]
        if ids != dataset_ids:
            raise ValueError(f"fold {fold_idx:02d}: prediction ids differ from dataset record order.")
        _validate_label_order(ids, list(preds["y_true"]), dataset.records, f"fold {fold_idx:02d}")

        overlap = pooled_ids.intersection(ids)
        if overlap:
            preview = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Pooled OOF duplicate ids across folds: {preview}")
        pooled_ids.update(ids)

        logits = _prediction_logits(preds)
        labels = np.asarray(preds["y_true"], dtype=np.float64)
        temp_result = fit_temperature(logits, labels)
        temp_probs = apply_temperature(logits, temp_result.temperature)

        for sample_id, y_true, logit, prob_uncal, prob_temp in zip(
            ids,
            preds["y_true"],
            logits.tolist(),
            preds["y_prob"],
            temp_probs.tolist(),
        ):
            samples.append(
                {
                    "id": str(sample_id),
                    "fold": int(fold_idx),
                    "y_true": int(y_true),
                    "y_logit": float(logit),
                    "y_prob_uncalibrated": float(prob_uncal),
                    "temperature": float(temp_result.temperature),
                    "y_prob_temperature_calibrated": float(prob_temp),
                    "y_prob_calibrated": None,
                }
            )

        fold_metadata.append(
            {
                "fold": int(fold_idx),
                "checkpoint_path": str(checkpoint_path),
                "n_train": int(len(fold_train)),
                "n_val": int(len(fold_val)),
                "tabular_feature_stats": tabular_stats,
                "temperature": float(temp_result.temperature),
                "temperature_summary": {
                    "nll_before": float(temp_result.nll_before),
                    "nll_after": float(temp_result.nll_after),
                    "ece_before": float(temp_result.ece_before),
                    "ece_after": float(temp_result.ece_after),
                },
            }
        )
        del model, dataloader, dataset
        release_gpu_memory()

    if pooled_ids != expected_cv_ids:
        missing = sorted(expected_cv_ids.difference(pooled_ids))
        extra = sorted(pooled_ids.difference(expected_cv_ids))
        raise ValueError(
            "Pooled OOF ids do not cover the reconstructed CV set exactly. "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    y_true = np.asarray([sample["y_true"] for sample in samples], dtype=np.int64)
    probs_uncal = np.asarray([sample["y_prob_uncalibrated"] for sample in samples], dtype=np.float64)
    probs_temp = np.asarray(
        [sample["y_prob_temperature_calibrated"] for sample in samples],
        dtype=np.float64,
    )

    isotonic = fit_isotonic(probs_temp, y_true)
    probs_iso = apply_isotonic(probs_temp, isotonic)
    for sample, prob_cal in zip(samples, probs_iso.tolist()):
        sample["y_prob_calibrated"] = float(prob_cal)

    calibration_metrics = {
        "ece_before_temperature": expected_calibration_error(probs_uncal, y_true),
        "ece_after_temperature": expected_calibration_error(probs_temp, y_true),
        "ece_after_pooled_isotonic": expected_calibration_error(probs_iso, y_true),
    }
    thresholds = compute_oof_thresholds(
        y_true=y_true,
        y_prob=probs_iso,
        seed=train_config.seed,
        n_bootstrap=n_bootstrap,
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "study_dir": str(study_dir),
        "trial_number": int(trial_number),
        "trial_dir": str(trial_dir),
        "source": "CV fold checkpoints only",
        "config": {
            "data": to_serializable(data_config),
            "model": to_serializable(model_config),
            "train": to_serializable(train_config),
            "trial_summary_params": trial_summary.get("params", {}),
        },
        "metadata": {
            "n_folds": int(n_folds),
            "n_oof": int(len(samples)),
            "n_expected_cv": int(len(expected_cv_ids)),
            "id_validation": {
                "fold_ids_unique": True,
                "pooled_ids_unique": True,
                "pooled_ids_cover_reconstructed_cv": True,
            },
            "folds": fold_metadata,
            "fold_specific_tabular_stats": {
                str(item["fold"]): item["tabular_feature_stats"] for item in fold_metadata
            },
            "pooled_isotonic": {
                "x": list(isotonic.x),
                "y": list(isotonic.y),
                "ece_before": float(isotonic.ece_before),
                "ece_after": float(isotonic.ece_after),
            },
            "tta_enabled": bool(getattr(train_config, "tta_enabled", False)),
        },
        "calibration_metrics": calibration_metrics,
        "thresholds": thresholds,
        "predictions": samples,
    }
    save_json(payload, canonical_path)

    convenience_path = study_dir / "oof_predictions.json"
    if convenience_path != canonical_path:
        shutil.copy2(canonical_path, convenience_path)
    if output_path is not None and Path(output_path) not in {canonical_path, convenience_path}:
        save_json(payload, Path(output_path))
    if not quiet:
        print(f"  Saved OOF predictions: {canonical_path}")
        print(f"  Saved convenience copy: {convenience_path}")
    return payload


def load_or_generate_oof_predictions(
    study_dir: Path,
    trial_number: int,
    force: bool = False,
    device_name: str = "auto",
    num_workers: int | None = None,
    batch_size: int | None = None,
    n_bootstrap: int = 1000,
    quiet: bool = False,
) -> dict[str, Any]:
    return generate_oof_predictions(
        study_dir=study_dir,
        trial_number=trial_number,
        force=force,
        reuse_existing=True,
        device_name=device_name,
        num_workers=num_workers,
        batch_size=batch_size,
        n_bootstrap=n_bootstrap,
        quiet=quiet,
    )


def prediction_map_from_oof(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for sample in payload.get("predictions", []):
        sample_id = str(sample["id"])
        if sample_id in result:
            raise ValueError(f"Duplicate OOF id in payload: {sample_id}")
        result[sample_id] = sample
    return result


def predict_trial_test_from_folds(
    study_dir: Path,
    trial_number: int,
    oof_payload: dict[str, Any],
    device_name: str = "auto",
    num_workers: int | None = None,
    batch_size: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    data_config, _aug_config, model_config, train_config, _summary = reconstruct_trial_configs(
        study_dir=study_dir,
        trial_number=trial_number,
        device_name=device_name,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    device = resolve_device(train_config.device)
    test_records = reconstruct_test_records(data_config)
    dataloader = build_inference_dataloader(test_records, data_config, train_config, device)
    dataset = dataloader.dataset
    if not isinstance(dataset, AlanKidneyDataset):
        raise TypeError("Expected AlanKidneyDataset for test inference.")
    test_ids = [str(record.roi_id) for record in dataset.records]
    y_true = [int(record.label) for record in dataset.records]
    _validate_ids(test_ids, len(test_ids), "test split")

    fold_payloads = sorted(
        oof_payload.get("metadata", {}).get("folds", []),
        key=lambda item: int(item["fold"]),
    )
    if not fold_payloads:
        raise ValueError("OOF payload does not contain fold metadata.")

    per_fold_prob_maps: list[dict[str, float]] = []
    fold_outputs: list[dict[str, Any]] = []
    trial_dir = trial_dir_for(Path(study_dir), trial_number)
    for fold_meta in fold_payloads:
        fold_idx = int(fold_meta["fold"])
        checkpoint_path = trial_dir / f"fold_{fold_idx:02d}" / "best_model.pt"
        if not quiet:
            print(f"  Test trial {int(trial_number):03d} fold {fold_idx:02d}: {len(test_ids)} samples")
        tabular_stats = fold_meta.get("tabular_feature_stats") if model_config.use_tabular_features else None
        model = _load_fold_model(checkpoint_path, model_config, device)
        preds = collect_predictions(
            model=model,
            dataloader=dataloader,
            device=device,
            tabular_feature_stats=tabular_stats,
            tta_enabled=bool(getattr(train_config, "tta_enabled", False)),
        )
        ids = _ids_from_predictions(preds, dataset)
        _validate_prediction_lengths(ids, preds, f"test fold {fold_idx:02d}")
        _validate_ids(ids, len(preds["y_prob"]), f"test fold {fold_idx:02d}")
        if set(ids) != set(test_ids):
            raise ValueError(f"test fold {fold_idx:02d}: prediction ids differ from reconstructed test ids.")
        _validate_label_order(ids, list(preds["y_true"]), dataset.records, f"test fold {fold_idx:02d}")

        logits = _prediction_logits(preds)
        calibrated = apply_trial_calibration(
            logits=logits,
            temperature=float(fold_meta.get("temperature", 1.0)),
            oof_payload=oof_payload,
        )
        fold_map = {str(sample_id): float(prob) for sample_id, prob in zip(ids, calibrated.tolist())}
        per_fold_prob_maps.append(fold_map)
        fold_outputs.append(
            {
                "fold": fold_idx,
                "temperature": float(fold_meta.get("temperature", 1.0)),
                "ids": ids,
                "y_prob_calibrated": [fold_map[sample_id] for sample_id in ids],
            }
        )
        del model
        release_gpu_memory()

    averaged = []
    for sample_id in test_ids:
        averaged.append(float(np.mean([fold_map[sample_id] for fold_map in per_fold_prob_maps])))

    del dataloader, dataset
    release_gpu_memory()
    return {
        "ids": test_ids,
        "y_true": y_true,
        "y_prob": averaged,
        "per_fold": fold_outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate calibrated OOF predictions from CV fold checkpoints.")
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical_path = _canonical_oof_path(args.study_dir, args.trial_number)
    if args.skip_existing and canonical_path.exists() and not args.force:
        print(f"  Skipping existing OOF predictions: {canonical_path}")
        return
    generate_oof_predictions(
        study_dir=args.study_dir,
        trial_number=args.trial_number,
        output_path=args.output_path,
        force=args.force,
        reuse_existing=True,
        device_name=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
