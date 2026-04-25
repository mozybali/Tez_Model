from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from Model.factory import build_model
from Preprocessing.dataset import AlanKidneyDataset, infer_positive_class_weight, load_records, split_records
from Preprocessing.transforms import build_train_augmentations
from Utils.calibration import (
    apply_isotonic,
    apply_temperature,
    expected_calibration_error,
    fit_isotonic,
    fit_temperature,
    logits_from_probs,
    reliability_bins,
    select_threshold_bootstrap,
)
from Utils.config import AugmentationConfig, DataConfig, ModelConfig, TrainConfig, to_serializable
from Utils.metrics import bootstrap_confidence_intervals, compute_binary_classification_metrics, optimize_threshold, select_model_score
from Utils.plot_metrics import generate_plots
from Utils.reproducibility import seed_everything


TABULAR_FEATURE_NAMES = ("log_voxel_count_z", "side_is_left")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def make_json_safe(obj):
    """Replace NaN/Inf with None so json.dumps never fails."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return obj


def save_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(payload), indent=2))


def compute_tabular_feature_stats(records: list) -> dict[str, float]:
    if not records:
        raise ValueError("Cannot compute tabular feature stats from an empty training split.")
    log_voxels = torch.tensor(
        [math.log1p(max(float(record.voxel_count), 0.0)) for record in records],
        dtype=torch.float32,
    )
    std = float(log_voxels.std(unbiased=False).item())
    if std < 1e-8:
        std = 1.0
    return {
        "log_voxel_count_mean": float(log_voxels.mean().item()),
        "log_voxel_count_std": std,
    }


def build_tabular_features(
    batch: dict[str, object],
    stats: dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    voxel_count = batch["voxel_count"]
    if isinstance(voxel_count, torch.Tensor):
        voxel_tensor = voxel_count.to(device=device, dtype=torch.float32)
    else:
        voxel_tensor = torch.tensor(voxel_count, dtype=torch.float32, device=device)
    log_voxel_z = (
        torch.log1p(torch.clamp(voxel_tensor, min=0.0)) - stats["log_voxel_count_mean"]
    ) / stats["log_voxel_count_std"]

    sides = batch["side"]
    if isinstance(sides, str):
        side_values = [sides]
    else:
        side_values = list(sides)
    side_is_left = torch.tensor(
        [1.0 if str(side).upper() == "L" else 0.0 for side in side_values],
        dtype=torch.float32,
        device=device,
    )
    return torch.stack([log_voxel_z, side_is_left], dim=1)


def effective_pin_memory(pin_memory: bool, device: torch.device) -> bool:
    return pin_memory and device.type == "cuda"


def _build_weighted_sampler(labels: list[int], seed: int) -> WeightedRandomSampler:
    """Inverse-frequency WeightedRandomSampler so each minibatch sees balanced classes."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.size == 0:
        raise ValueError("Cannot build weighted sampler from an empty label list.")
    counts = np.bincount(labels_arr, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero for single-class splits
    per_class_weight = 1.0 / counts
    sample_weights = per_class_weight[labels_arr]
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels_arr),
        replacement=True,
        generator=generator,
    )


def build_dataloaders(
    data_config: DataConfig,
    augmentation_config: AugmentationConfig,
    train_config: TrainConfig,
    device: torch.device | None = None,
) -> tuple[dict[str, DataLoader], dict[str, list]]:
    if device is None:
        device = resolve_device(train_config.device)

    records = load_records(
        info_csv=data_config.info_csv,
        volumes_dir=data_config.volumes_dir,
        metadata_csv=data_config.metadata_csv,
        summary_json=data_config.summary_json,
    )
    splits = split_records(
        records,
        train_subset=data_config.train_subset,
        val_subset=data_config.val_subset,
        test_subset=data_config.test_subset,
    )

    if data_config.nan_strategy == "drop_record":
        for split_name in list(splits):
            before = len(splits[split_name])
            splits[split_name] = [r for r in splits[split_name] if not r.has_nan]
            dropped = before - len(splits[split_name])
            if dropped:
                print(f"  [NaN drop_record] {split_name}: dropped {dropped}/{before} records")
            if not splits[split_name]:
                raise ValueError(
                    f"All records in '{split_name}' split were dropped by nan_strategy='drop_record'."
                )

    train_transform = build_train_augmentations(augmentation_config)
    common_kwargs = dict(
        target_shape=data_config.target_shape,
        use_bbox_crop=data_config.use_bbox_crop,
        bbox_margin=data_config.bbox_margin,
        pad_to_cube_input=data_config.pad_to_cube,
        canonicalize_right=data_config.canonicalize_right,
        right_flip_axis=data_config.right_flip_axis,
        nan_strategy=data_config.nan_strategy if data_config.nan_strategy != "drop_record" else "none",
        nan_fill_value=data_config.nan_fill_value,
    )
    datasets = {
        "train": AlanKidneyDataset(records=splits["train"], transform=train_transform, **common_kwargs),
        "val": AlanKidneyDataset(records=splits["val"], transform=None, **common_kwargs),
        "test": AlanKidneyDataset(records=splits["test"], transform=None, **common_kwargs),
    }

    generator = torch.Generator().manual_seed(train_config.seed)
    pin_memory = effective_pin_memory(train_config.pin_memory, device)

    train_sampler = None
    train_shuffle = True
    if getattr(train_config, "use_weighted_sampler", False):
        train_labels = [int(record.label) for record in splits["train"]]
        train_sampler = _build_weighted_sampler(train_labels, seed=train_config.seed)
        train_shuffle = False

    persistent = train_config.num_workers > 0
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=train_config.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=train_config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=2 if persistent else None,
            generator=generator if train_sampler is None else None,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=2 if persistent else None,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=2 if persistent else None,
        ),
    }
    return dataloaders, splits


def build_optimizer(model: nn.Module, train_config: TrainConfig) -> torch.optim.Optimizer:
    name = train_config.optimizer_name.lower()
    params = model.parameters()
    lr = train_config.learning_rate
    wd = train_config.weight_decay
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=train_config.sgd_momentum)
    raise ValueError(f"Unsupported optimizer: {train_config.optimizer_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, train_config: TrainConfig):
    name = train_config.scheduler_name.lower()
    if name == "none":
        base_scheduler = None
    elif name == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(train_config.epochs, 1))
    else:
        raise ValueError(f"Unsupported scheduler: {train_config.scheduler_name}")

    if train_config.warmup_epochs > 0 and base_scheduler is not None:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=train_config.warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, base_scheduler],
            milestones=[train_config.warmup_epochs],
        )
    return base_scheduler


def compute_pos_weight(records: list, strategy: str) -> float:
    raw = infer_positive_class_weight(records)
    if strategy == "ratio":
        return raw
    if strategy == "sqrt":
        return math.sqrt(raw)
    if strategy == "log":
        return math.log1p(raw)
    if strategy == "inverse":
        positives = sum(record.label for record in records)
        total = len(records)
        if positives == 0 or total == 0:
            return 1.0
        return float(total) / (2.0 * float(positives))
    if strategy == "effective":
        beta = 0.999
        positives = sum(record.label for record in records)
        negatives = len(records) - positives
        if positives == 0 or negatives == 0:
            return 1.0
        eff_pos = (1.0 - beta ** positives) / (1.0 - beta)
        eff_neg = (1.0 - beta ** negatives) / (1.0 - beta)
        return eff_neg / eff_pos
    if strategy == "none":
        return 1.0
    raise ValueError(f"Unsupported pos_weight_strategy: {strategy}")


class FocalLoss(nn.Module):
    """Sigmoid focal loss with optional positive-class reweighting.

    Focal loss suppresses the gradient contribution of already-confident
    predictions, which stabilises training on heavily imbalanced binary
    classification — precisely the regime the ALAN kidney dataset sits in.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer(
            "pos_weight",
            pos_weight if pos_weight is not None else torch.tensor([1.0]),
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        p_t = torch.where(targets > 0.5, probs, 1.0 - probs)
        focal_factor = (1.0 - p_t).clamp(min=1e-8) ** self.gamma
        return (focal_factor * bce).mean()


def build_criterion(train_config: TrainConfig, pos_weight: float, device: torch.device) -> nn.Module:
    pw_tensor = torch.tensor([pos_weight], device=device)
    loss_type = getattr(train_config, "loss_type", "bce").lower()
    if loss_type == "focal":
        return FocalLoss(gamma=getattr(train_config, "focal_gamma", 2.0), pos_weight=pw_tensor).to(device)
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def _run_epoch_raw(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_enabled: bool,
    tabular_feature_stats: dict[str, float] | None = None,
    gradient_clip_norm: float | None = None,
) -> tuple[list[float], list[float], float]:
    """Run one epoch and return (y_true, y_prob, mean_loss) without computing metrics."""
    is_training = optimizer is not None
    model.train(is_training)

    use_cuda_amp = amp_enabled and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    total_loss = 0.0
    total_samples = 0
    y_true: list[float] = []
    y_prob: list[float] = []

    grad_context = nullcontext() if is_training else torch.no_grad()
    with grad_context:
        for batch in dataloader:
            volumes = batch["volume"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            tabular_features = None
            if tabular_feature_stats is not None:
                tabular_features = build_tabular_features(batch, tabular_feature_stats, device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_cuda_amp else nullcontext()
            with autocast_ctx:
                logits = model(volumes, tabular_features=tabular_features)
                loss = criterion(logits, labels)

            if is_training:
                scaler.scale(loss).backward()
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            batch_size = labels.size(0)
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size
            probs = torch.sigmoid(logits.detach()).cpu().tolist()
            y_prob.extend(probs if isinstance(probs, list) else [probs])
            ground = labels.detach().cpu().tolist()
            y_true.extend(ground if isinstance(ground, list) else [ground])

    mean_loss = float(total_loss / max(total_samples, 1))
    return y_true, y_prob, mean_loss


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_enabled: bool,
    decision_threshold: float = 0.5,
    tabular_feature_stats: dict[str, float] | None = None,
    gradient_clip_norm: float | None = None,
) -> dict[str, float]:
    y_true, y_prob, mean_loss = _run_epoch_raw(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        amp_enabled=amp_enabled,
        tabular_feature_stats=tabular_feature_stats,
        gradient_clip_norm=gradient_clip_norm,
    )
    metrics = compute_binary_classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=decision_threshold,
    )
    metrics["loss"] = mean_loss
    return metrics


_TTA_FLIP_DIMS: tuple[tuple[int, ...], ...] = ((), (2,), (3,), (4,), (3, 4))


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    tabular_feature_stats: dict[str, float] | None = None,
    tta_enabled: bool = False,
) -> dict[str, list[float]]:
    """Run inference and return y_true / y_logit / y_prob lists.

    When ``tta_enabled`` is True, we average logits across identity and spatial
    flips — this typically adds a couple of recall points on the anomaly class
    at the cost of roughly 5× inference time.
    """
    model.eval()
    y_true: list[float] = []
    y_prob: list[float] = []
    y_logit: list[float] = []
    flip_dims_list = _TTA_FLIP_DIMS if tta_enabled else ((),)
    with torch.no_grad():
        for batch in dataloader:
            volumes = batch["volume"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            tabular_features = None
            if tabular_feature_stats is not None:
                tabular_features = build_tabular_features(batch, tabular_feature_stats, device)
            accum_logits: torch.Tensor | None = None
            for flip_dims in flip_dims_list:
                volumes_view = torch.flip(volumes, dims=list(flip_dims)) if flip_dims else volumes
                logits = model(volumes_view, tabular_features=tabular_features).detach().float()
                accum_logits = logits if accum_logits is None else accum_logits + logits
            assert accum_logits is not None
            mean_logits = accum_logits / float(len(flip_dims_list))
            logits_cpu = mean_logits.cpu().tolist()
            probs = torch.sigmoid(mean_logits).cpu().tolist()
            y_logit.extend(logits_cpu if isinstance(logits_cpu, list) else [logits_cpu])
            y_prob.extend(probs if isinstance(probs, list) else [probs])
            ground = labels.detach().cpu().tolist()
            y_true.extend(ground if isinstance(ground, list) else [ground])
    return {"y_true": y_true, "y_prob": y_prob, "y_logit": y_logit}


def run_training(
    data_config: DataConfig,
    augmentation_config: AugmentationConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    quiet: bool = False,
    skip_test: bool = False,
) -> dict[str, object]:
    data_config = data_config.resolved()
    output_dir = train_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(train_config.seed)

    device = resolve_device(train_config.device)
    dataloaders, splits = build_dataloaders(
        data_config=data_config,
        augmentation_config=augmentation_config,
        train_config=train_config,
        device=device,
    )
    tabular_feature_stats = (
        compute_tabular_feature_stats(splits["train"]) if model_config.use_tabular_features else None
    )

    model = build_model(
        model_config=model_config,
        num_tabular_features=len(TABULAR_FEATURE_NAMES) if model_config.use_tabular_features else 0,
    ).to(device)

    pos_weight = compute_pos_weight(splits["train"], train_config.pos_weight_strategy)
    criterion = build_criterion(train_config, pos_weight, device)
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config)

    history: list[dict[str, object]] = []
    best_score = float("-inf")
    best_epoch = -1
    best_val_metrics: dict[str, float] = {}
    best_val_threshold = train_config.decision_threshold
    best_checkpoint = output_dir / "best_model.pt"
    patience_counter = 0

    for epoch in range(1, train_config.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp_enabled=train_config.amp,
            decision_threshold=train_config.decision_threshold,
            tabular_feature_stats=tabular_feature_stats,
            gradient_clip_norm=train_config.gradient_clip_norm,
        )
        val_y_true, val_y_prob, val_loss = _run_epoch_raw(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            optimizer=None,
            device=device,
            amp_enabled=False,
            tabular_feature_stats=tabular_feature_stats,
        )
        val_threshold = optimize_threshold(val_y_true, val_y_prob, method="youden")
        val_metrics = compute_binary_classification_metrics(
            y_true=val_y_true, y_prob=val_y_prob, threshold=val_threshold,
        )
        val_metrics["loss"] = val_loss
        score = select_model_score(val_metrics, primary_metric=train_config.primary_metric)

        entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "score": score}
        history.append(entry)

        if scheduler is not None:
            scheduler.step()

        improved = score > best_score + train_config.early_stopping_min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_val_threshold = val_threshold
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "score": score,
                },
                best_checkpoint,
            )
            save_json(
                {
                    "epoch": epoch,
                    "score": score,
                    "data_config": to_serializable(data_config),
                    "model_config": to_serializable(model_config),
                    "train_config": to_serializable(train_config),
                    "tabular_feature_names": list(TABULAR_FEATURE_NAMES),
                    "tabular_feature_stats": tabular_feature_stats,
                },
                output_dir / "checkpoint_meta.json",
            )
        elif epoch > train_config.warmup_epochs:
            patience_counter += 1

        if not quiet:
            marker = "*" if improved else " "
            roc = val_metrics.get("roc_auc")
            roc_str = f"{roc:.4f}" if roc is not None and not math.isnan(roc) else "N/A"
            primary_label = f"val_{train_config.primary_metric}"
            print(
                f"  [{marker}] Epoch {epoch:03d}/{train_config.epochs:03d}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_roc_auc={roc_str}  "
                f"{primary_label}={score:.4f}  "
                f"patience={patience_counter}/{train_config.early_stopping_patience}"
            )

        if patience_counter >= train_config.early_stopping_patience:
            if not quiet:
                print(f"  Early stopping at epoch {epoch}.")
            break

    # Load best checkpoint
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Collect validation predictions once; reuse for calibration, threshold selection, and plots.
    tta_enabled = bool(getattr(train_config, "tta_enabled", False))
    val_preds = collect_predictions(
        model, dataloaders["val"], device, tabular_feature_stats, tta_enabled=tta_enabled,
    )
    val_logits_np = np.asarray(val_preds.get("y_logit") or logits_from_probs(np.asarray(val_preds["y_prob"])))
    val_labels_np = np.asarray(val_preds["y_true"], dtype=np.float64)

    # ── Post-hoc calibration on val ──────────────────────────────────
    calibration_method = getattr(train_config, "calibration_method", "temperature").lower()
    use_temperature = (
        getattr(train_config, "calibrate_temperature", True)
        and "temperature" in calibration_method
    )
    use_isotonic = "isotonic" in calibration_method

    calibration_summary: dict[str, float] = {}
    temperature = 1.0
    if use_temperature:
        temp_result = fit_temperature(val_logits_np, val_labels_np)
        temperature = float(temp_result.temperature)
        calibration_summary = {
            "temperature": temperature,
            "nll_before": temp_result.nll_before,
            "nll_after": temp_result.nll_after,
            "ece_before": temp_result.ece_before,
            "ece_after": temp_result.ece_after,
        }
        if not quiet:
            print(
                f"  Temperature scaling: T={temperature:.4f}  "
                f"ECE {temp_result.ece_before:.4f} -> {temp_result.ece_after:.4f}"
            )

    val_probs_after_temp = apply_temperature(val_logits_np, temperature)

    isotonic_result = None
    if use_isotonic:
        isotonic_result = fit_isotonic(val_probs_after_temp, val_labels_np)
        calibration_summary.update({
            "isotonic_ece_before": isotonic_result.ece_before,
            "isotonic_ece_after": isotonic_result.ece_after,
        })
        if not quiet:
            print(
                f"  Isotonic calibration: ECE "
                f"{isotonic_result.ece_before:.4f} -> {isotonic_result.ece_after:.4f}"
            )
        val_probs_cal = apply_isotonic(val_probs_after_temp, isotonic_result)
    else:
        val_probs_cal = val_probs_after_temp

    # ── Threshold selection: fixed, single-split, and bootstrap-median ──
    threshold_method = getattr(train_config, "threshold_selection", "youden")
    threshold_beta = float(getattr(train_config, "threshold_fbeta", 1.0))
    fixed_threshold = float(train_config.decision_threshold)
    if threshold_method == "fixed":
        optimal_threshold = fixed_threshold
    else:
        optimal_threshold = select_threshold_bootstrap(
            val_preds["y_true"], val_probs_cal.tolist(),
            method=threshold_method, seed=train_config.seed, beta=threshold_beta,
        )
    if not quiet:
        print(
            f"  Threshold — fixed: {fixed_threshold:.4f}  "
            f"bootstrap_{threshold_method}: {optimal_threshold:.4f}"
        )

    # Store val metrics at both thresholds (calibrated probabilities)
    val_metrics_fixed = compute_binary_classification_metrics(
        val_preds["y_true"], val_probs_cal.tolist(), threshold=fixed_threshold,
    )
    val_metrics_tuned = compute_binary_classification_metrics(
        val_preds["y_true"], val_probs_cal.tolist(), threshold=optimal_threshold,
    )

    # Reliability diagram data for plots
    reliability_uncal = reliability_bins(val_preds["y_prob"], val_preds["y_true"])
    reliability_cal = reliability_bins(val_probs_cal.tolist(), val_preds["y_true"])
    calibration_payload = {
        "temperature": temperature,
        "calibration_method": calibration_method,
        "summary": calibration_summary,
        "ece_val_uncalibrated": expected_calibration_error(val_preds["y_prob"], val_preds["y_true"]),
        "ece_val_calibrated": expected_calibration_error(val_probs_cal.tolist(), val_preds["y_true"]),
        "reliability_uncalibrated": reliability_uncal,
        "reliability_calibrated": reliability_cal,
        "val_metrics_fixed_threshold": val_metrics_fixed,
        "val_metrics_tuned_threshold": val_metrics_tuned,
        "fixed_threshold": fixed_threshold,
        "tuned_threshold": optimal_threshold,
        "tta_enabled": tta_enabled,
        "val_y_true": val_preds["y_true"],
        "val_y_prob_uncalibrated": val_preds["y_prob"],
        "val_y_prob_calibrated": val_probs_cal.tolist(),
    }
    if isotonic_result is not None:
        calibration_payload["isotonic_x"] = list(isotonic_result.x)
        calibration_payload["isotonic_y"] = list(isotonic_result.y)
    save_json(calibration_payload, output_dir / "calibration.json")

    test_metrics: dict[str, float] = {}
    test_metrics_fixed: dict[str, float] = {}
    test_ci: dict = {}
    test_payload: dict = {}
    test_eval_error: str | None = None

    if not skip_test:
        # Test eval can crash on Windows with OpenBLAS memory-allocation errors
        # mid-bootstrap; guard the whole block so the trained checkpoint and
        # calibration artifacts aren't lost when it does.
        try:
            # Raw test predictions, then apply calibration for reported metrics.
            test_preds = collect_predictions(
                model, dataloaders["test"], device, tabular_feature_stats, tta_enabled=tta_enabled,
            )
            test_logits_np = np.asarray(
                test_preds.get("y_logit") or logits_from_probs(np.asarray(test_preds["y_prob"]))
            )
            test_probs_after_temp = apply_temperature(test_logits_np, temperature)
            if isotonic_result is not None:
                test_probs_cal = apply_isotonic(test_probs_after_temp, isotonic_result)
            else:
                test_probs_cal = test_probs_after_temp

            # Compute loss on the test set (AMP off for numerical parity).
            _, _, test_loss = _run_epoch_raw(
                model=model, dataloader=dataloaders["test"], criterion=criterion,
                optimizer=None, device=device, amp_enabled=False,
                tabular_feature_stats=tabular_feature_stats,
            )

            test_metrics_fixed = compute_binary_classification_metrics(
                test_preds["y_true"], test_probs_cal.tolist(), threshold=fixed_threshold,
            )
            test_metrics_fixed["loss"] = test_loss
            test_metrics = compute_binary_classification_metrics(
                test_preds["y_true"], test_probs_cal.tolist(), threshold=optimal_threshold,
            )
            test_metrics["loss"] = test_loss

            test_payload = {
                "y_true": test_preds["y_true"],
                "y_prob_uncalibrated": test_preds["y_prob"],
                "y_prob_calibrated": test_probs_cal.tolist(),
                "temperature": temperature,
                "fixed_threshold": fixed_threshold,
                "tuned_threshold": optimal_threshold,
                "ece_uncalibrated": expected_calibration_error(test_preds["y_prob"], test_preds["y_true"]),
                "ece_calibrated": expected_calibration_error(test_probs_cal.tolist(), test_preds["y_true"]),
            }

            # Bootstrap CI is the most BLAS-allocation-heavy step; isolate it so
            # a crash here still leaves us with the point-estimate test metrics.
            try:
                test_ci = bootstrap_confidence_intervals(
                    test_preds["y_true"], test_probs_cal.tolist(),
                    threshold=optimal_threshold, seed=train_config.seed,
                )
            except Exception as exc:
                test_eval_error = f"bootstrap_ci_failed: {type(exc).__name__}: {exc}"
                if not quiet:
                    print(f"  Warning: bootstrap CI failed ({exc}); continuing without it.")
        except Exception as exc:
            test_eval_error = f"test_eval_failed: {type(exc).__name__}: {exc}"
            if not quiet:
                print(f"  Warning: test evaluation failed ({exc}); saving training artifacts only.")

    config_payload = {
        "data": to_serializable(data_config),
        "augmentation": to_serializable(augmentation_config),
        "model": to_serializable(model_config),
        "train": to_serializable(train_config),
        "positive_class_weight": pos_weight,
        "optimal_threshold": optimal_threshold,
        "fixed_threshold": fixed_threshold,
        "temperature": temperature,
        "tabular_feature_names": list(TABULAR_FEATURE_NAMES) if model_config.use_tabular_features else [],
        "tabular_feature_stats": tabular_feature_stats,
        "test_eval_error": test_eval_error,
    }
    save_json(config_payload, output_dir / "config.json")
    save_json({"history": history}, output_dir / "history.json")
    save_json(best_val_metrics, output_dir / "best_val_metrics.json")
    if test_metrics:
        save_json(test_metrics, output_dir / "test_metrics.json")
    if test_metrics_fixed:
        save_json(test_metrics_fixed, output_dir / "test_metrics_fixed_threshold.json")
    if test_payload:
        save_json(test_payload, output_dir / "test_predictions.json")
    if test_ci:
        save_json(test_ci, output_dir / "test_confidence_intervals.json")

    if not quiet:
        if test_metrics:
            roc = test_metrics.get("roc_auc")
            roc_str = f"{roc:.4f}" if roc is not None and not math.isnan(roc) else "N/A"
            primary_value = test_metrics.get(train_config.primary_metric)
            primary_str = (
                f"{primary_value:.4f}"
                if primary_value is not None and not math.isnan(primary_value)
                else "N/A"
            )
            print(
                f"  Best epoch: {best_epoch} | Test ROC-AUC: {roc_str} | "
                f"Test {train_config.primary_metric}: {primary_str}"
            )
            if test_ci:
                print("  Bootstrap 95% CI:")
                for key, ci_vals in test_ci.items():
                    print(f"    {key}: {ci_vals['mean']:.4f} [{ci_vals['ci_lower']:.4f}, {ci_vals['ci_upper']:.4f}]")
        else:
            print(f"  Best epoch: {best_epoch} | Test evaluation skipped")

    history_path = output_dir / "history.json"
    if history_path.exists():
        try:
            generate_plots(history_path, out_dir=output_dir)
        except Exception as exc:
            if not quiet:
                print(f"  Warning: plot generation failed: {exc}")

    return {
        "output_dir": str(output_dir),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_fixed_threshold": test_metrics_fixed,
        "test_confidence_intervals": test_ci,
        "optimal_threshold": optimal_threshold,
        "fixed_threshold": fixed_threshold,
        "temperature": temperature,
        "checkpoint_path": str(best_checkpoint),
        "test_eval_error": test_eval_error,
    }


def run_cross_validation(
    data_config: DataConfig,
    augmentation_config: AugmentationConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    n_folds: int = 5,
    quiet: bool = False,
) -> dict[str, object]:
    """Run k-fold cross-validation and aggregate results."""
    from sklearn.model_selection import StratifiedGroupKFold

    data_config = data_config.resolved()
    output_dir = train_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(train_config.seed)

    records = load_records(
        info_csv=data_config.info_csv,
        volumes_dir=data_config.volumes_dir,
        metadata_csv=data_config.metadata_csv,
        summary_json=data_config.summary_json,
    )

    # Combine train + val for CV, keep test separate
    splits = split_records(
        records,
        train_subset=data_config.train_subset,
        val_subset=data_config.val_subset,
        test_subset=data_config.test_subset,
    )
    cv_records = splits["train"] + splits["val"]
    labels = [r.label for r in cv_records]
    # Patient groups: strip L/R suffix so both kidneys stay in the same fold
    groups = [r.roi_id.rsplit("_", 1)[0] for r in cv_records]

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=train_config.seed)
    fold_results: list[dict] = []

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(cv_records, labels, groups)):
        if not quiet:
            print(f"\n{'='*60}")
            print(f"  FOLD {fold_idx + 1}/{n_folds}")
            print(f"{'='*60}")

        fold_train = [cv_records[i] for i in train_indices]
        fold_val = [cv_records[i] for i in val_indices]

        fold_dir = output_dir / f"fold_{fold_idx + 1:02d}"
        from dataclasses import replace as dc_replace
        fold_train_config = dc_replace(train_config, output_dir=fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Build dataloaders manually for this fold
        device = resolve_device(fold_train_config.device)
        seed_everything(fold_train_config.seed + fold_idx)

        if data_config.nan_strategy == "drop_record":
            fold_train = [r for r in fold_train if not r.has_nan]
            fold_val = [r for r in fold_val if not r.has_nan]
            if not fold_train:
                raise ValueError(
                    f"All records in fold {fold_idx + 1} train split were dropped by nan_strategy='drop_record'."
                )
            if not fold_val:
                raise ValueError(
                    f"All records in fold {fold_idx + 1} val split were dropped by nan_strategy='drop_record'."
                )

        train_transform = build_train_augmentations(augmentation_config)
        common_kwargs = dict(
            target_shape=data_config.target_shape,
            use_bbox_crop=data_config.use_bbox_crop,
            bbox_margin=data_config.bbox_margin,
            pad_to_cube_input=data_config.pad_to_cube,
            canonicalize_right=data_config.canonicalize_right,
            right_flip_axis=data_config.right_flip_axis,
            nan_strategy=data_config.nan_strategy if data_config.nan_strategy != "drop_record" else "none",
            nan_fill_value=data_config.nan_fill_value,
        )

        from Preprocessing.dataset import AlanKidneyDataset
        datasets = {
            "train": AlanKidneyDataset(records=fold_train, transform=train_transform, **common_kwargs),
            "val": AlanKidneyDataset(records=fold_val, transform=None, **common_kwargs),
        }
        generator = torch.Generator().manual_seed(fold_train_config.seed + fold_idx)
        pin_memory = effective_pin_memory(fold_train_config.pin_memory, device)

        fold_sampler = None
        fold_shuffle = True
        if getattr(fold_train_config, "use_weighted_sampler", False):
            fold_sampler = _build_weighted_sampler(
                [int(r.label) for r in fold_train],
                seed=fold_train_config.seed + fold_idx,
            )
            fold_shuffle = False

        fold_persistent = fold_train_config.num_workers > 0
        dataloaders = {
            "train": DataLoader(
                datasets["train"], batch_size=fold_train_config.batch_size,
                shuffle=fold_shuffle, sampler=fold_sampler,
                num_workers=fold_train_config.num_workers, pin_memory=pin_memory,
                persistent_workers=fold_persistent,
                prefetch_factor=2 if fold_persistent else None,
                generator=generator if fold_sampler is None else None,
            ),
            "val": DataLoader(datasets["val"], batch_size=fold_train_config.batch_size, shuffle=False,
                              num_workers=fold_train_config.num_workers, pin_memory=pin_memory,
                              persistent_workers=fold_persistent,
                              prefetch_factor=2 if fold_persistent else None),
        }

        tabular_feature_stats = (
            compute_tabular_feature_stats(fold_train) if model_config.use_tabular_features else None
        )
        model = build_model(
            model_config=model_config,
            num_tabular_features=len(TABULAR_FEATURE_NAMES) if model_config.use_tabular_features else 0,
        ).to(device)

        pos_weight = compute_pos_weight(fold_train, fold_train_config.pos_weight_strategy)
        criterion = build_criterion(fold_train_config, pos_weight, device)
        optimizer = build_optimizer(model, fold_train_config)
        scheduler = build_scheduler(optimizer, fold_train_config)

        best_score = float("-inf")
        best_epoch = -1
        best_val_metrics: dict[str, float] = {}
        best_checkpoint = fold_dir / "best_model.pt"
        patience_counter = 0

        for epoch in range(1, fold_train_config.epochs + 1):
            train_metrics = run_epoch(model=model, dataloader=dataloaders["train"], criterion=criterion,
                                      optimizer=optimizer, device=device, amp_enabled=fold_train_config.amp,
                                      decision_threshold=fold_train_config.decision_threshold,
                                      tabular_feature_stats=tabular_feature_stats,
                                      gradient_clip_norm=fold_train_config.gradient_clip_norm)
            val_y_true, val_y_prob, val_loss = _run_epoch_raw(
                model=model, dataloader=dataloaders["val"], criterion=criterion,
                optimizer=None, device=device, amp_enabled=False,
                tabular_feature_stats=tabular_feature_stats,
            )
            val_threshold = optimize_threshold(val_y_true, val_y_prob, method="youden")
            val_metrics = compute_binary_classification_metrics(
                y_true=val_y_true, y_prob=val_y_prob, threshold=val_threshold,
            )
            val_metrics["loss"] = val_loss
            score = select_model_score(val_metrics, primary_metric=fold_train_config.primary_metric)

            if scheduler is not None:
                scheduler.step()

            if score > best_score + fold_train_config.early_stopping_min_delta:
                best_score = score
                best_epoch = epoch
                best_val_metrics = val_metrics
                patience_counter = 0
                torch.save({"model_state_dict": model.state_dict()}, best_checkpoint)
            elif epoch > fold_train_config.warmup_epochs:
                patience_counter += 1

            if patience_counter >= fold_train_config.early_stopping_patience:
                break

        fold_result = {
            "fold": fold_idx + 1,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_val_metrics": best_val_metrics,
        }
        fold_results.append(fold_result)
        save_json(fold_result, fold_dir / "fold_result.json")

        if not quiet:
            roc = best_val_metrics.get("roc_auc")
            roc_str = f"{roc:.4f}" if roc is not None and not math.isnan(roc) else "N/A"
            primary_value = best_val_metrics.get(fold_train_config.primary_metric)
            primary_str = (
                f"{primary_value:.4f}"
                if primary_value is not None and not math.isnan(primary_value)
                else "N/A"
            )
            print(
                f"  Fold {fold_idx + 1} | Best epoch: {best_epoch} | "
                f"Val ROC-AUC: {roc_str} | Val {fold_train_config.primary_metric}: {primary_str}"
            )

    # Aggregate fold validation metrics
    metric_keys = [k for k in fold_results[0]["best_val_metrics"] if isinstance(fold_results[0]["best_val_metrics"][k], (int, float))]
    aggregated: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = [fr["best_val_metrics"][key] for fr in fold_results if not math.isnan(fr["best_val_metrics"].get(key, math.nan))]
        if values:
            arr = np.array(values)
            aggregated[key] = {"mean": float(arr.mean()), "std": float(arr.std()), "values": values}

    cv_summary = {
        "n_folds": n_folds,
        "fold_results": fold_results,
        "aggregated_val_metrics": aggregated,
    }
    save_json(cv_summary, output_dir / "cv_summary.json")

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  CROSS-VALIDATION SUMMARY ({n_folds} folds)")
        print(f"{'='*60}")
        for key, stats in aggregated.items():
            print(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    return cv_summary
