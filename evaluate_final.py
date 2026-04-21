"""Final test-set evaluation for the ALAN kidney binary classifier.

Architecture-agnostic: reads the backbone (ResNet3D / U-Net3D / PointNet) from
the run's ``config.json``, rebuilds the model through ``Model.factory``, runs
inference on the ZS-test split, applies temperature calibration saved at
training time, and writes a full set of high-resolution diagnostic plots + a
textual summary into ``results/final_evaluation/<arch>_<run>/``.

Usage (from project root, with the project venv active):

    python evaluate_final.py --run-dir outputs/<run>/best_run
    python evaluate_final.py --run-dir outputs/unet_trial/best_run
    python evaluate_final.py --run-dir outputs/pointnet_trial/best_run
    python evaluate_final.py --use-saved-predictions   # skip inference, reuse test_predictions.json

All saved figure filenames are printed at the end.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from Model.factory import build_model
from Model.engine import (
    TABULAR_FEATURE_NAMES,
    build_dataloaders,
    collect_predictions,
    resolve_device,
)
from Utils.calibration import apply_temperature, logits_from_probs
from Utils.config import AugmentationConfig, DataConfig, ModelConfig, TrainConfig

CLASS_NAMES: tuple[str, str] = ("Normal", "Anomaly")
DPI = 220
BAR_PALETTE = ("#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728")

ARCHITECTURE_LABELS: dict[str, str] = {
    "resnet3d": "ResNet3D",
    "unet3d": "U-Net3D",
    "pointnet": "PointNet",
}


# ─────────────────────────── configuration loading ──────────────────────────

def _filter_kwargs(cls, payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {f for f in cls.__dataclass_fields__}
    return {k: v for k, v in payload.items() if k in allowed}


def load_saved_configs(run_dir: Path) -> tuple[DataConfig, ModelConfig, TrainConfig, dict]:
    """Reconstruct the dataclasses that were used to train the checkpoint."""
    config_path = run_dir / "config.json"
    meta_path = run_dir / "checkpoint_meta.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under {run_dir}")
    cfg = json.loads(config_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    data_payload = _filter_kwargs(DataConfig, cfg["data"])
    model_payload = _filter_kwargs(ModelConfig, cfg["model"])
    train_payload = _filter_kwargs(TrainConfig, cfg["train"])

    for key in ("info_csv", "volumes_dir", "metadata_csv", "summary_json"):
        if key in data_payload and not isinstance(data_payload[key], Path):
            data_payload[key] = Path(str(data_payload[key]).replace("\\", "/"))
    if "output_dir" in train_payload and not isinstance(train_payload["output_dir"], Path):
        train_payload["output_dir"] = Path(str(train_payload["output_dir"]).replace("\\", "/"))

    data_config = DataConfig(**data_payload).resolved()
    model_config = ModelConfig(**model_payload)
    train_config = TrainConfig(**train_payload)
    return data_config, model_config, train_config, cfg | {"checkpoint_meta": meta}


# ─────────────────────────── inference / predictions ─────────────────────────

def obtain_predictions(
    run_dir: Path,
    output_dir: Path,
    use_saved: bool,
) -> dict[str, Any]:
    """Return a dict with y_true, y_prob (calibrated) and metadata."""
    data_config, model_config, train_config, cfg = load_saved_configs(run_dir)
    temperature = float(cfg.get("temperature", 1.0))
    tuned_threshold = float(cfg.get("optimal_threshold", 0.5))
    fixed_threshold = float(cfg.get("fixed_threshold", 0.5))

    saved_pred_path = run_dir / "test_predictions.json"

    if use_saved:
        if not saved_pred_path.exists():
            raise FileNotFoundError(
                f"--use-saved-predictions was set but {saved_pred_path} is missing."
            )
        payload = json.loads(saved_pred_path.read_text())
        y_true = np.asarray(payload["y_true"], dtype=np.float64)
        y_prob_cal = np.asarray(
            payload.get("y_prob_calibrated")
            or payload.get("y_prob")
            or payload["y_prob_uncalibrated"],
            dtype=np.float64,
        )
        y_prob_raw = np.asarray(
            payload.get("y_prob_uncalibrated") or payload.get("y_prob") or y_prob_cal,
            dtype=np.float64,
        )
        source = f"saved predictions ({saved_pred_path.name})"
    else:
        device = resolve_device(train_config.device)
        augmentation_config = AugmentationConfig(enabled=False)
        dataloaders, _ = build_dataloaders(
            data_config=data_config,
            augmentation_config=augmentation_config,
            train_config=train_config,
            device=device,
        )
        tabular_stats = cfg.get("tabular_feature_stats") if model_config.use_tabular_features else None
        model = build_model(
            model_config=model_config,
            num_tabular_features=len(TABULAR_FEATURE_NAMES) if model_config.use_tabular_features else 0,
        ).to(device)

        ckpt_path = run_dir / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"[info] loaded checkpoint {ckpt_path} (epoch {checkpoint.get('epoch')}) on {device}")

        preds = collect_predictions(
            model=model,
            dataloader=dataloaders["test"],
            device=device,
            tabular_feature_stats=tabular_stats,
            tta_enabled=bool(getattr(train_config, "tta_enabled", False)),
        )
        y_true = np.asarray(preds["y_true"], dtype=np.float64)
        y_prob_raw = np.asarray(preds["y_prob"], dtype=np.float64)
        logits = np.asarray(preds.get("y_logit") or logits_from_probs(y_prob_raw))
        y_prob_cal = apply_temperature(logits, temperature)
        source = "fresh inference (ZS-test)"

        fresh_payload = {
            "y_true": y_true.tolist(),
            "y_prob_uncalibrated": y_prob_raw.tolist(),
            "y_prob_calibrated": y_prob_cal.tolist(),
            "temperature": temperature,
            "tuned_threshold": tuned_threshold,
            "fixed_threshold": fixed_threshold,
        }
        (output_dir / "test_predictions_rerun.json").write_text(
            json.dumps(fresh_payload, indent=2)
        )

    architecture = (getattr(model_config, "architecture", "resnet3d") or "resnet3d").lower()
    arch_label = ARCHITECTURE_LABELS.get(architecture, architecture)

    return {
        "y_true": y_true.astype(np.int64),
        "y_prob_cal": np.clip(y_prob_cal.astype(np.float64), 0.0, 1.0),
        "y_prob_raw": np.clip(y_prob_raw.astype(np.float64), 0.0, 1.0),
        "tuned_threshold": tuned_threshold,
        "fixed_threshold": fixed_threshold,
        "temperature": temperature,
        "source": source,
        "run_config": cfg,
        "architecture": architecture,
        "architecture_label": arch_label,
    }


# ────────────────────────────────── metrics ─────────────────────────────────

def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    support_neg = int((y_true == 0).sum())
    support_pos = int((y_true == 1).sum())

    roc_auc_val = float(roc_auc_score(y_true, y_prob))
    pr_auc_val = float(average_precision_score(y_true, y_prob))

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    per_class = {
        CLASS_NAMES[0]: {
            "precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "support": support_neg,
        },
        CLASS_NAMES[1]: {
            "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "support": support_pos,
        },
    }

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "threshold": float(threshold),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "precision_positive": per_class[CLASS_NAMES[1]]["precision"],
        "recall_positive": per_class[CLASS_NAMES[1]]["recall"],
        "f1_positive": per_class[CLASS_NAMES[1]]["f1"],
        "specificity": specificity,
        "sensitivity": sensitivity,
        "roc_auc": roc_auc_val,
        "auc": roc_auc_val,  # alias — in binary case AUC == ROC-AUC
        "pr_auc": pr_auc_val,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "weighted_precision": weighted_precision,
        "macro_recall": macro_recall,
        "weighted_recall": weighted_recall,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "support": int(len(y_true)),
        "support_positive": support_pos,
        "support_negative": support_neg,
        "per_class": per_class,
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=list(CLASS_NAMES),
            digits=4, zero_division=0,
        ),
    }


# ────────────────────────────────── plotting ────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, path: Path, arch_label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", rotation=270, labelpad=15)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    title_suffix = f" — {arch_label}" if arch_label else ""
    ax.set_title(f"Confusion Matrix — ZS-test{title_suffix}", fontsize=13, pad=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:d}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path, arch_label: str = "") -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.4, label=f"ROC curve (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#777777", lw=1.2, linestyle="--", label="Random baseline (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#1f77b4")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    title_suffix = f" ({arch_label})" if arch_label else ""
    ax.set_title(f"ROC Curve — Anomaly vs Normal{title_suffix}", fontsize=13, pad=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.text(
        0.58, 0.08, f"AUC = {roc_auc_val:.4f}",
        fontsize=13, color="#1f3c6b",
        bbox=dict(facecolor="white", edgecolor="#1f77b4", boxstyle="round,pad=0.35"),
    )
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return float(roc_auc_val)


def plot_metric_comparison(metrics: dict[str, Any], path: Path, arch_label: str = "") -> None:
    labels = ["Accuracy", "Recall", "F1 Score", "AUC"]
    values = [
        metrics["accuracy"],
        metrics["recall_positive"],
        metrics["f1_positive"],
        metrics["auc"],
    ]
    colors = BAR_PALETTE[: len(labels)]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    title_prefix = f"{arch_label} — " if arch_label else ""
    ax.set_title(f"{title_prefix}Key Test-Set Metrics (threshold = {metrics['threshold']:.3f})", fontsize=13, pad=10)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.015,
            f"{v:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(y_true: np.ndarray, y_prob: np.ndarray, path: Path, arch_label: str = "") -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)
    baseline = float((y_true == 1).mean())

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot(recall, precision, color="#2ca02c", lw=2.4, label=f"PR curve (AP = {pr_auc_val:.4f})")
    ax.axhline(baseline, color="#777777", ls="--", lw=1.2, label=f"Positive prevalence = {baseline:.3f}")
    ax.fill_between(recall, precision, alpha=0.12, color="#2ca02c")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    title_suffix = f" ({arch_label})" if arch_label else ""
    ax.set_title(f"Precision-Recall Curve — Anomaly class{title_suffix}", fontsize=13, pad=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return float(pr_auc_val)


def plot_class_wise_metrics(metrics: dict[str, Any], path: Path, arch_label: str = "") -> None:
    classes = list(CLASS_NAMES)
    precisions = [metrics["per_class"][c]["precision"] for c in classes]
    recalls = [metrics["per_class"][c]["recall"] for c in classes]
    f1s = [metrics["per_class"][c]["f1"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    b1 = ax.bar(x - width, precisions, width, label="Precision", color="#1f77b4", edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x,          recalls,    width, label="Recall",    color="#2ca02c", edgecolor="black", linewidth=0.4)
    b3 = ax.bar(x + width,  f1s,        width, label="F1",        color="#ff7f0e", edgecolor="black", linewidth=0.4)

    for group in (b1, b2, b3):
        for bar in group:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.015,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    title_suffix = f" — {arch_label}" if arch_label else ""
    ax.set_title(f"Class-wise Precision / Recall / F1{title_suffix}", fontsize=13, pad=10)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_support(metrics: dict[str, Any], path: Path, arch_label: str = "") -> None:
    classes = list(CLASS_NAMES)
    rows = []
    for c in classes:
        pc = metrics["per_class"][c]
        rows.append([c, f"{pc['precision']:.4f}", f"{pc['recall']:.4f}",
                     f"{pc['f1']:.4f}", f"{pc['support']:d}"])
    rows.append([
        "macro avg",
        f"{metrics['macro_precision']:.4f}",
        f"{metrics['macro_recall']:.4f}",
        f"{metrics['macro_f1']:.4f}",
        f"{metrics['support']:d}",
    ])
    rows.append([
        "weighted avg",
        f"{metrics['weighted_precision']:.4f}",
        f"{metrics['weighted_recall']:.4f}",
        f"{metrics['weighted_f1']:.4f}",
        f"{metrics['support']:d}",
    ])

    col_labels = ["Class", "Precision", "Recall", "F1", "Support"]

    fig, ax = plt.subplots(figsize=(8.2, 3.0 + 0.35 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=col_labels, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.5)

    for col in range(len(col_labels)):
        cell = table[0, col]
        cell.set_facecolor("#1f77b4")
        cell.set_text_props(color="white", fontweight="bold")

    for r in range(1, len(rows) + 1):
        base = "#f2f2f2" if r % 2 else "#ffffff"
        if rows[r - 1][0] in ("macro avg", "weighted avg"):
            base = "#fff2cc"
        for col in range(len(col_labels)):
            table[r, col].set_facecolor(base)

    title_suffix = f" — {arch_label}" if arch_label else ""
    ax.set_title(f"Per-Class Performance Summary{title_suffix}", fontsize=13, pad=12)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_probability_distribution(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, path: Path, arch_label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bins = np.linspace(0.0, 1.0, 31)
    ax.hist(
        y_prob[y_true == 0], bins=bins, alpha=0.65, color="#1f77b4",
        edgecolor="black", linewidth=0.4, label=f"{CLASS_NAMES[0]} (n={(y_true == 0).sum()})",
    )
    ax.hist(
        y_prob[y_true == 1], bins=bins, alpha=0.65, color="#d62728",
        edgecolor="black", linewidth=0.4, label=f"{CLASS_NAMES[1]} (n={(y_true == 1).sum()})",
    )
    ax.axvline(threshold, color="black", ls="--", lw=1.4, label=f"Decision threshold = {threshold:.3f}")
    ax.set_xlabel("Predicted probability (calibrated)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    title_suffix = f" — {arch_label}" if arch_label else ""
    ax.set_title(f"Predicted probability distribution by true class{title_suffix}", fontsize=13, pad=10)
    ax.legend(loc="upper center", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────── reporting / driver ────────────────────────────

def interpretation(metrics_tuned: dict[str, Any], metrics_fixed: dict[str, Any], arch_label: str = "") -> str:
    pc = metrics_tuned["per_class"]
    normal = pc[CLASS_NAMES[0]]
    anomaly = pc[CLASS_NAMES[1]]
    weakest = CLASS_NAMES[0] if normal["f1"] < anomaly["f1"] else CLASS_NAMES[1]
    model_ref = arch_label if arch_label else "model"
    lines = []
    header = f"Interpretation — {arch_label}" if arch_label else "Interpretation"
    lines.append(header)
    lines.append("=" * 60)
    lines.append(
        f"- ROC-AUC = {metrics_tuned['roc_auc']:.4f} and PR-AUC = {metrics_tuned['pr_auc']:.4f} "
        f"on {metrics_tuned['support']} test kidneys "
        f"({metrics_tuned['support_positive']} anomalies, {metrics_tuned['support_negative']} normals)."
    )
    lines.append(
        f"- At the tuned threshold {metrics_tuned['threshold']:.3f}: "
        f"accuracy {metrics_tuned['accuracy']:.4f}, sensitivity (recall-Anomaly) {metrics_tuned['sensitivity']:.4f}, "
        f"specificity {metrics_tuned['specificity']:.4f}, F1-Anomaly {metrics_tuned['f1_positive']:.4f}."
    )
    lines.append(
        f"- Strengths: high ranking quality (ROC-AUC ~= {metrics_tuned['roc_auc']:.3f}) means the {model_ref} "
        f"separates Normal from Anomaly reliably before thresholding; the "
        f"{CLASS_NAMES[0]} class scores F1 = {normal['f1']:.4f} (n={normal['support']})."
    )
    lines.append(
        f"- Weakest class: {weakest}. "
        f"Anomaly: precision {anomaly['precision']:.4f}, recall {anomaly['recall']:.4f}, F1 {anomaly['f1']:.4f}; "
        f"this is a consequence of the class imbalance (~{metrics_tuned['support_positive']}/{metrics_tuned['support']} positives) - "
        f"lowering the threshold trades precision for additional anomaly recall."
    )
    lines.append(
        f"- At the fixed 0.5 threshold the {model_ref} scores F1-Anomaly = {metrics_fixed['f1_positive']:.4f} / "
        f"accuracy {metrics_fixed['accuracy']:.4f}; the tuned threshold improves anomaly F1 "
        f"by {metrics_tuned['f1_positive'] - metrics_fixed['f1_positive']:+.4f}."
    )
    lines.append(
        "- Improvement areas: tackle the minority-class limit with (a) stronger sampling or loss re-weighting "
        "(pos_weight / focal), (b) heavier anatomy-preserving augmentation targeted at Anomaly voxels, "
        "(c) calibrated ensembling across ResNet3D / UNet3D / PointNet heads, and "
        "(d) threshold selection on a held-out val bootstrap (already in pipeline) revisited with F-beta > 1 "
        "if clinical cost of missing anomalies is high."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final test-set evaluation for the ALAN kidney classifier "
                    "(supports ResNet3D / U-Net3D / PointNet via Model.factory).",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("outputs/retrain_trial_044_200ep/best_run"),
        help="Directory containing best_model.pt + config.json (architecture is read from config.json).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write metrics + figures (default: results/final_evaluation/<arch>_<run-dir-name>).",
    )
    parser.add_argument(
        "--use-saved-predictions", action="store_true",
        help="Skip inference and reuse test_predictions.json from the run dir.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override decision threshold (default: tuned threshold stored in config.json).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Peek at architecture so we can prefix the output dir before inference runs.
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under {run_dir}")
    peek_cfg = json.loads(config_path.read_text())
    architecture = (peek_cfg.get("model", {}).get("architecture") or "resnet3d").lower()
    arch_label = ARCHITECTURE_LABELS.get(architecture, architecture)

    default_out_name = f"{architecture}_{run_dir.name}"
    output_dir = (args.output_dir or Path("results/final_evaluation") / default_out_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] run_dir      = {run_dir}")
    print(f"[info] architecture = {arch_label} ({architecture})")
    print(f"[info] output_dir   = {output_dir}")

    bundle = obtain_predictions(run_dir=run_dir, output_dir=output_dir, use_saved=args.use_saved_predictions)
    y_true = bundle["y_true"]
    y_prob = bundle["y_prob_cal"]
    threshold = float(args.threshold if args.threshold is not None else bundle["tuned_threshold"])
    fixed_threshold = float(bundle["fixed_threshold"])

    print(f"[info] predictions source = {bundle['source']}")
    print(f"[info] n_test = {len(y_true)}  positives = {int((y_true == 1).sum())}  negatives = {int((y_true == 0).sum())}")
    print(f"[info] temperature = {bundle['temperature']:.4f}  tuned_threshold = {threshold:.4f}  fixed_threshold = {fixed_threshold:.4f}")

    metrics_tuned = compute_all_metrics(y_true, y_prob, threshold=threshold)
    metrics_fixed = compute_all_metrics(y_true, y_prob, threshold=fixed_threshold)

    # ── plots ──
    cm = np.array([[metrics_tuned["tn"], metrics_tuned["fp"]],
                   [metrics_tuned["fn"], metrics_tuned["tp"]]], dtype=np.int64)

    fig_paths: dict[str, Path] = {
        "confusion_matrix": output_dir / "01_confusion_matrix.png",
        "roc_curve":        output_dir / "02_roc_curve.png",
        "metric_bar":       output_dir / "03_metric_comparison.png",
        "pr_curve":         output_dir / "04_precision_recall_curve.png",
        "class_wise":       output_dir / "05_class_wise_metrics.png",
        "summary_table":    output_dir / "06_per_class_summary.png",
        "prob_hist":        output_dir / "07_probability_distribution.png",
    }

    arch_label = bundle["architecture_label"]

    plot_confusion_matrix(cm, fig_paths["confusion_matrix"], arch_label=arch_label)
    plot_roc_curve(y_true, y_prob, fig_paths["roc_curve"], arch_label=arch_label)
    plot_metric_comparison(metrics_tuned, fig_paths["metric_bar"], arch_label=arch_label)
    plot_precision_recall(y_true, y_prob, fig_paths["pr_curve"], arch_label=arch_label)
    plot_class_wise_metrics(metrics_tuned, fig_paths["class_wise"], arch_label=arch_label)
    plot_per_class_support(metrics_tuned, fig_paths["summary_table"], arch_label=arch_label)
    plot_probability_distribution(y_true, y_prob, threshold, fig_paths["prob_hist"], arch_label=arch_label)

    # ── save textual outputs ──
    serializable = {
        "source": bundle["source"],
        "run_dir": str(run_dir),
        "architecture": bundle["architecture"],
        "architecture_label": bundle["architecture_label"],
        "temperature": bundle["temperature"],
        "tuned_threshold": threshold,
        "fixed_threshold": fixed_threshold,
        "metrics_tuned_threshold": {k: v for k, v in metrics_tuned.items() if k != "classification_report"},
        "metrics_fixed_threshold": {k: v for k, v in metrics_fixed.items() if k != "classification_report"},
    }
    (output_dir / "final_test_metrics.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    (output_dir / "classification_report_tuned.txt").write_text(metrics_tuned["classification_report"], encoding="utf-8")
    (output_dir / "classification_report_fixed.txt").write_text(metrics_fixed["classification_report"], encoding="utf-8")

    # ── terminal summary ──
    print("\n" + "=" * 70)
    print("FINAL TEST-SET METRICS  (calibrated probabilities)".center(70))
    print("=" * 70)
    print(f"  Architecture     : {arch_label}")
    print(f"  Source           : {bundle['source']}")
    print(f"  Test support     : {metrics_tuned['support']} "
          f"(pos={metrics_tuned['support_positive']}, neg={metrics_tuned['support_negative']})")
    print(f"  Temperature      : {bundle['temperature']:.4f}")
    print()
    print(f"  --- Tuned threshold  (t = {threshold:.4f}) ---")
    print(f"    Accuracy       : {metrics_tuned['accuracy']:.4f}")
    print(f"    Recall (Anom.) : {metrics_tuned['recall_positive']:.4f}")
    print(f"    F1 Score (Anom): {metrics_tuned['f1_positive']:.4f}")
    print(f"    AUC / ROC-AUC  : {metrics_tuned['roc_auc']:.4f}")
    print(f"    PR-AUC         : {metrics_tuned['pr_auc']:.4f}")
    print(f"    Specificity    : {metrics_tuned['specificity']:.4f}")
    print(f"    Balanced Acc.  : {metrics_tuned['balanced_accuracy']:.4f}")
    print(f"    Macro-F1       : {metrics_tuned['macro_f1']:.4f}   "
          f"Weighted-F1: {metrics_tuned['weighted_f1']:.4f}")
    print(f"    Confusion Mat. : TN={metrics_tuned['tn']} FP={metrics_tuned['fp']} "
          f"FN={metrics_tuned['fn']} TP={metrics_tuned['tp']}")
    print()
    print(f"  --- Fixed threshold  (t = {fixed_threshold:.4f}) ---")
    print(f"    Accuracy       : {metrics_fixed['accuracy']:.4f}")
    print(f"    Recall (Anom.) : {metrics_fixed['recall_positive']:.4f}")
    print(f"    F1 Score (Anom): {metrics_fixed['f1_positive']:.4f}")
    print(f"    AUC / ROC-AUC  : {metrics_fixed['roc_auc']:.4f}")

    print("\nClassification report (tuned threshold):")
    print(metrics_tuned["classification_report"])

    interp = interpretation(metrics_tuned, metrics_fixed, arch_label=arch_label)
    (output_dir / "interpretation.txt").write_text(interp, encoding="utf-8")
    try:
        print(interp)
    except UnicodeEncodeError:
        print(interp.encode("ascii", "replace").decode("ascii"))

    print("\nSaved figures:")
    for key, p in fig_paths.items():
        print(f"  {key:<20s} -> {p}")
    print(f"\nAll outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
