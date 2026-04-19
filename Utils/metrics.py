from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_classification_metrics(
    y_true: list[float] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_prob_array = np.asarray(y_prob, dtype=np.float64)
    y_pred_array = (y_prob_array >= threshold).astype(np.int64)

    metrics: dict[str, float] = {
        "threshold": float(threshold),
        "accuracy": float((y_true_array == y_pred_array).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "support": float(len(y_true_array)),
        "support_positive": float(y_true_array.sum()),
        "support_negative": float((1 - y_true_array).sum()),
    }

    if len(np.unique(y_true_array)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_prob_array))
        metrics["pr_auc"] = float(average_precision_score(y_true_array, y_prob_array))
    else:
        metrics["roc_auc"] = math.nan
        metrics["pr_auc"] = math.nan

    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1]).ravel()
    metrics["tn"] = float(tn)
    metrics["fp"] = float(fp)
    metrics["fn"] = float(fn)
    metrics["tp"] = float(tp)
    return metrics


def optimize_threshold(
    y_true: list[float] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    method: str = "youden",
    beta: float = 1.0,
) -> float:
    """Find optimal classification threshold on a validation set.

    Methods:
        youden: Maximizes Youden's J statistic (sensitivity + specificity - 1).
        f1:     Maximizes F1 score over a grid of thresholds.
        fbeta:  Maximizes F_beta (beta>1 weights recall — useful for anomaly recall).
    """
    from sklearn.metrics import roc_curve

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    if len(np.unique(y_true_arr)) < 2:
        return 0.5

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true_arr, y_prob_arr)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        return float(thresholds[best_idx])

    if method in ("f1", "fbeta"):
        b = 1.0 if method == "f1" else float(beta)
        b2 = b * b
        best_score = -1.0
        best_thresh = 0.5
        for thresh in np.linspace(0.05, 0.95, 181):
            preds = (y_prob_arr >= thresh).astype(np.int64)
            tp = float(((preds == 1) & (y_true_arr == 1)).sum())
            fp = float(((preds == 1) & (y_true_arr == 0)).sum())
            fn = float(((preds == 0) & (y_true_arr == 1)).sum())
            denom = (1.0 + b2) * tp + b2 * fn + fp
            score = ((1.0 + b2) * tp / denom) if denom > 0 else 0.0
            if score > best_score:
                best_score = score
                best_thresh = float(thresh)
        return best_thresh

    raise ValueError(f"Unsupported threshold optimization method: {method}")


def bootstrap_confidence_intervals(
    y_true: list[float] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap confidence intervals for key classification metrics."""
    rng = np.random.RandomState(seed)
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true_arr)

    metric_keys = ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "precision", "recall"]
    boot_results: dict[str, list[float]] = {k: [] for k in metric_keys}

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        bt_true = y_true_arr[indices]
        bt_prob = y_prob_arr[indices]

        if len(np.unique(bt_true)) < 2:
            continue

        bt_metrics = compute_binary_classification_metrics(bt_true.tolist(), bt_prob.tolist(), threshold)
        for key in metric_keys:
            val = bt_metrics.get(key)
            if val is not None and not np.isnan(val):
                boot_results[key].append(val)

    alpha = 1 - confidence
    ci: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = np.array(boot_results[key])
        if len(values) < 10:
            continue
        ci[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "ci_lower": float(np.percentile(values, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(values, 100 * (1 - alpha / 2))),
        }
    return ci


def select_model_score(metrics: dict[str, float], primary_metric: str = "roc_auc") -> float:
    candidates = [primary_metric, "pr_auc", "balanced_accuracy", "f1"]
    for key in candidates:
        value = metrics.get(key)
        if value is None:
            continue
        if not np.isnan(value):
            return float(value)

    loss = metrics.get("loss")
    if loss is not None:
        return -float(loss)
    return float("-inf")

