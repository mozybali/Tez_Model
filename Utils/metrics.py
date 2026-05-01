from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator <= 0:
        return float(default)
    return float(numerator) / float(denominator)


def compute_binary_classification_metrics(
    y_true: list[float] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    threshold: float = 0.5,
    constrained_f1_min_specificity: float | None = None,
) -> dict[str, float]:
    """Compute the full binary-classification metric bundle used across the project.

    Existing keys (preserved for JSON backward-compatibility):
        threshold, accuracy, balanced_accuracy, f1, precision, recall,
        support, support_positive, support_negative, roc_auc, pr_auc,
        tn, fp, fn, tp.

    Newly added keys:
        sensitivity, recall_positive, precision_positive, f1_positive,
        specificity, npv, fpr, fnr, fdr, for, mcc, cohen_kappa,
        macro_precision, macro_recall, macro_f1,
        weighted_precision, weighted_recall, weighted_f1.

    ROC-AUC and PR-AUC fall back to NaN when ``y_true`` only contains one class.
    """
    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_prob_array = np.asarray(y_prob, dtype=np.float64)
    y_pred_array = (y_prob_array >= threshold).astype(np.int64)

    n_samples = int(y_true_array.shape[0])
    cm = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    sensitivity = _safe_divide(tp, tp + fn)         # recall on positive class
    specificity = _safe_divide(tn, tn + fp)
    precision_pos = _safe_divide(tp, tp + fp)
    npv = _safe_divide(tn, tn + fn)
    fpr = _safe_divide(fp, fp + tn)
    fnr = _safe_divide(fn, fn + tp)
    fdr = _safe_divide(fp, fp + tp)
    for_rate = _safe_divide(fn, fn + tn)
    f1_pos = _safe_divide(2.0 * precision_pos * sensitivity, precision_pos + sensitivity)

    # Matthews correlation coefficient — guard the sqrt denominator manually so a
    # single-class prediction returns 0 instead of nan.
    mcc_denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = _safe_divide(tp * tn - fp * fn, mcc_denom) if mcc_denom > 0 else 0.0

    metrics: dict[str, float] = {
        "threshold": float(threshold),
        "accuracy": float((y_true_array == y_pred_array).mean()) if n_samples > 0 else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)) if n_samples > 0 else 0.0,
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "support": float(n_samples),
        "support_positive": float(y_true_array.sum()),
        "support_negative": float((1 - y_true_array).sum()),
        # New positive-class aliases
        "sensitivity": sensitivity,
        "recall_positive": sensitivity,
        "precision_positive": precision_pos,
        "f1_positive": f1_pos,
        "specificity": specificity,
        # Confusion-matrix-derived rates
        "npv": npv,
        "fpr": fpr,
        "fnr": fnr,
        "fdr": fdr,
        "for": for_rate,
        # Agreement statistics
        "mcc": mcc,
        "cohen_kappa": float(cohen_kappa_score(y_true_array, y_pred_array, labels=[0, 1])),
        # Macro / weighted averages
        "macro_precision": float(precision_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true_array, y_pred_array, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true_array, y_pred_array, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_array, y_pred_array, average="weighted", zero_division=0)),
    }

    if len(np.unique(y_true_array)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_prob_array))
        except ValueError:
            metrics["roc_auc"] = math.nan
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true_array, y_prob_array))
        except ValueError:
            metrics["pr_auc"] = math.nan
    else:
        metrics["roc_auc"] = math.nan
        metrics["pr_auc"] = math.nan

    # Sanity: matthews_corrcoef gives the same result; we keep our own derivation
    # so the value lines up with the (tn, fp, fn, tp) we just reported, but cross-check
    # if both classes were predicted.
    if 0 < (tp + fn) < n_samples and 0 < (tp + fp) < n_samples:
        try:
            metrics["mcc"] = float(matthews_corrcoef(y_true_array, y_pred_array))
        except ValueError:
            pass

    metrics["tn"] = float(tn)
    metrics["fp"] = float(fp)
    metrics["fn"] = float(fn)
    metrics["tp"] = float(tp)

    # constrained_f1: F1 if specificity meets the floor, else 0. Lets HPO use
    # F1 as the objective without rewarding the recall-greedy collapse where
    # a model spams positives to drive F1 up at near-zero specificity.
    if constrained_f1_min_specificity is not None:
        metrics["constrained_f1"] = (
            float(metrics["f1"])
            if specificity >= float(constrained_f1_min_specificity)
            else 0.0
        )
    return metrics


def compute_per_class_report(
    y_true: list[float] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    threshold: float = 0.5,
    class_names: tuple[str, str] = ("Negative", "Positive"),
) -> dict[str, dict[str, float]]:
    """Per-class precision / recall / F1 / support broken out by label.

    Companion to ``compute_binary_classification_metrics``. Useful when callers
    want the same numbers shaped for tabular reporting (e.g. ``evaluate_final``).
    """
    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_prob_array = np.asarray(y_prob, dtype=np.float64)
    y_pred_array = (y_prob_array >= threshold).astype(np.int64)

    return {
        class_names[0]: {
            "precision": float(precision_score(y_true_array, y_pred_array, pos_label=0, zero_division=0)),
            "recall": float(recall_score(y_true_array, y_pred_array, pos_label=0, zero_division=0)),
            "f1": float(f1_score(y_true_array, y_pred_array, pos_label=0, zero_division=0)),
            "support": int((y_true_array == 0).sum()),
        },
        class_names[1]: {
            "precision": float(precision_score(y_true_array, y_pred_array, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true_array, y_pred_array, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_true_array, y_pred_array, pos_label=1, zero_division=0)),
            "support": int((y_true_array == 1).sum()),
        },
    }


def optimize_threshold(
    y_true: list[float] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    method: str = "youden",
    beta: float = 1.0,
    min_specificity: float | None = None,
    min_precision: float | None = None,
) -> float:
    """Find optimal classification threshold on a validation set.

    Methods:
        youden: Maximizes Youden's J statistic (sensitivity + specificity - 1).
        f1:     Maximizes F1 score over a grid of thresholds.
        fbeta:  Maximizes F_beta (beta>1 weights recall — useful for anomaly recall).

    ``min_specificity`` / ``min_precision`` (only honored for f1/fbeta) restrict
    the candidate set to thresholds whose specificity / precision on the
    provided sample meet the floors. Without this, β>1 will happily pick
    threshold≈0.05 (recall=1, specificity≈0.5) on small val sets.
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
        # Track unconstrained best as a fallback so we never return 0.5 just
        # because no candidate cleared the floor (can happen on a single fold).
        best_unconstrained_score = -1.0
        best_unconstrained_thresh = 0.5
        for thresh in np.linspace(0.05, 0.95, 181):
            preds = (y_prob_arr >= thresh).astype(np.int64)
            tp = float(((preds == 1) & (y_true_arr == 1)).sum())
            fp = float(((preds == 1) & (y_true_arr == 0)).sum())
            fn = float(((preds == 0) & (y_true_arr == 1)).sum())
            tn = float(((preds == 0) & (y_true_arr == 0)).sum())
            denom = (1.0 + b2) * tp + b2 * fn + fp
            score = ((1.0 + b2) * tp / denom) if denom > 0 else 0.0
            if score > best_unconstrained_score:
                best_unconstrained_score = score
                best_unconstrained_thresh = float(thresh)
            if min_specificity is not None:
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                if spec < float(min_specificity):
                    continue
            if min_precision is not None:
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                if prec < float(min_precision):
                    continue
            if score > best_score:
                best_score = score
                best_thresh = float(thresh)
        if best_score < 0.0:
            return best_unconstrained_thresh
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
    # Order matters: try the requested metric first, then fall back through
    # general-purpose metrics. mcc/constrained_f1 are only consulted when the
    # caller explicitly asks for them.
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
