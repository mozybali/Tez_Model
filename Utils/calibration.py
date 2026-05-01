"""Post-hoc probability calibration and threshold selection utilities.

- ``fit_temperature``: scales logits by a single scalar T fitted on the
  validation set (Platt/temperature scaling for binary logits).
- ``apply_temperature``: applies the fitted temperature to new logits.
- ``reliability_bins``: computes binned accuracy vs. confidence for reliability
  diagrams and expected calibration error (ECE).
- ``select_threshold_bootstrap``: picks a decision threshold on the validation
  set with bootstrap-averaged Youden's J to reduce single-split variance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TemperatureResult:
    temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _bce_from_logits_np(logits: np.ndarray, labels: np.ndarray) -> float:
    max_l = np.maximum(logits, 0.0)
    return float(np.mean(max_l - logits * labels + np.log1p(np.exp(-np.abs(logits)))))


def logits_from_probs(probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Recover logits from probabilities via the logit function."""
    clipped = np.clip(probs, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


_TEMPERATURE_MIN = 0.25
_TEMPERATURE_MAX = 10.0


def fit_temperature(
    logits: np.ndarray | list[float],
    labels: np.ndarray | list[float],
    max_iter: int = 200,
    lr: float = 0.01,
    t_min: float = _TEMPERATURE_MIN,
    t_max: float = _TEMPERATURE_MAX,
) -> TemperatureResult:
    """Fit a single-parameter temperature T minimizing BCE(logits / T, labels).

    T is clamped to ``[t_min, t_max]`` (default [0.25, 10.0]). Small val splits
    with saturated logits can otherwise drive LBFGS toward pathological T values
    (e.g. T=100+) that collapse all probabilities near 0.5 and destroy the
    threshold ranking.
    """
    logits_arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels_arr = np.asarray(labels, dtype=np.float64).reshape(-1)
    if logits_arr.size == 0:
        return TemperatureResult(1.0, float("nan"), float("nan"), float("nan"), float("nan"))

    nll_before = _bce_from_logits_np(logits_arr, labels_arr)
    ece_before = expected_calibration_error(_sigmoid_np(logits_arr), labels_arr)

    logit_t = torch.tensor(logits_arr, dtype=torch.float64)
    label_t = torch.tensor(labels_arr, dtype=torch.float64)
    log_temp = nn.Parameter(torch.zeros((), dtype=torch.float64))  # T = exp(log_temp)
    optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temp = torch.exp(log_temp)
        loss = F.binary_cross_entropy_with_logits(logit_t / temp, label_t)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        pass

    temperature = float(torch.exp(log_temp).detach().item())
    if not np.isfinite(temperature) or temperature <= 0.0:
        temperature = 1.0
    temperature = float(np.clip(temperature, t_min, t_max))

    scaled = logits_arr / temperature
    nll_after = _bce_from_logits_np(scaled, labels_arr)
    ece_after = expected_calibration_error(_sigmoid_np(scaled), labels_arr)
    return TemperatureResult(temperature, nll_before, nll_after, ece_before, ece_after)


def apply_temperature(logits: np.ndarray | list[float], temperature: float) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    if temperature <= 0.0 or not np.isfinite(temperature):
        temperature = 1.0
    return _sigmoid_np(arr / temperature)


def reliability_bins(
    probs: np.ndarray | list[float],
    labels: np.ndarray | list[float],
    n_bins: int = 15,
) -> dict[str, list[float]]:
    """Equal-width reliability bins: confidence vs. empirical accuracy."""
    probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels_arr = np.asarray(labels, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, acc, conf, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs_arr >= lo) & (probs_arr <= hi)
        else:
            mask = (probs_arr >= lo) & (probs_arr < hi)
        count = int(mask.sum())
        centers.append(float((lo + hi) / 2))
        counts.append(count)
        if count > 0:
            acc.append(float(labels_arr[mask].mean()))
            conf.append(float(probs_arr[mask].mean()))
        else:
            acc.append(float("nan"))
            conf.append(float("nan"))
    return {"bin_center": centers, "accuracy": acc, "confidence": conf, "count": counts}


def expected_calibration_error(
    probs: np.ndarray | list[float],
    labels: np.ndarray | list[float],
    n_bins: int = 15,
) -> float:
    probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels_arr = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probs_arr.size == 0:
        return float("nan")
    bins = reliability_bins(probs_arr, labels_arr, n_bins=n_bins)
    total = float(sum(bins["count"]))
    if total == 0:
        return float("nan")
    ece = 0.0
    for acc, conf, count in zip(bins["accuracy"], bins["confidence"], bins["count"]):
        if count == 0 or np.isnan(acc) or np.isnan(conf):
            continue
        ece += (count / total) * abs(acc - conf)
    return float(ece)


def select_threshold_bootstrap(
    y_true: np.ndarray | list[float],
    y_prob: np.ndarray | list[float],
    method: str = "youden",
    n_bootstrap: int = 200,
    seed: int = 42,
    beta: float = 1.0,
    min_specificity: float | None = None,
    min_precision: float | None = None,
) -> float:
    """Bootstrap-averaged threshold selection — more stable than a single split.

    ``min_specificity`` / ``min_precision`` are forwarded to ``optimize_threshold``
    on every bootstrap resample so a recall-only objective cannot pick the
    threshold-collapses-to-0.05 corner.
    """
    from Utils.metrics import optimize_threshold

    y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if len(np.unique(y_true_arr)) < 2:
        return 0.5

    rng = np.random.RandomState(seed)
    n = len(y_true_arr)
    thresholds: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        bt_true = y_true_arr[idx]
        bt_prob = y_prob_arr[idx]
        if len(np.unique(bt_true)) < 2:
            continue
        thresholds.append(
            optimize_threshold(
                bt_true.tolist(), bt_prob.tolist(),
                method=method, beta=beta,
                min_specificity=min_specificity, min_precision=min_precision,
            )
        )
    if not thresholds:
        return optimize_threshold(
            y_true_arr.tolist(), y_prob_arr.tolist(),
            method=method, beta=beta,
            min_specificity=min_specificity, min_precision=min_precision,
        )
    return float(np.median(thresholds))


@dataclass(frozen=True, slots=True)
class IsotonicResult:
    """Monotone piecewise-linear probability calibration fitted on held-out logits."""
    x: tuple[float, ...]
    y: tuple[float, ...]
    ece_before: float
    ece_after: float


def fit_isotonic(
    probs: np.ndarray | list[float],
    labels: np.ndarray | list[float],
) -> IsotonicResult:
    """Fit isotonic regression p_calibrated = f(p_raw). Robust when temperature
    scaling isn't enough (e.g. bimodal logit distribution, skewed confidence)."""
    from sklearn.isotonic import IsotonicRegression

    probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels_arr = np.asarray(labels, dtype=np.float64).reshape(-1)
    ece_before = expected_calibration_error(probs_arr, labels_arr)
    if probs_arr.size == 0 or len(np.unique(labels_arr)) < 2:
        return IsotonicResult(tuple(probs_arr.tolist()), tuple(labels_arr.tolist()), ece_before, ece_before)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(probs_arr, labels_arr)
    calibrated = iso.predict(probs_arr)
    ece_after = expected_calibration_error(calibrated, labels_arr)
    return IsotonicResult(
        x=tuple(float(v) for v in iso.X_thresholds_),
        y=tuple(float(v) for v in iso.y_thresholds_),
        ece_before=ece_before,
        ece_after=ece_after,
    )


def apply_isotonic(probs: np.ndarray | list[float], result: IsotonicResult) -> np.ndarray:
    """Apply a fitted isotonic mapping to new probabilities."""
    arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    if not result.x:
        return arr
    xs = np.asarray(result.x, dtype=np.float64)
    ys = np.asarray(result.y, dtype=np.float64)
    clipped = np.clip(arr, xs[0], xs[-1])
    return np.clip(np.interp(clipped, xs, ys), 0.0, 1.0)
