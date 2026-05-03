from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from Model.engine import save_json
from Model.oof_predictions import (
    compute_oof_thresholds,
    load_or_generate_oof_predictions,
    prediction_map_from_oof,
    predict_trial_test_from_folds,
)
from Utils.calibration import expected_calibration_error, logits_from_probs
from Utils.metrics import bootstrap_confidence_intervals, compute_binary_classification_metrics


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _member_name(study_dir: Path, trial_number: int) -> str:
    return f"{Path(study_dir).name}/trial_{int(trial_number):03d}"


def _check_unique(ids: list[str], context: str) -> None:
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"{context}: duplicate ids detected: {preview}")


def _combine_probabilities(prob_stack: np.ndarray, mode: str) -> np.ndarray:
    if prob_stack.ndim != 2:
        raise ValueError(f"Expected a 2D probability stack, got shape {prob_stack.shape}.")
    if mode == "arithmetic":
        return np.mean(prob_stack, axis=0)
    if mode == "logit":
        clipped = np.clip(prob_stack, 1e-7, 1.0 - 1e-7)
        logits = logits_from_probs(clipped)
        return 1.0 / (1.0 + np.exp(-np.mean(logits, axis=0)))
    raise ValueError(f"Unsupported probability mode: {mode}")


def _align_oof_members(members: list[dict[str, Any]]) -> tuple[list[str], list[int], list[dict[str, Any]], np.ndarray]:
    reference_ids: list[str] | None = None
    reference_y: dict[str, int] | None = None
    aligned_members: list[dict[str, Any]] = []
    stacks: list[np.ndarray] = []

    for member in members:
        payload = member["oof_payload"]
        predictions = payload.get("predictions", [])
        ids = [str(sample["id"]) for sample in predictions]
        _check_unique(ids, f"OOF {member['name']}")
        sample_map = prediction_map_from_oof(payload)
        y_map = {sample_id: int(sample["y_true"]) for sample_id, sample in sample_map.items()}

        if reference_ids is None:
            reference_ids = ids
            reference_y = y_map
        else:
            if set(ids) != set(reference_ids):
                missing = sorted(set(reference_ids).difference(ids))
                extra = sorted(set(ids).difference(reference_ids))
                raise ValueError(
                    f"OOF ids differ for {member['name']}: missing={missing[:5]} extra={extra[:5]}"
                )
            assert reference_y is not None
            for sample_id in reference_ids:
                if y_map[sample_id] != reference_y[sample_id]:
                    raise ValueError(f"OOF label mismatch for id {sample_id} in {member['name']}.")

        assert reference_ids is not None
        probs = np.asarray(
            [float(sample_map[sample_id]["y_prob_calibrated"]) for sample_id in reference_ids],
            dtype=np.float64,
        )
        stacks.append(probs)
        aligned_members.append(
            {
                "member": member["name"],
                "study_dir": str(member["study_dir"]),
                "trial_number": int(member["trial_number"]),
                "y_prob": probs.tolist(),
            }
        )

    if reference_ids is None or reference_y is None:
        raise ValueError("No OOF ensemble members were provided.")
    y_true = [int(reference_y[sample_id]) for sample_id in reference_ids]
    return reference_ids, y_true, aligned_members, np.stack(stacks, axis=0)


def _align_test_members(members: list[dict[str, Any]]) -> tuple[list[str], list[int], list[dict[str, Any]], np.ndarray]:
    reference_ids: list[str] | None = None
    reference_y: dict[str, int] | None = None
    aligned_members: list[dict[str, Any]] = []
    stacks: list[np.ndarray] = []

    for member in members:
        payload = member["test_payload"]
        ids = [str(sample_id) for sample_id in payload["ids"]]
        y_true = [int(value) for value in payload["y_true"]]
        _check_unique(ids, f"test {member['name']}")
        y_map = {sample_id: label for sample_id, label in zip(ids, y_true)}

        if reference_ids is None:
            reference_ids = ids
            reference_y = y_map
        else:
            if set(ids) != set(reference_ids):
                missing = sorted(set(reference_ids).difference(ids))
                extra = sorted(set(ids).difference(reference_ids))
                raise ValueError(
                    f"Test ids differ for {member['name']}: missing={missing[:5]} extra={extra[:5]}"
                )
            assert reference_y is not None
            for sample_id in reference_ids:
                if y_map[sample_id] != reference_y[sample_id]:
                    raise ValueError(f"Test label mismatch for id {sample_id} in {member['name']}.")

        assert reference_ids is not None
        prob_map = {sample_id: float(prob) for sample_id, prob in zip(ids, payload["y_prob"])}
        probs = np.asarray([prob_map[sample_id] for sample_id in reference_ids], dtype=np.float64)
        stacks.append(probs)
        aligned_members.append(
            {
                "member": member["name"],
                "study_dir": str(member["study_dir"]),
                "trial_number": int(member["trial_number"]),
                "y_prob": probs.tolist(),
            }
        )

    if reference_ids is None or reference_y is None:
        raise ValueError("No test ensemble members were provided.")
    y_true = [int(reference_y[sample_id]) for sample_id in reference_ids]
    return reference_ids, y_true, aligned_members, np.stack(stacks, axis=0)


def _load_single_model_comparisons(study_dirs: list[Path]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for study_dir in study_dirs:
        final_metrics_path = Path(study_dir) / "final_evaluation" / "final_test_metrics.json"
        study_summary_path = Path(study_dir) / "study_summary.json"
        item: dict[str, Any] = {
            "study_dir": str(study_dir),
            "comparison_only": True,
            "source": None,
            "test_f1": None,
            "metrics": None,
        }
        if final_metrics_path.exists():
            payload = _load_json(final_metrics_path)
            metrics = payload.get("metrics_tuned_threshold") or payload.get("test_metrics") or {}
            item.update(
                {
                    "source": str(final_metrics_path),
                    "threshold": payload.get("tuned_threshold") or metrics.get("threshold"),
                    "test_f1": metrics.get("f1"),
                    "metrics": metrics,
                }
            )
        elif study_summary_path.exists():
            payload = _load_json(study_summary_path)
            metrics = payload.get("final_results", {}).get("test_metrics", {})
            item.update(
                {
                    "source": str(study_summary_path),
                    "threshold": metrics.get("threshold"),
                    "test_f1": metrics.get("f1"),
                    "metrics": metrics,
                }
            )
        comparisons.append(item)
    return comparisons


def _calibration_summary(members: list[dict[str, Any]], ensemble_oof_prob: np.ndarray, oof_y_true: list[int]) -> dict[str, Any]:
    member_metrics = [
        {
            "member": member["name"],
            "study_dir": str(member["study_dir"]),
            "trial_number": int(member["trial_number"]),
            **(member["oof_payload"].get("calibration_metrics", {}) or {}),
        }
        for member in members
    ]

    def values_for(key: str) -> list[float]:
        values: list[float] = []
        for item in member_metrics:
            value = item.get(key)
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
        return values

    def mean_for(key: str) -> float:
        values = values_for(key)
        return float(np.mean(values)) if values else float("nan")

    return {
        "ensemble_oof_ece_calibrated": expected_calibration_error(ensemble_oof_prob, oof_y_true),
        "pooled_ece_before_temperature": mean_for("ece_before_temperature"),
        "pooled_ece_after_temperature": mean_for("ece_after_temperature"),
        "pooled_ece_after_pooled_isotonic": mean_for("ece_after_pooled_isotonic"),
        "pooled_ece_by_member": {
            "before_temperature": {
                item["member"]: item.get("ece_before_temperature") for item in member_metrics
            },
            "after_temperature": {
                item["member"]: item.get("ece_after_temperature") for item in member_metrics
            },
            "after_pooled_isotonic": {
                item["member"]: item.get("ece_after_pooled_isotonic") for item in member_metrics
            },
        },
        "members": member_metrics,
    }


def _interpretation_text(
    threshold_name: str,
    threshold_value: float,
    probability_mode: str,
    probability_mode_reason: str,
    seed: int,
    threshold_n_bootstrap: int,
    test_ci_n_bootstrap: int,
    oof_calibration: dict[str, Any],
    test_metrics: dict[str, float],
    comparisons: list[dict[str, Any]],
) -> str:
    final_f1 = float(test_metrics.get("f1", float("nan")))
    best_baseline = max(
        [float(item["test_f1"]) for item in comparisons if item.get("test_f1") is not None],
        default=float("nan"),
    )
    ece = float(oof_calibration.get("ensemble_oof_ece_calibrated", float("nan")))
    ece_ok = bool(np.isfinite(ece) and ece < 0.10)
    improved = bool(np.isfinite(final_f1) and np.isfinite(best_baseline) and final_f1 > best_baseline)
    target_hit = bool(np.isfinite(final_f1) and final_f1 >= 0.74)

    lines = [
        "OOF Ensemble Interpretation",
        "===========================",
        f"- Probability mode: {probability_mode} ({probability_mode_reason}).",
        f"- Bootstrap/random seed for ensemble thresholding and test CIs: {seed}.",
        f"- Bootstrap budgets: OOF threshold selection n={threshold_n_bootstrap}, test CI n={test_ci_n_bootstrap}.",
        f"- Locked threshold: {threshold_name} = {threshold_value:.4f}, selected from calibrated OOF predictions only.",
        "- The test set was evaluated once with the locked OOF threshold.",
        "- No test-set tuning was performed for threshold, ensemble weights, probability mode, member choice, or model selection.",
        f"- Pooled calibrated OOF ECE = {ece:.4f}; target < 0.10 was {'reached' if ece_ok else 'not reached'}.",
        f"- Final test F1 = {final_f1:.4f}.",
    ]
    if np.isfinite(best_baseline):
        lines.append(
            f"- Best comparison-only single-model tuned F1 = {best_baseline:.4f}; "
            f"the ensemble {'improved over it' if improved else 'did not improve over it'}."
        )
    else:
        lines.append("- Single-model baseline F1 was unavailable for comparison.")
    lines.append(f"- Target test F1 >= 0.74 was {'reached' if target_hit else 'not reached'}.")
    if not target_hit:
        lines.append(
            "- The target was not reached; likely causes are residual fold disagreement, calibration stair-stepping, "
            "and limited anomaly support. The threshold, members, weights, and probability mode were left unchanged."
        )
    return "\n".join(lines) + "\n"


def run_oof_ensemble(
    study_dirs: list[Path],
    trial_numbers: list[int],
    output_dir: Path,
    probability_mode: str = "arithmetic",
    threshold_name: str = "f1_threshold",
    force: bool = False,
    skip_existing: bool = False,
    device_name: str = "auto",
    n_bootstrap: int = 1000,
    test_ci_n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    if len(study_dirs) != len(trial_numbers):
        raise ValueError("--study-dirs and --trial-numbers must have the same length.")
    if len(study_dirs) < 2:
        raise ValueError("At least two study/trial pairs are required for this ensemble.")
    if probability_mode not in {"arithmetic", "logit"}:
        raise ValueError("probability_mode must be 'arithmetic' or 'logit'.")
    if int(n_bootstrap) <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if int(test_ci_n_bootstrap) <= 0:
        raise ValueError("test_ci_n_bootstrap must be positive.")

    output_dir = Path(output_dir)
    expected_outputs = [
        output_dir / "ensemble_predictions.json",
        output_dir / "oof_thresholds.json",
        output_dir / "final_test_metrics.json",
        output_dir / "test_confidence_intervals.json",
        output_dir / "interpretation.txt",
    ]
    if skip_existing and not force and all(path.exists() for path in expected_outputs):
        print(f"  Skipping existing ensemble outputs in {output_dir}")
        return {"output_dir": str(output_dir), "skipped": True}
    if not force and not skip_existing and any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"{output_dir} already contains ensemble outputs; use --force to overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    members: list[dict[str, Any]] = []
    for study_dir, trial_number in zip(study_dirs, trial_numbers):
        name = _member_name(study_dir, trial_number)
        print(f"  Member {name}: loading/generating OOF predictions")
        oof_payload = load_or_generate_oof_predictions(
            study_dir=study_dir,
            trial_number=int(trial_number),
            force=force,
            device_name=device_name,
            n_bootstrap=n_bootstrap,
            quiet=False,
        )
        print(f"  Member {name}: generating calibrated fold-averaged test predictions")
        test_payload = predict_trial_test_from_folds(
            study_dir=study_dir,
            trial_number=int(trial_number),
            oof_payload=oof_payload,
            device_name=device_name,
            quiet=False,
        )
        members.append(
            {
                "name": name,
                "study_dir": Path(study_dir),
                "trial_number": int(trial_number),
                "oof_payload": oof_payload,
                "test_payload": test_payload,
            }
        )

    oof_ids, oof_y_true, oof_member_probs, oof_stack = _align_oof_members(members)
    test_ids, test_y_true, test_member_probs, test_stack = _align_test_members(members)

    ensemble_oof_prob = _combine_probabilities(oof_stack, probability_mode)
    ensemble_test_prob = _combine_probabilities(test_stack, probability_mode)
    thresholds = compute_oof_thresholds(
        y_true=oof_y_true,
        y_prob=ensemble_oof_prob,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    if threshold_name not in thresholds:
        raise ValueError(f"Unknown threshold name '{threshold_name}'. Expected one of {sorted(thresholds)}.")
    locked_threshold = float(thresholds[threshold_name]["selected_threshold"])

    test_metrics_locked = compute_binary_classification_metrics(
        test_y_true,
        ensemble_test_prob.tolist(),
        threshold=locked_threshold,
    )
    test_metrics_fixed = compute_binary_classification_metrics(
        test_y_true,
        ensemble_test_prob.tolist(),
        threshold=0.5,
    )
    test_ci = bootstrap_confidence_intervals(
        test_y_true,
        ensemble_test_prob.tolist(),
        threshold=locked_threshold,
        n_bootstrap=test_ci_n_bootstrap,
        seed=seed,
    )
    comparisons = _load_single_model_comparisons([Path(path) for path in study_dirs])
    probability_mode_reason = (
        "explicit CLI setting; default is arithmetic and no test metrics were used"
        if probability_mode == "arithmetic"
        else "explicit CLI setting; no test metrics were used"
    )
    calibration_metrics = _calibration_summary(members, ensemble_oof_prob, oof_y_true)

    ensemble_predictions = {
        "test_ids": test_ids,
        "y_true": test_y_true,
        "per_member_calibrated_test_probabilities": test_member_probs,
        "ensemble_test_probability": ensemble_test_prob.tolist(),
        "oof_ids": oof_ids,
        "oof_y_true": oof_y_true,
        "per_member_calibrated_oof_probabilities": oof_member_probs,
        "ensemble_oof_probability": ensemble_oof_prob.tolist(),
        "probability_mode": probability_mode,
        "probability_mode_reason": probability_mode_reason,
        "seed": int(seed),
        "threshold_n_bootstrap": int(n_bootstrap),
        "test_ci_n_bootstrap": int(test_ci_n_bootstrap),
        "selected_threshold_name": threshold_name,
        "selected_threshold_value": locked_threshold,
    }
    oof_thresholds = {
        "probability_mode": probability_mode,
        "probability_mode_reason": probability_mode_reason,
        "seed": int(seed),
        "threshold_n_bootstrap": int(n_bootstrap),
        "test_ci_n_bootstrap": int(test_ci_n_bootstrap),
        "selected_threshold_name": threshold_name,
        "selected_threshold_value": locked_threshold,
        "f1_threshold": thresholds["f1_threshold"],
        "clinical_threshold": thresholds["clinical_threshold"],
        "calibration_metrics": calibration_metrics,
    }
    final_test_metrics = {
        "locked_threshold": locked_threshold,
        "selected_threshold_name": threshold_name,
        "probability_mode": probability_mode,
        "probability_mode_reason": probability_mode_reason,
        "seed": int(seed),
        "threshold_n_bootstrap": int(n_bootstrap),
        "test_ci_n_bootstrap": int(test_ci_n_bootstrap),
        "test_metrics_at_locked_oof_threshold": test_metrics_locked,
        "comparison_metrics_at_threshold_0_5": test_metrics_fixed,
        "single_model_best_run_comparison_only": comparisons,
        "test_set_tuning_performed": False,
    }

    save_json(ensemble_predictions, output_dir / "ensemble_predictions.json")
    save_json(oof_thresholds, output_dir / "oof_thresholds.json")
    save_json(final_test_metrics, output_dir / "final_test_metrics.json")
    save_json(
        {
            "locked_threshold": locked_threshold,
            "selected_threshold_name": threshold_name,
            "probability_mode": probability_mode,
            "seed": int(seed),
            "n_bootstrap": int(test_ci_n_bootstrap),
            "test_ci_n_bootstrap": int(test_ci_n_bootstrap),
            "threshold_n_bootstrap": int(n_bootstrap),
            "confidence_intervals": test_ci,
        },
        output_dir / "test_confidence_intervals.json",
    )
    interpretation = _interpretation_text(
        threshold_name=threshold_name,
        threshold_value=locked_threshold,
        probability_mode=probability_mode,
        probability_mode_reason=probability_mode_reason,
        seed=seed,
        threshold_n_bootstrap=n_bootstrap,
        test_ci_n_bootstrap=test_ci_n_bootstrap,
        oof_calibration=calibration_metrics,
        test_metrics=test_metrics_locked,
        comparisons=comparisons,
    )
    (output_dir / "interpretation.txt").write_text(interpretation)

    print(
        "  Ensemble "
        f"mode={probability_mode} threshold={locked_threshold:.4f} "
        f"F1={test_metrics_locked.get('f1', float('nan')):.4f} "
        f"ROC-AUC={test_metrics_locked.get('roc_auc', float('nan')):.4f} "
        f"PR-AUC={test_metrics_locked.get('pr_auc', float('nan')):.4f}"
    )
    return {
        "output_dir": str(output_dir),
        "test_metrics": test_metrics_locked,
        "threshold": locked_threshold,
        "probability_mode": probability_mode,
        "seed": int(seed),
        "threshold_n_bootstrap": int(n_bootstrap),
        "test_ci_n_bootstrap": int(test_ci_n_bootstrap),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF-locked calibrated ensemble over explicit CV trials.")
    parser.add_argument("--study-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--trial-numbers", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probability-mode", choices=["arithmetic", "logit"], default="arithmetic")
    parser.add_argument("--threshold-name", choices=["f1_threshold", "clinical_threshold"], default="f1_threshold")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap samples for OOF threshold selection.")
    parser.add_argument("--test-ci-n-bootstrap", type=int, default=1000,
                        help="Bootstrap samples for final test confidence intervals.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_oof_ensemble(
        study_dirs=args.study_dirs,
        trial_numbers=args.trial_numbers,
        output_dir=args.output_dir,
        probability_mode=args.probability_mode,
        threshold_name=args.threshold_name,
        force=args.force,
        skip_existing=args.skip_existing,
        device_name=args.device,
        n_bootstrap=args.n_bootstrap,
        test_ci_n_bootstrap=args.test_ci_n_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
