"""Post-hoc threshold scan.

Reads val probabilities from calibration.json (already calibrated) and scans
thresholds to find F1-optimal and balanced (|precision - recall| min) options.
Then applies those thresholds to the test set and prints both confusion
matrices + metrics.

Usage:
    python -m Model.threshold_scan --run-dir outputs/retrain_trial_044_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def confusion(y_true, y_pred):
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        t, p = int(t), int(p)
        if t == 1 and p == 1: tp += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 0 and p == 0: tn += 1
        else: fn += 1
    return tp, fp, tn, fn


def metrics_from_cm(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    bal_acc = (rec + spec) / 2
    return dict(accuracy=acc, precision=prec, recall=rec, specificity=spec,
                f1=f1, balanced_acc=bal_acc)


def evaluate_at(y_true, y_prob, threshold):
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    tp, fp, tn, fn = confusion(y_true, y_pred)
    m = metrics_from_cm(tp, fp, tn, fn)
    m.update(tp=tp, fp=fp, tn=tn, fn=fn, threshold=threshold)
    return m


def scan(y_true, y_prob, grid=None):
    if grid is None:
        grid = [i / 1000 for i in range(5, 995, 5)]
    return [evaluate_at(y_true, y_prob, t) for t in grid]


def pick_f1(results):
    return max(results, key=lambda r: (r["f1"], -abs(r["precision"] - r["recall"])))


def pick_balanced(results, min_f1=0.70):
    viable = [r for r in results if r["f1"] >= min_f1]
    pool = viable if viable else results
    return min(pool, key=lambda r: (abs(r["precision"] - r["recall"]), -r["f1"]))


def print_cm(m, label):
    print(f"\n  {label}  (threshold = {m['threshold']:.3f})")
    print(f"    Confusion Matrix:")
    print(f"                  Pred=0   Pred=1")
    print(f"      True=0      {m['tn']:6d}   {m['fp']:6d}")
    print(f"      True=1      {m['fn']:6d}   {m['tp']:6d}")
    print(f"    Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  BalAcc={m['balanced_acc']:.4f}  Acc={m['accuracy']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--use-uncalibrated", action="store_true",
                    help="Scan on uncalibrated probs (default: calibrated).")
    args = ap.parse_args()

    cal = json.loads((args.run_dir / "calibration.json").read_text())
    val_y_true = cal["val_y_true"]
    val_prob_key = "val_y_prob_uncalibrated" if args.use_uncalibrated else "val_y_prob_calibrated"
    val_y_prob = cal[val_prob_key]

    print(f"Val size: {len(val_y_true)}  ({sum(val_y_true):.0f} positives)")
    print(f"Probs:    {val_prob_key}")

    val_scan = scan(val_y_true, val_y_prob)
    val_f1_best = pick_f1(val_scan)
    val_bal = pick_balanced(val_scan)
    val_tuned = cal.get("tuned_threshold")
    val_fixed = cal.get("fixed_threshold", 0.5)

    print("\n" + "=" * 72)
    print("VAL SET — threshold options")
    print("=" * 72)
    print_cm(evaluate_at(val_y_true, val_y_prob, val_fixed), f"Fixed 0.5 (baseline)")
    if val_tuned is not None:
        print_cm(evaluate_at(val_y_true, val_y_prob, val_tuned), f"Training's tuned ({val_tuned:.3f})")
    print_cm(val_f1_best, "F1-optimal on val")
    print_cm(val_bal, "Balanced (|P-R| min) on val")

    # Apply chosen thresholds to TEST
    tp_path = args.run_dir / "test_predictions.json"
    if tp_path.exists():
        tp = json.loads(tp_path.read_text())
        test_y_true = tp["y_true"]
        test_prob_key = "y_prob_uncalibrated" if args.use_uncalibrated else "y_prob_calibrated"
        test_y_prob = tp[test_prob_key]

        print("\n" + "=" * 72)
        print(f"TEST SET — applying val-selected thresholds "
              f"(size={len(test_y_true)}, pos={sum(test_y_true):.0f})")
        print("=" * 72)
        print_cm(evaluate_at(test_y_true, test_y_prob, val_fixed), "Fixed 0.5")
        if val_tuned is not None:
            print_cm(evaluate_at(test_y_true, test_y_prob, val_tuned), f"Training's tuned ({val_tuned:.3f})")
        print_cm(evaluate_at(test_y_true, test_y_prob, val_f1_best["threshold"]),
                 f"Val F1-optimal ({val_f1_best['threshold']:.3f})")
        print_cm(evaluate_at(test_y_true, test_y_prob, val_bal["threshold"]),
                 f"Val balanced ({val_bal['threshold']:.3f})")


if __name__ == "__main__":
    main()
