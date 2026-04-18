"""Eğitim geçmişinden (history.json) performans metriklerini görselleştirir.

Kullanım:
    python -m Utils.plot_metrics <history.json> [--out-dir <çıktı_dizini>] [--show]

Üretilen grafikler:
    1. Loss (Train vs Val)
    2. ROC-AUC & PR-AUC
    3. Accuracy & Balanced Accuracy
    4. Precision, Recall & F1
    5. Confusion Matrix (son epoch - Train & Val)
    6. Tüm metrikler özet tablosu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ──────────────────────────── yardımcılar ────────────────────────────
def _load_history(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "history" in data:
        return data["history"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Beklenmeyen history.json formatı: {type(data)}")


def _extract(history: list[dict], split: str, metric: str) -> tuple[list[int], list[float]]:
    """Belirli bir split ve metrik için epoch-değer çiftlerini döndürür."""
    epochs, values = [], []
    for entry in history:
        val = entry.get(split, {}).get(metric)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            epochs.append(entry["epoch"])
            values.append(val)
    return epochs, values


def _style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


# ──────────────────────────── grafik fonksiyonları ────────────────────
def plot_loss(ax: plt.Axes, history: list[dict]) -> None:
    for split, color, label in [("train", "#2196F3", "Train Loss"), ("val", "#F44336", "Val Loss")]:
        epochs, vals = _extract(history, split, "loss")
        ax.plot(epochs, vals, color=color, linewidth=2, label=label, marker="o", markersize=3)
    best_epoch = min(range(len(history)), key=lambda i: history[i].get("val", {}).get("loss", float("inf")))
    ax.axvline(history[best_epoch]["epoch"], color="gray", linestyle=":", alpha=0.6, label=f"Min Val Loss (epoch {history[best_epoch]['epoch']})")
    _style(ax, "Loss", "Loss")


def plot_auc(ax: plt.Axes, history: list[dict]) -> None:
    for metric, color, label in [
        ("roc_auc", "#4CAF50", "ROC-AUC"),
        ("pr_auc", "#FF9800", "PR-AUC"),
    ]:
        for split, ls in [("train", "-"), ("val", "--")]:
            epochs, vals = _extract(history, split, metric)
            if vals:
                prefix = "Train" if split == "train" else "Val"
                ax.plot(epochs, vals, color=color, linestyle=ls, linewidth=2,
                        label=f"{prefix} {label}", marker="o", markersize=3)
    ax.set_ylim(-0.05, 1.05)
    _style(ax, "ROC-AUC & PR-AUC", "Skor")


def plot_accuracy(ax: plt.Axes, history: list[dict]) -> None:
    colors = {"accuracy": "#9C27B0", "balanced_accuracy": "#00BCD4"}
    for metric, color in colors.items():
        nice = "Accuracy" if metric == "accuracy" else "Balanced Acc"
        for split, ls in [("train", "-"), ("val", "--")]:
            epochs, vals = _extract(history, split, metric)
            if vals:
                prefix = "Train" if split == "train" else "Val"
                ax.plot(epochs, vals, color=color, linestyle=ls, linewidth=2,
                        label=f"{prefix} {nice}", marker="o", markersize=3)
    ax.set_ylim(-0.05, 1.05)
    _style(ax, "Accuracy & Balanced Accuracy", "Skor")


def plot_prf(ax: plt.Axes, history: list[dict]) -> None:
    colors = {"precision": "#E91E63", "recall": "#3F51B5", "f1": "#009688"}
    for metric, color in colors.items():
        nice = metric.capitalize()
        for split, ls in [("train", "-"), ("val", "--")]:
            epochs, vals = _extract(history, split, metric)
            if vals:
                prefix = "Train" if split == "train" else "Val"
                ax.plot(epochs, vals, color=color, linestyle=ls, linewidth=2,
                        label=f"{prefix} {nice}", marker="o", markersize=3)
    ax.set_ylim(-0.05, 1.05)
    _style(ax, "Precision, Recall & F1", "Skor")


def plot_confusion_matrix(ax: plt.Axes, history: list[dict], split: str) -> None:
    last = history[-1].get(split, {})
    tp = last.get("tp", 0)
    fp = last.get("fp", 0)
    fn = last.get("fn", 0)
    tn = last.get("tn", 0)
    cm = np.array([[tn, fp], [fn, tp]])
    total = cm.sum()
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100 if total > 0 else 0
            ax.text(j, i, f"{int(cm[i, j])}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal (0)", "Anomali (1)"])
    ax.set_yticklabels(["Normal (0)", "Anomali (1)"])
    ax.set_xlabel("Tahmin", fontsize=11)
    ax.set_ylabel("Gerçek", fontsize=11)
    prefix = "Train" if split == "train" else "Val"
    ax.set_title(f"Confusion Matrix - {prefix} (Son Epoch)", fontsize=13, fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, shrink=0.8)


def plot_summary_table(ax: plt.Axes, history: list[dict]) -> None:
    """Son epoch için tüm metrik değerlerini tablo olarak gösterir."""
    ax.axis("off")
    last = history[-1]
    metrics_order = ["loss", "accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    nice_names = {
        "loss": "Loss", "accuracy": "Accuracy", "balanced_accuracy": "Balanced Acc",
        "f1": "F1 Score", "precision": "Precision", "recall": "Recall",
        "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
    }

    rows = []
    for m in metrics_order:
        train_val = last.get("train", {}).get(m)
        val_val = last.get("val", {}).get(m)
        t_str = f"{train_val:.4f}" if train_val is not None and not (isinstance(train_val, float) and np.isnan(train_val)) else "N/A"
        v_str = f"{val_val:.4f}" if val_val is not None and not (isinstance(val_val, float) and np.isnan(val_val)) else "N/A"
        rows.append([nice_names.get(m, m), t_str, v_str])

    table = ax.table(
        cellText=rows,
        colLabels=["Metrik", "Train", "Val"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ECEFF1")
        cell.set_edgecolor("#B0BEC5")

    ax.set_title(f"Son Epoch ({last['epoch']}) Metrikleri", fontsize=13, fontweight="bold", pad=20, y=0.95)


# ──────────────────────────── ana fonksiyon ────────────────────────────
def generate_plots(history_path: Path, out_dir: Path | None = None, show: bool = False) -> Path:
    history = _load_history(history_path)
    if not history:
        print("history.json boş, grafik üretilemiyor.")
        sys.exit(1)

    if out_dir is None:
        out_dir = history_path.parent

    fig = plt.figure(figsize=(20, 24), facecolor="white")
    fig.suptitle("Model Performans Metrikleri", fontsize=18, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3, top=0.94, bottom=0.04, left=0.07, right=0.95)

    plot_loss(fig.add_subplot(gs[0, 0]), history)
    plot_auc(fig.add_subplot(gs[0, 1]), history)
    plot_accuracy(fig.add_subplot(gs[1, 0]), history)
    plot_prf(fig.add_subplot(gs[1, 1]), history)
    plot_confusion_matrix(fig.add_subplot(gs[2, 0]), history, "train")
    plot_confusion_matrix(fig.add_subplot(gs[2, 1]), history, "val")
    plot_summary_table(fig.add_subplot(gs[3, :]), history)

    out_path = out_dir / "metrics_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Grafik kaydedildi: {out_path}")

    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ──────────────────────────── CLI ────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Eğitim metriklerini görselleştir")
    parser.add_argument("history", type=Path, help="history.json dosya yolu")
    parser.add_argument("--out-dir", type=Path, default=None, help="Grafik çıktı dizini (varsayılan: history.json ile aynı)")
    parser.add_argument("--show", action="store_true", help="Grafiği ekranda göster")
    args = parser.parse_args()

    if not args.history.exists():
        print(f"Dosya bulunamadı: {args.history}", file=sys.stderr)
        sys.exit(1)

    generate_plots(args.history, args.out_dir, args.show)


if __name__ == "__main__":
    main()
