# -*- coding: utf-8 -*-
"""plot_results.py

Generates four paper-ready figures from finetune run logs:

  Fig 1 — Val vs Test R² bar chart (overfitting story)
  Fig 2 — Val loss training curves per backbone (mean ± std across folds)
  Fig 3 — Predicted vs actual scatter for best backbone (ImageNet)
  Fig 4 — Pretraining loss curves v1 vs v2 (if logs exist on disk)

Output: model-training/figures/  (PNG + PDF for each figure)

Run:
  python3 model-training/eval/plot_results.py
"""

from __future__ import annotations
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE           = Path(__file__).resolve().parent
_MODEL_TRAINING = _HERE.parent
RUNS_DIR        = _MODEL_TRAINING / "finetune-runs"
PRETRAIN_LOG_V1 = _MODEL_TRAINING / "training_log.csv"
PRETRAIN_LOG_V2 = _MODEL_TRAINING / "training_log_v2.csv"
OUT_DIR         = _MODEL_TRAINING / "figures"

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "imagenet":       "#2196F3",   # blue
    "ssl":            "#FF9800",   # orange
    "dino_counts":    "#4CAF50",   # green
    "dino_counts_v2": "#9C27B0",   # purple
}
LABELS = {
    "imagenet":       "ImageNet",
    "ssl":            "SSL",
    "dino_counts":    "DINO-counts v1\n(warmup + cosine fix)",
    "dino_counts_v2": "DINO-counts v2\n(bbox, warmup + cosine fix)",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── Data loading ──────────────────────────────────────────────────────────────
def load_test_summary(backbone: str, run: str = "n800") -> list[dict]:
    path = RUNS_DIR / backbone / run / "test_summary.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def load_fold_summary(backbone: str, run: str = "n800") -> list[dict]:
    path = RUNS_DIR / backbone / run / "fold_summary.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def load_training_logs(backbone: str, run: str = "n800") -> list[list[dict]]:
    logs = []
    for fold_dir in sorted((RUNS_DIR / backbone / run).glob("fold_*")):
        log = fold_dir / "training_log.csv"
        if log.exists():
            with log.open() as f:
                logs.append(list(csv.DictReader(f)))
    return logs


def load_test_predictions(backbone: str, run: str = "n800") -> tuple[list, list]:
    trues, preds = [], []
    for fold_dir in sorted((RUNS_DIR / backbone / run).glob("fold_*")):
        pred_file = fold_dir / "test_predictions.csv"
        if pred_file.exists():
            with pred_file.open() as f:
                for row in csv.DictReader(f):
                    trues.append(float(row["true_target"]))
                    preds.append(float(row["pred_target"]))
    return trues, preds


def mean_std(vals: list[float]) -> tuple[float, float]:
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, s


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight")
    print(f"  Saved {name}.png / .pdf")


# ── Figure 1: Val vs Test R² bar chart ────────────────────────────────────────
def fig_val_test_r2() -> None:
    # (backbone_key, run_folder, display_label, color)
    conditions = [
        ("imagenet",    "n800",                     "ImageNet",                    COLORS["imagenet"]),
        ("ssl",         "n800",                     "SSL",                         COLORS["ssl"]),
        ("dino_counts", "n800",                     "DINO-counts v1\n(no warmup)", COLORS["dino_counts"]),
        ("dino_counts", "n800_warmup10_lr1e5",      "DINO-counts v1\n(warmup)",    "#81C784"),
        ("dino_counts", "n800_warmup10_lr1e5_cosfix","DINO-counts v1\n(+ cosine)", "#2E7D32"),
        ("dino_counts", "n800_v2bbox",              "DINO-counts v2\n(bbox)",      COLORS["dino_counts_v2"]),
    ]

    val_means, val_stds, test_means, test_stds, labels, colors = [], [], [], [], [], []

    for backbone, run, label, color in conditions:
        fold_rows = load_fold_summary(backbone, run)
        test_rows = load_test_summary(backbone, run)
        if not fold_rows or not test_rows:
            print(f"  Skipping {backbone}/{run} (no data)")
            continue
        val_r2  = [float(r["r2"]) for r in fold_rows]
        test_r2 = [float(r["r2"]) for r in test_rows]
        vm, vs = mean_std(val_r2)
        tm, ts = mean_std(test_r2)
        val_means.append(vm);  val_stds.append(vs)
        test_means.append(tm); test_stds.append(ts)
        labels.append(label);  colors.append(color)

    n = len(labels)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_val  = ax.bar(x - w/2, val_means,  w, yerr=val_stds,  capsize=4,
                       color=[c + "99" for c in colors], label="Val", edgecolor="none")
    bars_test = ax.bar(x + w/2, test_means, w, yerr=test_stds, capsize=4,
                       color=colors, label="Test", edgecolor="none")

    ax.set_ylabel(r"$R^2$")
    ax.set_title("Backbone ablation: validation vs. test $R^2$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 0.70)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(frameon=False)

    # Annotate test bars with value
    for bar, m in zip(bars_test, test_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save(fig, "fig1_val_test_r2")
    plt.close(fig)


# ── Figure 2: Val loss training curves ────────────────────────────────────────
def fig_training_curves() -> None:
    conditions = [
        ("imagenet",    "n800",                      "ImageNet",                     COLORS["imagenet"]),
        ("ssl",         "n800",                      "SSL",                          COLORS["ssl"]),
        ("dino_counts", "n800",                      "DINO-counts v1 (no warmup)",   COLORS["dino_counts"]),
        ("dino_counts", "n800_warmup10_lr1e5",       "DINO-counts v1 (warmup)",      "#81C784"),
        ("dino_counts", "n800_warmup10_lr1e5_cosfix","DINO-counts v1 (cosine fix)",  "#2E7D32"),
        ("dino_counts", "n800_v2bbox",               "DINO-counts v2 (bbox)",        COLORS["dino_counts_v2"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for backbone, run, label, color in conditions:
        logs = load_training_logs(backbone, run)
        if not logs:
            print(f"  Skipping {backbone}/{run} (no logs)")
            continue
        max_ep = max(len(log) for log in logs)
        # Align folds by epoch index (some folds may have fewer epochs)
        val_by_epoch = []
        for ep in range(max_ep):
            vals = [float(log[ep]["val_loss"]) for log in logs if ep < len(log)]
            val_by_epoch.append(vals)

        means = [sum(v)/len(v) for v in val_by_epoch]
        stds  = [math.sqrt(sum((x - m)**2 for x in v)/(len(v)-1)) if len(v)>1 else 0
                 for v, m in zip(val_by_epoch, means)]
        epochs = list(range(1, max_ep + 1))

        ax.plot(epochs, means, color=color, label=label, linewidth=2)
        ax.fill_between(epochs,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         color=color, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (Huber)")
    ax.set_title("Fine-tuning validation loss over epochs")
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    save(fig, "fig2_training_curves")
    plt.close(fig)


# ── Figure 3: Predicted vs actual scatter (ImageNet + DINO-counts side by side)
def fig_scatter() -> None:
    panels = [
        ("imagenet", "n800",                      "ImageNet",                  COLORS["imagenet"]),
        ("dino_counts", "n800_warmup10_lr1e5_cosfix", "DINO-counts v1\n(warmup + cosine fix)", COLORS["dino_counts"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, (backbone, run, title, color) in zip(axes, panels):
        trues, preds = load_test_predictions(backbone, run)
        if not trues:
            print(f"  Skipping scatter for {backbone}/{run} — no test predictions found")
            continue

        trues = np.array(trues)
        preds = np.array(preds)

        ss_res = ((trues - preds) ** 2).sum()
        ss_tot = ((trues - trues.mean()) ** 2).sum()
        r2  = 1 - ss_res / ss_tot
        mae = np.abs(trues - preds).mean()

        lo = min(trues.min(), preds.min()) - 0.2
        hi = max(trues.max(), preds.max()) + 0.2

        ax.scatter(trues, preds, alpha=0.45, s=18, color=color, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("True brightness ($z$-score)")
        ax.set_ylabel("Predicted brightness ($z$-score)")
        ax.set_title(title)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.text(0.05, 0.92, f"$R^2 = {r2:.3f}$\nMAE $= {mae:.3f}$",
                transform=ax.transAxes, fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray"))
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Test set predictions: ImageNet vs. DINO-counts backbone", fontsize=13)
    fig.tight_layout()
    save(fig, "fig3_scatter_comparison")
    plt.close(fig)


# ── Figure 4: Pretraining loss v1 vs v2 ───────────────────────────────────────
def fig_pretraining_curves() -> None:
    curves = [
        (PRETRAIN_LOG_V1, "DINO-counts v1 (counts only)", COLORS["dino_counts"]),
        (PRETRAIN_LOG_V2, "DINO-counts v2 (counts + bbox area)", COLORS["dino_counts_v2"]),
    ]

    found = [(p, l, c) for p, l, c in curves if p.exists()]
    if not found:
        print("  Skipping pretraining curves — no training_log*.csv found locally.")
        print("  Copy model-training/training_log.csv and training_log_v2.csv from HPC.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for path, label, color in found:
        with path.open() as f:
            rows = list(csv.DictReader(f))
        epochs    = [int(r["epoch"])     for r in rows]
        val_loss  = [float(r["val_loss"]) for r in rows]
        train_loss = [float(r["train_loss"]) for r in rows]
        ax.plot(epochs, val_loss,   color=color, linewidth=2,   label=f"{label} (val)")
        ax.plot(epochs, train_loss, color=color, linewidth=1.2, linestyle="--",
                label=f"{label} (train)", alpha=0.6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Huber)")
    ax.set_title("EfficientNet pretraining loss: v1 vs v2")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig4_pretraining_curves")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    print("\n[1/4] Val vs Test R² bar chart")
    fig_val_test_r2()
    print("\n[2/4] Training curves")
    fig_training_curves()
    print("\n[3/4] Predicted vs actual scatter")
    fig_scatter()
    print("\n[4/4] Pretraining curves")
    fig_pretraining_curves()
    print(f"\nDone. Figures saved to {OUT_DIR}")
