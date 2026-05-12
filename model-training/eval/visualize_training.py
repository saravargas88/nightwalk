"""
Visualize EfficientNet Multi-Head Validation Performance

This script reads validation prediction CSVs from a training epoch directory
and generates plots showing per-target MAE trends across epochs.

Expected file names:
  model-training/efficientnet-train-epoch/val_preds_epoch_005.csv
  model-training/efficientnet-train-epoch/val_preds_epoch_010.csv
  ...

Run after training:
  python model-training/visualize_training.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_MODEL_TRAINING = Path(__file__).resolve().parent.parent
VAL_PREDS_DIR = _MODEL_TRAINING / "efficientnet-train-epoch"
OUTPUT_DIR = _MODEL_TRAINING
TARGETS = ["tree", "streetlight", "storefront"]


def extract_epoch(filename: str) -> int:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename}")
    return int(parts[-1])


def compute_mae_from_csv(path: Path):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        abs_errors = {t: [] for t in TARGETS}
        for row in reader:
            for t in TARGETS:
                label = float(row[t])
                pred = float(row[f"pred_{t}"])
                abs_errors[t].append(abs(label - pred))
    return {t: np.mean(abs_errors[t]) for t in TARGETS}


def build_metrics_dataframe(pred_dir: Path) -> pd.DataFrame:
    paths = sorted(pred_dir.glob("val_preds_epoch_*.csv"), key=lambda p: extract_epoch(p.name))
    if not paths:
        raise FileNotFoundError(f"No validation prediction CSVs found in {pred_dir}")

    rows = []
    for path in paths:
        epoch = extract_epoch(path.name)
        maes = compute_mae_from_csv(path)
        row = {"epoch": epoch}
        row.update({f"{t}_mae": maes[t] for t in TARGETS})
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    return df


def plot_mae_curves(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["green", "orange", "purple"]
    for target, color in zip(TARGETS, colors):
        ax.plot(df['epoch'], df[f'{target}_mae'], label=f'{target.capitalize()} MAE', color=color, linewidth=2, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Validation MAE by Target Across Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_overall_mae(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    df['mean_mae'] = df[[f'{t}_mae' for t in TARGETS]].mean(axis=1)
    ax.plot(df['epoch'], df['mean_mae'], label='Mean MAE', color='blue', linewidth=2, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Overall Validation MAE Across Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    if not VAL_PREDS_DIR.exists():
        print(f"Validation predictions directory not found: {VAL_PREDS_DIR}")
        return

    df = build_metrics_dataframe(VAL_PREDS_DIR)
    print(f"Loaded validation predictions for {len(df)} epochs")

    OUTPUT_DIR.mkdir(exist_ok=True)

    fig_mae = plot_mae_curves(df)
    fig_mae.savefig(OUTPUT_DIR / "val_mae_curves.png", dpi=300, bbox_inches='tight')
    plt.close(fig_mae)
    print(f"Saved validation MAE curves to {OUTPUT_DIR / 'val_mae_curves.png'}")

    fig_overall = plot_overall_mae(df)
    fig_overall.savefig(OUTPUT_DIR / "val_mean_mae.png", dpi=300, bbox_inches='tight')
    plt.close(fig_overall)
    print(f"Saved overall MAE curve to {OUTPUT_DIR / 'val_mean_mae.png'}")

    latest = df.iloc[-1]
    print("\nLatest validation MAE:")
    for t in TARGETS:
        print(f"{t.capitalize()}: {latest[f'{t}_mae']:.4f}")
    print(f"Mean MAE: {latest['mean_mae']:.4f}")

if __name__ == "__main__":
    main()
