"""run_experiments.py

Master orchestration script. Runs all backbone conditions × label sizes
and saves a unified results summary.

Steps:
  0. Checks that prepare_splits.py has been run (train_split.csv exists)
  1. Optionally runs self-supervised pretraining (Script 3) if checkpoint missing
  2. Runs finetune_brightness.py for every condition × label size combination
  3. Saves results_summary.csv comparing all conditions

Backbone conditions:
  - imagenet      (pure ImageNet baseline)
  - dino_counts   (your existing Script 1 checkpoint)
  - ssl           (self-supervised SimCLR pretraining)

Label sizes: configurable, default [50, 100, 200, 400, 800]

Usage:
    # Full run (all conditions, all label sizes)
    python run_experiments.py

    # Skip SSL pretraining if checkpoint already exists
    python run_experiments.py --skip-ssl-pretrain

    # Only run specific conditions
    python run_experiments.py --backbones imagenet dino_counts

    # Only run specific label sizes
    python run_experiments.py --label-sizes 100 400 800

    # Change brightness metric
    python run_experiments.py --metric luma_mean
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "splits"
TRAIN_CSV = SPLITS_DIR / "train_split.csv"
TEST_CSV = SPLITS_DIR / "test_split.csv"
SSL_CHECKPOINT = ROOT / "model-training" / "ssl-pretrain" / "best_ssl_backbone.pt"
OUTPUT_BASE = ROOT / "model-training" / "finetune-runs"
RESULTS_SUMMARY = ROOT / "model-training" / "results_summary.csv"

# Scripts (assumed to be in same directory as this file)
SCRIPT_DIR = Path(__file__).resolve().parent
SSL_SCRIPT = SCRIPT_DIR / "pretrain_selfsupervised.py"
FINETUNE_SCRIPT = SCRIPT_DIR / "finetune_brightness.py"

# ── Defaults ──────────────────────────────────────────────────────────────────
ALL_BACKBONES = ["imagenet", "dino_counts", "ssl"]
DEFAULT_LABEL_SIZES = [50, 100, 200, 400, 800]
DEFAULT_METRIC = "gray_mean_zscore"
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42


def check_prerequisites() -> None:
    """Fail fast if splits haven't been created yet."""
    if not TRAIN_CSV.exists():
        print("ERROR: train_split.csv not found.")
        print("Run prepare_splits.py first:\n  python prepare_splits.py")
        sys.exit(1)
    if not TEST_CSV.exists():
        print("ERROR: test_split.csv not found.")
        print("Run prepare_splits.py first:\n  python prepare_splits.py")
        sys.exit(1)
    print(f"✓ Found train split: {TRAIN_CSV}")
    print(f"✓ Found test split:  {TEST_CSV}")


def run_ssl_pretraining(ssl_epochs: int) -> None:
    """Run self-supervised pretraining if checkpoint doesn't exist."""
    if SSL_CHECKPOINT.exists():
        print(f"\n✓ SSL checkpoint already exists: {SSL_CHECKPOINT}")
        print("  Skipping SSL pretraining. Use --force-ssl-pretrain to re-run.")
        return

    print(f"\n{'='*60}")
    print("Running self-supervised pretraining (SimCLR)...")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(SSL_SCRIPT), "--epochs", str(ssl_epochs)],
        check=True,
    )
    elapsed = time.time() - t0
    print(f"\n✓ SSL pretraining complete ({elapsed/60:.1f} min)")


def run_finetune(
    backbone: str,
    n_train: int,
    metric: str,
    n_folds: int,
    seed: int,
) -> dict[str, str]:
    """Run a single fine-tuning condition and return result metadata."""
    print(f"\n{'='*60}")
    print(f"Fine-tuning: backbone={backbone}  n_train={n_train}  metric={metric}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [
            sys.executable, str(FINETUNE_SCRIPT),
            "--backbone", backbone,
            "--n-train", str(n_train),
            "--metric", metric,
            "--folds", str(n_folds),
            "--seed", str(seed),
        ],
        check=True,
    )
    elapsed = time.time() - t0
    print(f"✓ Done ({elapsed/60:.1f} min)")

    # Read fold summary and compute mean/std
    summary_path = OUTPUT_BASE / backbone / f"n{n_train}" / "fold_summary.csv"
    row = {
        "backbone": backbone,
        "n_train": n_train,
        "metric": metric,
        "elapsed_min": round(elapsed / 60, 2),
        "mae_mean": "",
        "mae_std": "",
        "rmse_mean": "",
        "rmse_std": "",
        "r2_mean": "",
        "r2_std": "",
    }

    if summary_path.exists():
        import numpy as np
        fold_rows = []
        with summary_path.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                fold_rows.append(r)
        if fold_rows:
            for k in ["mae", "rmse", "r2"]:
                vals = [float(r[k]) for r in fold_rows]
                row[f"{k}_mean"] = round(float(np.mean(vals)), 6)
                row[f"{k}_std"] = round(float(np.std(vals)), 6)

    return row


def save_results(all_rows: list[dict]) -> None:
    if not all_rows:
        return
    fieldnames = list(all_rows[0].keys())
    RESULTS_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n✓ Results summary saved → {RESULTS_SUMMARY}")


def print_final_summary(all_rows: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Backbone':<15} {'n_train':<10} {'MAE':<20} {'RMSE':<20} {'R²':<20}")
    print("-" * 85)
    for row in all_rows:
        mae_str = f"{row['mae_mean']} ± {row['mae_std']}" if row['mae_mean'] else "N/A"
        rmse_str = f"{row['rmse_mean']} ± {row['rmse_std']}" if row['rmse_mean'] else "N/A"
        r2_str = f"{row['r2_mean']} ± {row['r2_std']}" if row['r2_mean'] else "N/A"
        print(f"{row['backbone']:<15} {str(row['n_train']):<10} {mae_str:<20} {rmse_str:<20} {r2_str:<20}")


def main(
    backbones: list[str],
    label_sizes: list[int],
    metric: str,
    n_folds: int,
    seed: int,
    skip_ssl_pretrain: bool,
    force_ssl_pretrain: bool,
    ssl_epochs: int,
) -> None:
    print("Nighttime brightness prediction — experiment runner")
    print(f"Backbones:    {backbones}")
    print(f"Label sizes:  {label_sizes}")
    print(f"Metric:       {metric}")
    print(f"Folds:        {n_folds}")

    check_prerequisites()

    # Run SSL pretraining if needed
    if "ssl" in backbones:
        if force_ssl_pretrain or not skip_ssl_pretrain:
            run_ssl_pretraining(ssl_epochs)
        else:
            print(f"\n--skip-ssl-pretrain set. Using existing checkpoint: {SSL_CHECKPOINT}")
            if not SSL_CHECKPOINT.exists():
                print("ERROR: SSL checkpoint not found. Remove --skip-ssl-pretrain or run pretrain_selfsupervised.py")
                sys.exit(1)

    # Run all conditions
    all_rows = []
    total = len(backbones) * len(label_sizes)
    current = 0

    for backbone in backbones:
        for n_train in label_sizes:
            current += 1
            print(f"\n[{current}/{total}] backbone={backbone}  n_train={n_train}")
            try:
                row = run_finetune(backbone, n_train, metric, n_folds, seed)
                all_rows.append(row)
                # Save incrementally so partial runs aren't lost
                save_results(all_rows)
            except subprocess.CalledProcessError as e:
                print(f"ERROR: run failed for backbone={backbone} n_train={n_train}: {e}")
                all_rows.append({
                    "backbone": backbone,
                    "n_train": n_train,
                    "metric": metric,
                    "status": "FAILED",
                })

    print_final_summary(all_rows)
    save_results(all_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all brightness prediction experiments.")
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=ALL_BACKBONES,
        choices=ALL_BACKBONES,
        help="Which backbone conditions to run.",
    )
    parser.add_argument(
        "--label-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_LABEL_SIZES,
        help="Training set sizes to sweep over.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=[
            "gray_mean", "gray_median", "gray_trimmed_mean", "gray_p90",
            "luma_mean", "value_mean", "gray_mean_over_std",
            "gray_mean_zscore", "gray_mean_robust_zscore",
        ],
        help="Brightness metric to regress.",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--skip-ssl-pretrain",
        action="store_true",
        help="Skip SSL pretraining and use existing checkpoint.",
    )
    parser.add_argument(
        "--force-ssl-pretrain",
        action="store_true",
        help="Re-run SSL pretraining even if checkpoint exists.",
    )
    parser.add_argument(
        "--ssl-epochs",
        type=int,
        default=100,
        help="Number of epochs for SSL pretraining.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        backbones=args.backbones,
        label_sizes=args.label_sizes,
        metric=args.metric,
        n_folds=args.folds,
        seed=args.seed,
        skip_ssl_pretrain=args.skip_ssl_pretrain,
        force_ssl_pretrain=args.force_ssl_pretrain,
        ssl_epochs=args.ssl_epochs,
    )