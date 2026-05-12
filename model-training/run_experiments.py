"""run_experiments.py

Master orchestration script. Runs all backbone conditions and saves a
unified results summary including both val and test metrics.

Backbone conditions:
  - imagenet      (pure ImageNet baseline)
  - dino_counts   (Script 1 checkpoint)
  - ssl           (self-supervised SimCLR pretraining)

Usage:
    python run_experiments.py                          # full run
    python run_experiments.py --skip-ssl-pretrain      # if SSL checkpoint already exists
    python run_experiments.py --backbones imagenet dino_counts  # subset of conditions
    python run_experiments.py --metric luma_mean       # different brightness metric
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
TRAIN_CSV = ROOT / "splits" / "train_split.csv"
TEST_CSV = ROOT / "splits" / "test_split.csv"
SSL_CHECKPOINT = ROOT / "model-training" / "ssl-pretrain" / "best_ssl_backbone.pt"
OUTPUT_BASE = ROOT / "model-training" / "finetune-runs"
RESULTS_SUMMARY = ROOT / "model-training" / "results_summary.csv"

SCRIPT_DIR = Path(__file__).resolve().parent
SSL_SCRIPT = SCRIPT_DIR / "pretraining" / "pretrain_selfsupervised.py"
FINETUNE_SCRIPT = SCRIPT_DIR / "regression" / "finetune_brightness.py"

# ── Defaults ──────────────────────────────────────────────────────────────────
ALL_BACKBONES = ["imagenet", "dino_counts", "ssl"]
DEFAULT_N_TRAIN = 800
DEFAULT_METRIC = "gray_mean_zscore"
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42


def check_prerequisites() -> None:
    missing = []
    if not TRAIN_CSV.exists():
        missing.append(str(TRAIN_CSV))
    if not TEST_CSV.exists():
        missing.append(str(TEST_CSV))
    if missing:
        print("ERROR: The following split files are missing:")
        for p in missing:
            print(f"  {p}")
        print("Run prepare_splits.py first.")
        sys.exit(1)
    print(f"✓ train split: {TRAIN_CSV}")
    print(f"✓ test split:  {TEST_CSV}")


def run_ssl_pretraining(ssl_epochs: int, force: bool) -> None:
    if SSL_CHECKPOINT.exists() and not force:
        print(f"\n✓ SSL checkpoint already exists: {SSL_CHECKPOINT}")
        return
    print(f"\n{'='*60}")
    print("Running self-supervised pretraining (SimCLR)...")
    print(f"{'='*60}")
    t0 = time.time()
    subprocess.run(
        [sys.executable, str(SSL_SCRIPT), "--epochs", str(ssl_epochs)],
        check=True,
    )
    print(f"\n✓ SSL pretraining complete ({(time.time()-t0)/60:.1f} min)")


def read_summary_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def run_finetune(
    backbone: str,
    n_train: int,
    metric: str,
    n_folds: int,
    seed: int,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Fine-tuning: backbone={backbone}  n_train={n_train}  metric={metric}")
    print(f"{'='*60}")

    t0 = time.time()
    subprocess.run(
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
    elapsed = round((time.time() - t0) / 60, 2)
    print(f"✓ Done ({elapsed} min)")

    run_dir = OUTPUT_BASE / backbone / f"n{n_train}"
    val_rows = read_summary_csv(run_dir / "fold_summary.csv")
    test_rows = read_summary_csv(run_dir / "test_summary.csv")

    def agg(rows: list[dict], prefix: str) -> dict:
        if not rows:
            return {f"{prefix}_{k}_{s}": "" for k in ["mae", "rmse", "r2"] for s in ["mean", "std"]}
        import numpy as np
        result = {}
        for k in ["mae", "rmse", "r2"]:
            vals = [float(r[k]) for r in rows if r.get(k)]
            result[f"{prefix}_{k}_mean"] = round(float(np.mean(vals)), 6)
            result[f"{prefix}_{k}_std"]  = round(float(np.std(vals)),  6)
        return result

    return {
        "backbone": backbone,
        "n_train": n_train,
        "metric": metric,
        "elapsed_min": elapsed,
        **agg(val_rows,  "val"),
        **agg(test_rows, "test"),
    }


def save_results(rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    RESULTS_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Results saved → {RESULTS_SUMMARY}")


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    header = f"{'Backbone':<15} {'n_train':<8} {'Val MAE':<20} {'Test MAE':<20} {'Test R²':<20}"
    print(header)
    print("-" * len(header))
    for r in rows:
        val_mae  = f"{r.get('val_mae_mean','')} ± {r.get('val_mae_std','')}"
        test_mae = f"{r.get('test_mae_mean','')} ± {r.get('test_mae_std','')}"
        test_r2  = f"{r.get('test_r2_mean','')} ± {r.get('test_r2_std','')}"
        print(f"{r['backbone']:<15} {str(r['n_train']):<8} {val_mae:<20} {test_mae:<20} {test_r2:<20}")


def main(
    backbones: list[str],
    n_train: int,
    metric: str,
    n_folds: int,
    seed: int,
    skip_ssl_pretrain: bool,
    force_ssl_pretrain: bool,
    ssl_epochs: int,
) -> None:
    print("Nighttime brightness prediction — experiment runner")
    print(f"Backbones: {backbones}  |  n_train: {n_train}  |  metric: {metric}  |  folds: {n_folds}")

    check_prerequisites()

    if "ssl" in backbones:
        if skip_ssl_pretrain and not force_ssl_pretrain:
            if not SSL_CHECKPOINT.exists():
                print("ERROR: --skip-ssl-pretrain set but SSL checkpoint not found.")
                sys.exit(1)
            print(f"\n✓ Skipping SSL pretraining, using: {SSL_CHECKPOINT}")
        else:
            run_ssl_pretraining(ssl_epochs, force=force_ssl_pretrain)

    all_rows = []
    for i, backbone in enumerate(backbones):
        print(f"\n[{i+1}/{len(backbones)}] backbone={backbone}")
        try:
            row = run_finetune(backbone, n_train, metric, n_folds, seed)
            all_rows.append(row)
            save_results(all_rows)  # save incrementally
        except subprocess.CalledProcessError as e:
            print(f"ERROR: run failed for backbone={backbone}: {e}")
            all_rows.append({
                "backbone": backbone, "n_train": n_train,
                "metric": metric, "status": "FAILED",
            })

    print_summary(all_rows)
    save_results(all_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all brightness prediction experiments.")
    parser.add_argument("--backbones", nargs="+", default=ALL_BACKBONES, choices=ALL_BACKBONES)
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument(
        "--metric", default=DEFAULT_METRIC,
        choices=[
            "gray_mean", "gray_median", "gray_trimmed_mean", "gray_p90",
            "luma_mean", "value_mean", "gray_mean_over_std",
            "gray_mean_zscore", "gray_mean_robust_zscore",
        ],
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-ssl-pretrain", action="store_true")
    parser.add_argument("--force-ssl-pretrain", action="store_true")
    parser.add_argument("--ssl-epochs", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        backbones=args.backbones,
        n_train=args.n_train,
        metric=args.metric,
        n_folds=args.folds,
        seed=args.seed,
        skip_ssl_pretrain=args.skip_ssl_pretrain,
        force_ssl_pretrain=args.force_ssl_pretrain,
        ssl_epochs=args.ssl_epochs,
    )