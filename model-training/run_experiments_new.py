"""run_experiments_new.py

Ablation study for the 4-class brightness classifier (gray_mean quartile bins).

Sweep conditions:
  backbone  × n_train
  ────────────────────
  imagenet  × [None (full), 600, 400]
  dino_counts × [None, 600, 400]

Each run outputs to:
  model-training/brightness-class-runs/<backbone>/n<n_train>/

A unified CSV is written to:
  model-training/brightness_class_results.csv

Usage:
    python run_experiments_new.py                             # full sweep
    python run_experiments_new.py --backbones imagenet        # single backbone
    python run_experiments_new.py --n-trains 600              # single size
    python run_experiments_new.py --epochs 45 --lr-backbone 5e-6
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = ROOT / "model-training" / "brightness-class-runs"
RESULTS_CSV = ROOT / "model-training" / "brightness_class_results.csv"
TRAIN_CSV = ROOT / "splits" / "train_split.csv"
TEST_CSV = ROOT / "splits" / "test_split.csv"
SCRIPT = Path(__file__).resolve().parent / "classification" / "train_brightness_class.py"

ALL_BACKBONES = ["imagenet", "dino_counts"]
ALL_N_TRAINS = [None, 600, 400]   # None = full training set


def check_prerequisites() -> None:
    missing = [str(p) for p in [TRAIN_CSV, TEST_CSV] if not p.exists()]
    if missing:
        print("ERROR: Missing split files:")
        for p in missing:
            print(f"  {p}")
        print("Run prepare_splits.py first.")
        sys.exit(1)
    print(f"✓ train split: {TRAIN_CSV}")
    print(f"✓ test split:  {TEST_CSV}")


def run_condition(
    backbone: str,
    n_train: int | None,
    epochs: int,
    lr_backbone: float,
    lr_head: float,
    seed: int,
) -> dict:
    tag = "full" if n_train is None else str(n_train)
    out_dir = OUTPUT_BASE / backbone / f"n{tag}"

    print(f"\n{'='*60}")
    print(f"backbone={backbone}  n_train={tag}  epochs={epochs}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(SCRIPT),
        "--backbone", backbone,
        "--epochs", str(epochs),
        "--lr-backbone", str(lr_backbone),
        "--lr-head", str(lr_head),
        "--seed", str(seed),
        "--output-dir", str(out_dir),
    ]
    if n_train is not None:
        cmd += ["--n-train", str(n_train)]

    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = round((time.time() - t0) / 60, 2)

    # Read best val accuracy from training log
    log_path = out_dir / "training_log.csv"
    best_val_acc = ""
    best_val_accs_per_class: dict[str, str] = {}
    if log_path.exists():
        rows = list(csv.DictReader(log_path.open(newline="")))
        if rows:
            best_row = max(rows, key=lambda r: float(r["val_acc"] or 0))
            best_val_acc = best_row["val_acc"]
            best_val_accs_per_class = {
                k: best_row.get(k, "") for k in best_row if k.startswith("val_acc_")
            }

    return {
        "backbone": backbone,
        "n_train": tag,
        "epochs": epochs,
        "lr_backbone": lr_backbone,
        "lr_head": lr_head,
        "elapsed_min": elapsed,
        "best_val_acc": best_val_acc,
        **best_val_accs_per_class,
    }


def save_results(rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Results saved → {RESULTS_CSV}")


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("BRIGHTNESS CLASSIFICATION ABLATION — RESULTS")
    print(f"{'='*60}")
    header = f"{'Backbone':<15} {'n_train':<8} {'Val Acc':<12} {'Elapsed':<10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        status = r.get("status", "")
        acc = r.get("best_val_acc", "") if not status else f"FAILED"
        print(f"{r['backbone']:<15} {str(r['n_train']):<8} {str(acc):<12} {str(r.get('elapsed_min','')):<10}")


def main(
    backbones: list[str],
    n_trains: list[int | None],
    epochs: int,
    lr_backbone: float,
    lr_head: float,
    seed: int,
) -> None:
    print("Brightness classification ablation study")
    print(f"Backbones: {backbones}  |  n_trains: {n_trains}  |  epochs: {epochs}")
    print(f"LR: backbone={lr_backbone}  head={lr_head}  seed={seed}")

    check_prerequisites()

    conditions = [(b, n) for b in backbones for n in n_trains]
    all_rows: list[dict] = []

    for i, (backbone, n_train) in enumerate(conditions):
        print(f"\n[{i+1}/{len(conditions)}]")
        try:
            row = run_condition(backbone, n_train, epochs, lr_backbone, lr_head, seed)
            all_rows.append(row)
        except subprocess.CalledProcessError as e:
            tag = "full" if n_train is None else str(n_train)
            print(f"ERROR: backbone={backbone} n_train={tag}: {e}")
            all_rows.append({
                "backbone": backbone,
                "n_train": tag,
                "epochs": epochs,
                "lr_backbone": lr_backbone,
                "lr_head": lr_head,
                "status": "FAILED",
            })
        save_results(all_rows)  # incremental save

    print_summary(all_rows)
    save_results(all_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brightness classification ablation.")
    parser.add_argument("--backbones", nargs="+", default=ALL_BACKBONES, choices=ALL_BACKBONES)
    parser.add_argument(
        "--n-trains", nargs="+", type=lambda x: None if x == "full" else int(x),
        default=ALL_N_TRAINS,
        help="Training set sizes to sweep. Use 'full' for no cap. E.g.: full 600 400",
    )
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        backbones=args.backbones,
        n_trains=args.n_trains,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        seed=args.seed,
    )
