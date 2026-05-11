"""prepare_splits.py

Creates a geographically spread test split from finetune_pairs.csv.
Run this ONCE before any training. Outputs:
  - splits/test_split.csv       (200 geographically spread pairs, never touched during training)
  - splits/train_split.csv      (all remaining valid pairs)

Geographic spreading uses k-means clustering on day_lat/day_lon,
sampling evenly across clusters to maximize spatial coverage.

Usage:
    python prepare_splits.py
    python prepare_splits.py --test-size 200 --n-clusters 20 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
FINETUNE_CSV = ROOT / "finetune_pairs.csv"
OUT_DIR = ROOT 

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TEST_SIZE = 200
DEFAULT_N_CLUSTERS = 20
DEFAULT_SEED = 42


def load_valid_pairs(csv_path: Path) -> pd.DataFrame:
    """Load only rows that have a day_image (i.e. valid pairs)."""
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df[df["day_image"].notna() & (df["day_image"].str.strip() != "")].reset_index(drop=True)
    after = len(df)
    print(f"Loaded {before} rows, {after} valid pairs (dropped {before - after} unpaired night images)")
    return df


def geographic_test_split(
    df: pd.DataFrame,
    test_size: int,
    n_clusters: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Spread test samples geographically using k-means on day_lat/day_lon.
    Samples roughly test_size / n_clusters points per cluster,
    then trims/pads to exactly test_size.
    """
    rng = np.random.RandomState(seed)

    coords = df[["day_lat", "day_lon"]].values
    n_clusters = min(n_clusters, len(df) // 2)

    print(f"Clustering {len(df)} pairs into {n_clusters} geographic clusters ...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    df = df.copy()
    df["_cluster"] = kmeans.fit_predict(coords)

    # Sample evenly from each cluster — use dict to guarantee no duplicates
    per_cluster = max(1, test_size // n_clusters)
    test_indices: dict[int, None] = {}
    for cluster_id in range(n_clusters):
        cluster_idx = df.index[df["_cluster"] == cluster_id].tolist()
        rng.shuffle(cluster_idx)
        for idx in cluster_idx[:per_cluster]:
            test_indices[idx] = None

    # If we're short, top up randomly from remaining (never re-add existing)
    remaining = [i for i in df.index if i not in test_indices]
    rng.shuffle(remaining)
    while len(test_indices) < test_size and remaining:
        test_indices[remaining.pop(0)] = None

    # If we overshot, trim randomly
    if len(test_indices) > test_size:
        keys = list(test_indices.keys())
        rng.shuffle(keys)
        test_indices = {k: None for k in keys[:test_size]}

    test_idx_set = set(test_indices.keys())
    train_idx = [i for i in df.index if i not in test_idx_set]

    test_df = df.loc[list(test_indices.keys())].drop(columns=["_cluster"]).reset_index(drop=True)
    train_df = df.loc[train_idx].drop(columns=["_cluster"]).reset_index(drop=True)

    return train_df, test_df


def print_split_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print(f"\n── Split summary ────────────────────────────────")
    print(f"  Train: {len(train_df)} pairs")
    print(f"  Test:  {len(test_df)} pairs")
    for name, df in [("Train", train_df), ("Test", test_df)]:
        print(f"\n  {name} geographic coverage:")
        print(f"    lat  [{df['day_lat'].min():.5f}, {df['day_lat'].max():.5f}]")
        print(f"    lon  [{df['day_lon'].min():.5f}, {df['day_lon'].max():.5f}]")


def main(test_size: int, n_clusters: int, seed: int) -> None:
    df = load_valid_pairs(FINETUNE_CSV)

    if len(df) < test_size + 50:
        raise ValueError(
            f"Only {len(df)} valid pairs found — not enough to reserve {test_size} for test. "
            "Check your CSV or reduce --test-size."
        )

    train_df, test_df = geographic_test_split(df, test_size, n_clusters, seed)
    print_split_summary(train_df, test_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train_split.csv"
    test_path = OUT_DIR / "test_split.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n  Saved train split → {train_path}")
    print(f"  Saved test split  → {test_path}")
    print("\n  !! Do not re-run this script. test_split.csv is locked for final evaluation only. !!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create geographically spread train/test splits.")
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS,
                        help="Number of geographic clusters for spread sampling.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.test_size, args.n_clusters, args.seed)