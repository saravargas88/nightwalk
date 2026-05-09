"""prepare_splits.py
Produces two clean, non-overlapping image manifests:
  - splits/efficientnet_train_images.csv   (DINO pretraining candidates)
  - splits/finetune_pairs.csv              (day-night pairs, held out)

Usage:
  python prepare_splits.py [--n 50000]

Location diversity is achieved by:
  1. Binning snapped_lat/lon into ~50m grid cells
  2. Capping images per (grid_cell, android_id) stratum to avoid one route dominating
  3. Proportional downsampling to TARGET_N while preserving geographic spread
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG_CSV     = Path("urban-mosaic/washington-square.csv")
ALL_MATCHES_CSV = Path("all-matches.csv")
IMAGE_ROOT      = Path("urban-mosaic/washington-square")

OUT_DIR         = Path("splits")
OUT_TRAIN       = OUT_DIR / "efficientnet_train_images.csv"
OUT_PAIRS       = OUT_DIR / "finetune_pairs.csv"

# Spatial grid resolution in degrees (~50m at NYC latitudes)
LAT_BIN_SIZE    = 0.0005
LON_BIN_SIZE    = 0.0005

# Max images per (grid_cell, android_id) stratum before global downsampling.
# Keeps no single vehicle route dominant within a cell.
MAX_PER_STRATUM = 3

# Only use daytime images for EfficientNet pretraining
DAYTIME_PERIODS = {"morning", "afternoon"}

# Column names in all-matches.csv that hold image paths
DAY_COL         = "day_image"
NIGHT_COL       = "night_image"

RANDOM_SEED     = 42

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--n", type=int, default=50_000,
    help="Target number of images for EfficientNet pretraining (default: 50000)"
)
parser.add_argument(
    "--no-disk-check", action="store_true",
    help="Skip checking whether each image exists on disk (faster, use if images are remote)"
)
args = parser.parse_args()

TARGET_N = args.n

# ── Load catalog ──────────────────────────────────────────────────────────────
print(f"Loading catalog: {CATALOG_CSV}")
df = pd.read_csv(CATALOG_CSV)
df["image"] = df["image"].str.strip()
print(f"  Total rows: {len(df):,}")

# ── Load and save finetune pairs ──────────────────────────────────────────────
print(f"\nLoading all-matches: {ALL_MATCHES_CSV}")
matches = pd.read_csv(ALL_MATCHES_CSV)
protected_day_images = set(matches[DAY_COL].str.strip())
print(f"  Protected daytime images: {len(protected_day_images):,}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
matches.to_csv(OUT_PAIRS, index=False)
print(f"  Saved finetune pairs → {OUT_PAIRS}")

# ── Filter to valid daytime images ────────────────────────────────────────────
day_df = df[df["period"].isin(DAYTIME_PERIODS)].copy()
print(f"\nDaytime images in catalog: {len(day_df):,}")

if not args.no_disk_check:
    print("  Checking images exist on disk (use --no-disk-check to skip)...")
    day_df = day_df[
        day_df["image"].apply(lambda p: (IMAGE_ROOT / p).exists())
    ].copy()
    print(f"  Images found on disk: {len(day_df):,}")

# Exclude protected images
day_df = day_df[~day_df["image"].isin(protected_day_images)].copy()
print(f"  After excluding paired images: {len(day_df):,}")

if len(day_df) == 0:
    raise RuntimeError("No candidate images remain after filtering. Check your paths and column names.")

# ── Spatial binning ───────────────────────────────────────────────────────────
before = len(day_df)
day_df = day_df.dropna(subset=["snapped_lat", "snapped_lon"]).copy()
day_df = day_df[np.isfinite(day_df["snapped_lat"]) & np.isfinite(day_df["snapped_lon"])].copy()
print(f"  Dropped {before - len(day_df):,} rows with missing/invalid coordinates")

day_df["lat_bin"]   = (day_df["snapped_lat"] / LAT_BIN_SIZE).round().astype(int)
day_df["lon_bin"]   = (day_df["snapped_lon"] / LON_BIN_SIZE).round().astype(int)
day_df["grid_cell"] = day_df["lat_bin"].astype(str) + "_" + day_df["lon_bin"].astype(str)
print(f"\nSpatial grid cells: {day_df['grid_cell'].nunique():,}")
print(f"Android IDs (vehicle routes): {day_df['android_id'].nunique():,}")

# ── Stage 1: stratified sample by (grid_cell, android_id) ────────────────────
# Caps images per route-per-cell so no single driver dominates a location
print(f"\nStage 1: capping at {MAX_PER_STRATUM} images per (grid_cell, android_id)...")
sampled = (
    day_df
    .groupby(["grid_cell", "android_id"], group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), MAX_PER_STRATUM), random_state=RANDOM_SEED))
    .reset_index(drop=True)
)
print(f"  After stage 1: {len(sampled):,} images across {sampled['grid_cell'].nunique():,} cells")

# ── Stage 2: downsample to TARGET_N, preserving geographic spread ─────────────
if len(sampled) > TARGET_N:
    print(f"\nStage 2: downsampling to {TARGET_N:,} (proportional per grid cell)...")
    frac = TARGET_N / len(sampled)
    sampled = (
        sampled
        .groupby("grid_cell", group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )
    # frac-based sampling is approximate — trim or pad to exact count
    if len(sampled) > TARGET_N:
        sampled = sampled.sample(n=TARGET_N, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  After stage 2: {len(sampled):,} images")
else:
    print(f"\nStage 2: pool ({len(sampled):,}) is already under target ({TARGET_N:,}), no downsampling needed.")

# ── Integrity check ───────────────────────────────────────────────────────────
overlap = set(sampled["image"]) & protected_day_images
assert len(overlap) == 0, f"INTEGRITY FAIL: {len(overlap)} images appear in both splits"
print("\n✓ No overlap between pretraining and finetune splits")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nFinal sample: {len(sampled):,} images")
print(f"Grid cells covered: {sampled['grid_cell'].nunique():,}")
print(f"Android IDs covered: {sampled['android_id'].nunique():,}")
print(f"\nImages per neighbourhood:")
print(sampled["neighbourhood"].value_counts().to_string())
print(f"\nImages per period:")
print(sampled["period"].value_counts().to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
out_cols = [
    "image", "grid_cell", "neighbourhood", "borough",
    "android_id", "snapped_lat", "snapped_lon", "taken_on", "period"
]
sampled[out_cols].to_csv(OUT_TRAIN, index=False)
print(f"\nSaved EfficientNet training manifest → {OUT_TRAIN}  ({len(sampled):,} images)")
