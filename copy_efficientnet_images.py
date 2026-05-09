"""
Copies 50k spatially-diverse day images (morning/afternoon only) from
urban-mosaic/washington-square into efficientnet-training-images/.
Excludes any image already in all-matches.csv.

Spatial deduplication: rounds lat/lon to a grid (~11m cells) and caps
the number of images taken from each cell, so stopped-car clusters
don't dominate the sample.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path
from PIL import Image

DAY_CSV      = "urban-mosaic/washington-square.csv"
MATCH_CSV    = "all-matches.csv"
IMAGE_ROOT   = Path("urban-mosaic/washington-square")
OUT_DIR      = Path("eff-training-upload")
N_SAMPLES    = 5_000
MAX_PER_CELL = 20      # max images per ~11m cell
GRID_SIZE    = 4       # decimal places → ~11m precision
RANDOM_SEED  = 42
DAY_PERIODS  = {"morning", "afternoon"}

# ── Collect reserved images ───────────────────────────────────────────────────
reserved = set()
with open(MATCH_CSV) as f:
    for row in csv.DictReader(f):
        img = row.get("day_image", "").strip()
        if img and img not in ("", "None"):
            reserved.add(img)
print(f"Excluded (in all-matches.csv): {len(reserved)}")

# ── Load eligible rows with lat/lon ──────────────────────────────────────────
eligible = []
with open(DAY_CSV) as f:
    for row in csv.DictReader(f):
        if row.get("period") not in DAY_PERIODS:
            continue
        img = row["image"].strip()
        if img in reserved:
            continue
        try:
            lat = round(float(row["lat"]), GRID_SIZE)
            lon = round(float(row["lon"]), GRID_SIZE)
        except (ValueError, KeyError):
            continue
        eligible.append((img, (lat, lon)))

print(f"Eligible day images: {len(eligible)}")

# ── Shuffle then cap per spatial cell ────────────────────────────────────────
random.seed(RANDOM_SEED)
random.shuffle(eligible)

cell_counts = defaultdict(int)
pool = []
for img, cell in eligible:
    if cell_counts[cell] < MAX_PER_CELL:
        pool.append(img)
        cell_counts[cell] += 1

print(f"After spatial dedup (max {MAX_PER_CELL}/cell): {len(pool)} images")

# ── Sample from the deduplicated pool ────────────────────────────────────────
sample = random.sample(pool, min(N_SAMPLES, len(pool)))
print(f"Sampled: {len(sample)}")

# ── Copy images ───────────────────────────────────────────────────────────────
OUT_DIR.mkdir(exist_ok=True)
missing = 0
for i, img_path in enumerate(sample, 1):
    src = IMAGE_ROOT / img_path
    dst = OUT_DIR / src.name
    if src.exists():
        try:
            img = Image.open(src)
            img.thumbnail((1024, 1024), Image.LANCZOS)
            img.save(dst, quality=85)
        except (TimeoutError, OSError):
            missing += 1
    else:
        missing += 1
    if i % 500 == 0:
        print(f"  {i}/{len(sample)} copied...")

print(f"\nDone. Copied to {OUT_DIR}/")
if missing:
    print(f"Warning: {missing} files not found on disk")
