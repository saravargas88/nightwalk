"""
Samples ~100k day images from washington-square.csv, excluding any images
already matched to night photos (reserved for brightness regression training).

Outputs:
  dino_training_sample.csv  — sampled rows from washington-square.csv
  dino_training_paths.txt   — one image path per line (for HPC rsync/copy)
"""

import csv
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DAY_CSV        = "urban-mosaic/washington-square.csv"
MATCH_CSVS     = ["all-matches.csv", "matches_sara.csv"]
IMAGE_ROOT     = Path("urban-mosaic/washington-square")
OUTPUT_CSV     = "dino_training_sample.csv"
OUTPUT_PATHS   = "dino_training_paths.txt"
N_SAMPLES      = 100_000
RANDOM_SEED    = 42

# ── Collect reserved day images ───────────────────────────────────────────────
reserved = set()
for match_csv in MATCH_CSVS:
    try:
        with open(match_csv) as f:
            for row in csv.DictReader(f):
                img = row.get("day_image", "").strip()
                if img and img not in ("", "None"):
                    reserved.add(img)
    except FileNotFoundError:
        print(f"Warning: {match_csv} not found, skipping.")

print(f"Reserved (excluded) day images: {len(reserved)}")

# ── Load all day image rows, filter out reserved ──────────────────────────────
eligible = []
with open(DAY_CSV) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        img = row.get("image", "").strip()
        if img and img not in reserved:
            eligible.append(row)

print(f"Eligible day images: {len(eligible)}")

# ── Sample ────────────────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
n = min(N_SAMPLES, len(eligible))
sample = random.sample(eligible, n)
print(f"Sampled: {n}")

# ── Write output CSV ──────────────────────────────────────────────────────────
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sample)

# ── Write paths file ──────────────────────────────────────────────────────────
missing = 0
with open(OUTPUT_PATHS, "w") as f:
    for row in sample:
        p = IMAGE_ROOT / row["image"].strip()
        f.write(str(p) + "\n")
        if not p.exists():
            missing += 1

print(f"Written: {OUTPUT_CSV}, {OUTPUT_PATHS}")
if missing:
    print(f"Warning: {missing} image files not found on disk (paths still written)")
