# -*- coding: utf-8 -*-
"""resize_for_hpc.py

Resizes only the day images used in train/test splits to 224x224 and saves
them into a flat output directory ready to zip and upload to HPC.

The model resizes to 224 during training anyway, so nothing is lost.

Usage:
    python resize_for_hpc.py --out ~/nightwalk-images-224
    python resize_for_hpc.py --out ~/nightwalk-images-224 --quality 85
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "splits" / "train_split.csv"
TEST_CSV  = ROOT / "splits" / "test_split.csv"
DAY_IMAGE_ROOT = ROOT / "urban-mosaic" / "washington-square"

IMG_SIZE = 224
DEFAULT_QUALITY = 85


def short_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:10]


def flat_name(rel):
    stem = Path(rel).stem
    ext  = Path(rel).suffix
    return "{}_{}{}".format(stem, short_hash(rel), ext)


def read_day_images(csv_path):
    images = []
    with open(str(csv_path), newline="") as f:
        for row in csv.DictReader(f):
            di = row.get("day_image", "").strip()
            if di:
                images.append(di)
    return images


def run(out_dir, quality):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_imgs = read_day_images(TRAIN_CSV)
    test_imgs  = read_day_images(TEST_CSV)
    all_imgs   = sorted(set(train_imgs + test_imgs))
    print("Unique images to resize: {}".format(len(all_imgs)))

    ok, missing, failed = 0, 0, 0
    name_map = []

    for rel in all_imgs:
        src = DAY_IMAGE_ROOT / rel
        if not src.exists():
            missing += 1
            continue

        dst_name = flat_name(rel)
        dst = out_dir / dst_name

        try:
            img = Image.open(str(src)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(str(dst), "JPEG", quality=quality)
            name_map.append((rel, dst_name))
            ok += 1
        except Exception as e:
            print("  FAILED {}: {}".format(rel, e))
            failed += 1

        if ok % 100 == 0 and ok > 0:
            print("  Resized {}/{}...".format(ok, len(all_imgs)))

    # Write name mapping CSV so the training scripts know the new filenames
    map_path = out_dir / "filename_map.csv"
    with open(str(map_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original_day_image", "resized_filename"])
        writer.writerows(name_map)

    total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.jpg"))
    print("\nDone.")
    print("  Resized: {}  Missing: {}  Failed: {}".format(ok, missing, failed))
    print("  Total size: {:.1f} MB".format(total_bytes / 1024 / 1024))
    print("  Name map: {}".format(map_path))
    print("\nNext steps:")
    print("  zip -r nightwalk-images-224.zip {}".format(out_dir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="nightwalk-images-224",
                        help="Output directory for resized images")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY,
                        help="JPEG quality 1-95 (default 85)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.out, args.quality)
