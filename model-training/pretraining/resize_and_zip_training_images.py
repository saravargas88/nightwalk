# -*- coding: utf-8 -*-
"""resize_and_zip_training_images.py

Resizes all images in urban-mosaic/washington-square/ to 224x224, saves them
to a flat output directory with a filename_map.csv, then zips everything for
HPC upload.

Output:
  model-training/pretraining/eff-training-224/   (flat resized images + filename_map.csv)
  model-training/pretraining/eff-training-224.zip

The zip can be uploaded to HPC and used with train_efficientnet_multihead_v2.py:
  python3 model-training/pretraining/train_efficientnet_multihead_v2.py \
      --image-dir pretraining/eff-training-224 \
      --save-path model-training/best_efficientnet_multihead_v2.pt

Run:
  python3 model-training/pretraining/resize_and_zip_training_images.py
  python3 model-training/pretraining/resize_and_zip_training_images.py --no-zip
"""

import argparse
import csv
import hashlib
import zipfile
from pathlib import Path

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Config ────────────────────────────────────────────────────────────────────
_PRETRAINING  = Path(__file__).resolve().parent
_MODEL_TRAINING = _PRETRAINING.parent
_ROOT         = _MODEL_TRAINING.parent

SOURCE_DIR  = _ROOT / "urban-mosaic" / "washington-square"
OUT_DIR     = _PRETRAINING / "eff-training-224"
ZIP_PATH    = _PRETRAINING / "eff-training-224.zip"
MAP_FILE    = OUT_DIR / "filename_map.csv"

IMG_SIZE    = 224
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def iter_images(source_dir: Path):
    """Yield all image paths, skipping macOS metadata files."""
    for p in sorted(source_dir.rglob("*")):
        if p.name.startswith("._"):
            continue
        if p.suffix in IMG_EXTS and p.is_file():
            yield p


def flat_name(source_dir: Path, img_path: Path) -> str:
    """
    Build a collision-free flat filename from the relative path.
    Strategy: replace path separators with '__', then append a 6-char hash
    of the full relative path to handle any edge-case collisions.
    """
    rel = img_path.relative_to(source_dir)
    rel_str = str(rel).replace("/", "__").replace("\\", "__")
    h = hashlib.md5(str(rel).encode()).hexdigest()[:6]
    stem, suffix = rel_str.rsplit(".", 1) if "." in rel_str else (rel_str, "jpg")
    return f"{stem}__{h}.{suffix}"


def resize_images(source_dir: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = 0
    total = 0

    all_imgs = list(iter_images(source_dir))
    print(f"Found {len(all_imgs)} images in {source_dir}")

    for i, img_path in enumerate(all_imgs, 1):
        rel = str(img_path.relative_to(source_dir))
        flat = flat_name(source_dir, img_path)
        out_path = out_dir / flat

        if (i % 5000) == 0:
            print(f"  {i}/{len(all_imgs)}  skipped={skipped}")

        if out_path.exists():
            rows.append({"original_day_image": rel, "resized_filename": flat})
            total += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(out_path, "JPEG", quality=90)
            rows.append({"original_day_image": rel, "resized_filename": flat})
            total += 1
        except Exception as e:
            print(f"  SKIP {img_path.name}: {e}")
            skipped += 1

    print(f"Done: {total} resized, {skipped} skipped")
    return rows


def write_map(rows: list[dict], map_file: Path) -> None:
    with map_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original_day_image", "resized_filename"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote filename_map.csv ({len(rows)} entries) -> {map_file}")


def zip_directory(out_dir: Path, zip_path: Path) -> None:
    files = sorted(out_dir.iterdir())
    print(f"Zipping {len(files)} files -> {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for i, f in enumerate(files, 1):
            zf.write(f, arcname=f"eff-training-224/{f.name}")
            if (i % 10000) == 0:
                print(f"  zipped {i}/{len(files)}")
    size_gb = zip_path.stat().st_size / 1e9
    print(f"Zip complete: {zip_path}  ({size_gb:.2f} GB)")


def main(do_zip: bool) -> None:
    if not SOURCE_DIR.exists():
        print(f"ERROR: source directory not found: {SOURCE_DIR}")
        return

    rows = resize_images(SOURCE_DIR, OUT_DIR)
    write_map(rows, MAP_FILE)

    if do_zip:
        zip_directory(OUT_DIR, ZIP_PATH)
        print(f"\nUpload {ZIP_PATH} to HPC, then:")
        print(f"  unzip eff-training-224.zip -d pretraining/")
    else:
        print(f"\nSkipped zip. Run without --no-zip to create {ZIP_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-zip", action="store_true",
                        help="Resize images but skip creating the zip archive")
    args = parser.parse_args()
    main(do_zip=not args.no_zip)
