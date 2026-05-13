# -*- coding: utf-8 -*-
"""prepare_pretraining_data.py

Prepares the 13k EfficientNet pretraining dataset for HPC upload.

Steps:
  1. Reads model-training/dino_labels/13k-sample-all.csv  (counts)
         and model-training/dino_labels/13k-sample-all.json (bounding boxes)
  2. Resizes the corresponding images (urban-mosaic/washington-square/) to 224x224
  3. Saves them flat into model-training/pretraining/eff-training-224/
  4. Writes filename_map.csv  (original nested path -> flat filename)
  5. Writes enriched_dino_labels.csv with:
       image, tree, streetlight, storefront,
       bbox_area_sum_tree, bbox_area_sum_streetlight, bbox_area_sum_storefront,
       bbox_cx_tree, bbox_cy_tree,          (mean normalized centroid x/y)
       bbox_cx_streetlight, bbox_cy_streetlight,
       bbox_cx_storefront, bbox_cy_storefront
  6. Zips eff-training-224/ + enriched_dino_labels.csv into eff-training-224.zip

On HPC:
  unzip eff-training-224.zip -d pretraining/
  python3 model-training/pretraining/train_efficientnet_multihead_v2.py \
      --image-dir pretraining/eff-training-224 \
      --dino-csv  pretraining/eff-training-224/enriched_dino_labels.csv \
      --save-path model-training/best_efficientnet_multihead_v2.pt

Run:
  python3 model-training/pretraining/prepare_pretraining_data.py
  python3 model-training/pretraining/prepare_pretraining_data.py --no-zip
"""

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE           = Path(__file__).resolve().parent          # pretraining/
_MODEL_TRAINING = _HERE.parent
_ROOT           = _MODEL_TRAINING.parent

SOURCE_DIR   = _ROOT / "urban-mosaic" / "washington-square"
DINO_CSV     = _MODEL_TRAINING / "dino_labels" / "13k-sample-all.csv"
DINO_JSON    = _MODEL_TRAINING / "dino_labels" / "13k-sample-all.json"
OUT_DIR      = _HERE / "eff-training-224"
MAP_CSV      = OUT_DIR / "filename_map.csv"
ENRICHED_CSV = OUT_DIR / "enriched_dino_labels.csv"
ZIP_PATH     = _HERE / "eff-training-224.zip"

IMG_SIZE  = 224
IMG_AREA  = IMG_SIZE * IMG_SIZE          # for normalizing bbox areas
CATEGORIES = ["tree", "streetlight", "storefront"]


# ── Flat filename ─────────────────────────────────────────────────────────────
def flat_name(rel: str) -> str:
    safe = rel.replace("/", "__").replace("\\", "__")
    h = hashlib.md5(rel.encode()).hexdigest()[:6]
    if "." in safe:
        stem, ext = safe.rsplit(".", 1)
        return f"{stem}__{h}.{ext}"
    return f"{safe}__{h}.jpg"


# ── Bbox feature extraction ───────────────────────────────────────────────────
def bbox_features(detections: list, img_w: int = 1632, img_h: int = 1224) -> dict:
    """
    From a list of DINO detection dicts, compute per-category:
      - bbox_area_sum   : total pixel area (in 224-normalized space)
      - bbox_cx / bbox_cy : mean centroid (normalized 0-1)
    The raw coordinates are in original image space (default 1024x1024).
    We scale to 224x224 for consistency with the resized images.
    """
    feats = {}
    for cat in CATEGORIES:
        boxes = [d["box"] for d in detections if d.get("category") == cat]
        if not boxes:
            feats[f"bbox_area_sum_{cat}"] = 0.0
            feats[f"bbox_cx_{cat}"]       = 0.0
            feats[f"bbox_cy_{cat}"]       = 0.0
            continue

        areas, cxs, cys = [], [], []
        for box in boxes:
            x1, y1, x2, y2 = box
            # Scale to 224 space
            x1s = x1 / img_w * IMG_SIZE
            y1s = y1 / img_h * IMG_SIZE
            x2s = x2 / img_w * IMG_SIZE
            y2s = y2 / img_h * IMG_SIZE
            w = max(x2s - x1s, 0)
            h = max(y2s - y1s, 0)
            areas.append(w * h)
            cxs.append((x1s + x2s) / 2 / IMG_SIZE)
            cys.append((y1s + y2s) / 2 / IMG_SIZE)

        feats[f"bbox_area_sum_{cat}"] = round(sum(areas), 2)
        feats[f"bbox_cx_{cat}"]       = round(sum(cxs) / len(cxs), 4)
        feats[f"bbox_cy_{cat}"]       = round(sum(cys) / len(cys), 4)

    return feats


# ── Step 1: Load labels ───────────────────────────────────────────────────────
def load_labels() -> tuple[list[dict], dict]:
    with open(DINO_CSV, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    with open(DINO_JSON) as f:
        json_data: dict = json.load(f)
    print(f"CSV rows: {len(csv_rows)}  |  JSON keys: {len(json_data)}")
    return csv_rows, json_data


# ── Step 2 & 3: Resize images + build filename map ────────────────────────────
def resize_images(csv_rows: list[dict]) -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name_map: dict[str, str] = {}  # original -> flat
    skipped = 0

    print(f"Resizing {len(csv_rows)} images to {IMG_SIZE}x{IMG_SIZE} ...")
    for i, row in enumerate(csv_rows, 1):
        rel = row["image"]
        flat = flat_name(rel)
        name_map[rel] = flat
        out_path = OUT_DIR / flat

        if (i % 2000) == 0:
            print(f"  {i}/{len(csv_rows)}  skipped={skipped}")

        if out_path.exists():
            continue

        src = SOURCE_DIR / rel
        try:
            img = Image.open(src).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(out_path, "JPEG", quality=90)
        except Exception as e:
            print(f"  SKIP {src.name}: {e}")
            skipped += 1

    print(f"Done resizing. Skipped: {skipped}")
    return name_map


# ── Step 4: Write filename_map.csv ────────────────────────────────────────────
def write_filename_map(name_map: dict[str, str]) -> None:
    with MAP_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original_day_image", "resized_filename"])
        writer.writeheader()
        for orig, flat in name_map.items():
            writer.writerow({"original_day_image": orig, "resized_filename": flat})
    print(f"Wrote filename_map.csv ({len(name_map)} entries)")


# ── Step 5: Write enriched_dino_labels.csv ────────────────────────────────────
def write_enriched_csv(csv_rows: list[dict], json_data: dict) -> None:
    bbox_cols = (
        [f"bbox_area_sum_{c}" for c in CATEGORIES] +
        [f"bbox_cx_{c}" for c in CATEGORIES] +
        [f"bbox_cy_{c}" for c in CATEGORIES]
    )
    fieldnames = ["image", "tree", "streetlight", "storefront"] + bbox_cols

    missing_json = 0
    with ENRICHED_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            img = row["image"]
            detections = json_data.get(img, [])
            if not detections:
                missing_json += 1
            feats = bbox_features(detections)
            writer.writerow({
                "image":       img,
                "tree":        row.get("tree", 0),
                "streetlight": row.get("streetlight", 0),
                "storefront":  row.get("storefront", 0),
                **feats,
            })

    print(f"Wrote enriched_dino_labels.csv ({len(csv_rows)} rows, {missing_json} with no JSON entry)")


# ── Step 6: Zip ───────────────────────────────────────────────────────────────
def zip_output() -> None:
    files = sorted(OUT_DIR.iterdir())
    print(f"Zipping {len(files)} files -> {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for i, f in enumerate(files, 1):
            zf.write(f, arcname=f"eff-training-224/{f.name}")
            if (i % 3000) == 0:
                print(f"  zipped {i}/{len(files)}")
    size_mb = ZIP_PATH.stat().st_size / 1e6
    print(f"Zip done: {ZIP_PATH}  ({size_mb:.0f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(do_zip: bool) -> None:
    csv_rows, json_data = load_labels()

    name_map = resize_images(csv_rows)
    write_filename_map(name_map)
    write_enriched_csv(csv_rows, json_data)

    if do_zip:
        zip_output()
        print(f"\nUpload {ZIP_PATH.name} to HPC pretraining/ folder, then:")
        print("  unzip eff-training-224.zip -d pretraining/")
        print("  python3 model-training/pretraining/train_efficientnet_multihead_v2.py \\")
        print("      --image-dir pretraining/eff-training-224 \\")
        print("      --dino-csv  pretraining/eff-training-224/enriched_dino_labels.csv \\")
        print("      --save-path model-training/best_efficientnet_multihead_v2.pt")
    else:
        print(f"\n--no-zip: skipping zip. Output is in {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-zip", action="store_true",
                        help="Skip creating the zip archive")
    args = parser.parse_args()
    main(do_zip=not args.no_zip)
