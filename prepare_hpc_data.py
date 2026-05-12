"""prepare_hpc_data.py

Packages the minimal set of files needed to run training on HPC into
a single output directory, avoiding uploading the full 25GB urban-mosaic.

What gets copied:
  1. Day images referenced in train_split.csv + test_split.csv (~950 files, ~240MB)
  2. Optionally: N additional day images for SSL pretraining sampled from
     the full urban-mosaic pool (--ssl-sample N)
  3. All CSVs needed at runtime (splits, brightness metrics)
  4. Python training scripts

Output layout:
  <out_dir>/
    day_images/          ← flat copy of all needed day images, renamed to avoid
                            path collisions (uses a hash of original path)
    splits/
      train_split.csv    ← updated with new flat image paths
      test_split.csv     ← updated with new flat image paths
    brightnessmetricexperiments/experiment_outputs/paired_dataset_with_brightness.csv
    model-training/      ← all .py training scripts + existing .pt checkpoints

Usage:
    # Minimum for finetune + linear probe
    python prepare_hpc_data.py --out /tmp/nightwalk-hpc

    # Include 5000 extra images for SSL pretraining
    python prepare_hpc_data.py --out /tmp/nightwalk-hpc --ssl-sample 5000

    # Check sizes without copying
    python prepare_hpc_data.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "splits" / "train_split.csv"
TEST_CSV  = ROOT / "splits" / "test_split.csv"
BRIGHTNESS_CSV = (
    ROOT / "brightnessmetricexperiments"
    / "experiment_outputs"
    / "paired_dataset_with_brightness.csv"
)
DAY_IMAGE_ROOT = ROOT / "urban-mosaic" / "washington-square"
MODEL_TRAINING_DIR = ROOT / "model-training"

RANDOM_SEED = 42


def short_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:10]


def flat_name(original_rel: str) -> str:
    """Turn a deep relative path into a flat unique filename."""
    stem = Path(original_rel).stem
    ext  = Path(original_rel).suffix
    return f"{stem}_{short_hash(original_rel)}{ext}"


def read_split_day_images(csv_path: Path) -> list[str]:
    """Return list of day_image relative paths from a split CSV."""
    images = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            di = row.get("day_image", "").strip()
            if di:
                images.append(di)
    return images


def rewrite_split_csv(src_csv: Path, dst_csv: Path, remap: dict[str, str]) -> None:
    """Write a new split CSV with day_image paths replaced by flat names."""
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with src_csv.open(newline="") as fin, dst_csv.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            di = row.get("day_image", "").strip()
            if di and di in remap:
                row["day_image"] = remap[di]
            writer.writerow(row)


def rewrite_brightness_csv(src: Path, dst: Path, remap: dict[str, str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(newline="") as fin, dst.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            di = row.get("day_image", "").strip()
            if di and di in remap:
                row["day_image"] = remap[di]
            writer.writerow(row)


def collect_ssl_sample(
    exclude: set[str], n: int, seed: int
) -> list[Path]:
    """Sample N additional day images from urban-mosaic not already in the split."""
    rng = random.Random(seed)
    all_imgs = list(DAY_IMAGE_ROOT.rglob("*.jpg"))
    candidates = [
        p for p in all_imgs
        if p.relative_to(DAY_IMAGE_ROOT).as_posix() not in exclude
        and not p.name.startswith("._")
    ]
    rng.shuffle(candidates)
    return candidates[:n]


def fmt_size(n_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"


def run(out_dir: Path, ssl_sample: int, dry_run: bool) -> None:
    print(f"Output directory: {out_dir}")
    print(f"Dry run: {dry_run}\n")

    # ── Collect needed day images from splits ─────────────────────────────────
    train_images = read_split_day_images(TRAIN_CSV)
    test_images  = read_split_day_images(TEST_CSV)
    split_images = sorted(set(train_images + test_images))
    print(f"Unique day images in splits: {len(split_images)}")

    # Check which actually exist
    missing = []
    existing = []
    for rel in split_images:
        p = DAY_IMAGE_ROOT / rel
        if p.exists():
            existing.append(rel)
        else:
            missing.append(rel)
    if missing:
        print(f"  WARNING: {len(missing)} referenced images not found on disk")
    print(f"  Found on disk: {len(existing)}")

    total_bytes = sum((DAY_IMAGE_ROOT / r).stat().st_size for r in existing)
    print(f"  Size: {fmt_size(total_bytes)}")

    # ── Build flat name remap ─────────────────────────────────────────────────
    remap: dict[str, str] = {rel: flat_name(rel) for rel in existing}

    # ── SSL sample ────────────────────────────────────────────────────────────
    ssl_paths: list[Path] = []
    if ssl_sample > 0:
        print(f"\nSampling {ssl_sample} additional images for SSL pretraining...")
        ssl_paths = collect_ssl_sample(set(split_images), ssl_sample, RANDOM_SEED)
        ssl_bytes = sum(p.stat().st_size for p in ssl_paths)
        print(f"  Sampled: {len(ssl_paths)}  Size: {fmt_size(ssl_bytes)}")
        total_bytes += ssl_bytes

    print(f"\nTotal image data: {fmt_size(total_bytes)}")

    if dry_run:
        print("\n[dry-run] No files copied.")
        return

    # ── Copy split day images ─────────────────────────────────────────────────
    day_out = out_dir / "day_images"
    day_out.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying {len(existing)} split day images...")
    for i, rel in enumerate(existing, 1):
        src = DAY_IMAGE_ROOT / rel
        dst = day_out / remap[rel]
        if not dst.exists():
            shutil.copy2(src, dst)
        if i % 100 == 0:
            print(f"  {i}/{len(existing)}")

    # ── Copy SSL images ───────────────────────────────────────────────────────
    if ssl_paths:
        ssl_out = out_dir / "ssl_images"
        ssl_out.mkdir(parents=True, exist_ok=True)
        ssl_list_path = out_dir / "ssl_image_list.txt"
        print(f"\nCopying {len(ssl_paths)} SSL images...")
        ssl_names = []
        for i, src in enumerate(ssl_paths, 1):
            dst = ssl_out / src.name
            if dst.exists():
                dst = ssl_out / flat_name(src.relative_to(DAY_IMAGE_ROOT).as_posix())
            if not dst.exists():
                shutil.copy2(src, dst)
            ssl_names.append(dst.name)
            if i % 500 == 0:
                print(f"  {i}/{len(ssl_paths)}")
        ssl_list_path.write_text("\n".join(ssl_names))
        print(f"  SSL image list → {ssl_list_path}")

    # ── Rewrite CSVs with flat paths ──────────────────────────────────────────
    print("\nRewriting CSVs with updated image paths...")
    rewrite_split_csv(TRAIN_CSV, out_dir / "splits" / "train_split.csv", remap)
    rewrite_split_csv(TEST_CSV,  out_dir / "splits" / "test_split.csv",  remap)
    rewrite_brightness_csv(
        BRIGHTNESS_CSV,
        out_dir / "brightnessmetricexperiments" / "experiment_outputs" / "paired_dataset_with_brightness.csv",
        remap,
    )
    print("  Done.")

    # ── Copy training scripts + checkpoints ───────────────────────────────────
    print("\nCopying model-training scripts...")
    mt_out = out_dir / "model-training"
    mt_out.mkdir(parents=True, exist_ok=True)
    for f in MODEL_TRAINING_DIR.glob("*.py"):
        shutil.copy2(f, mt_out / f.name)
    for f in MODEL_TRAINING_DIR.glob("*.pt"):
        print(f"  Checkpoint: {f.name} ({fmt_size(f.stat().st_size)})")
        shutil.copy2(f, mt_out / f.name)

    # Copy ssl pretrain checkpoint if it exists
    ssl_ckpt = MODEL_TRAINING_DIR / "ssl-pretrain" / "best_ssl_backbone.pt"
    if ssl_ckpt.exists():
        (mt_out / "ssl-pretrain").mkdir(exist_ok=True)
        shutil.copy2(ssl_ckpt, mt_out / "ssl-pretrain" / ssl_ckpt.name)
        print(f"  SSL checkpoint: {fmt_size(ssl_ckpt.stat().st_size)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_out = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"\n{'='*50}")
    print(f"Output directory: {out_dir}")
    print(f"Total size on disk: {fmt_size(total_out)}")
    print(f"\nOn HPC, point DAY_IMAGE_ROOT to: {day_out}")
    print("The rewritten CSVs use flat filenames matching that directory.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package NightWalk data for HPC upload.")
    parser.add_argument("--out", type=Path, default=Path("/tmp/nightwalk-hpc"),
                        help="Output directory to write packaged data into.")
    parser.add_argument("--ssl-sample", type=int, default=0,
                        help="Number of extra images to sample for SSL pretraining (0 = skip).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report sizes without copying any files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(out_dir=args.out, ssl_sample=args.ssl_sample, dry_run=args.dry_run)
