"""build_predictions.py — produce predictions.csv for the route-planning demo.

Output schema (the demo's input contract):

    image_id, lat, lon, heading, predicted_brightness

Modes
-----
--mock         Skip the model; fill predicted_brightness with plausible random values.
               Use while training is still in development, or for unit testing the
               aggregation/routing layers without GPU access.

(default)      Load --checkpoint (default: fold_3 of the imagenet-backbone finetune run)
               and run inference over --source.

Sources
-------
--source train_images  Use splits/efficientnet_train_images.csv (~13k day images, dense
                       GPS coverage of WSP / Greenwich Village). lat/lon from
                       snapped_lat/snapped_lon; heading column will be NaN.
--source test_split    Use splits/test_split.csv (~200 paired day images, has heading).

Predictions are written in the same units the model was trained on (target metric =
gray_mean_zscore — z-score of greyscale mean across the dataset's train split).
The fold_3 checkpoint stores no de-normalisation stats, so the values stay in
that train-z-score space. Downstream aggregation re-normalises to [0, 1] anyway.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "splits"
DEMO_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_OUT = DEMO_DATA / "predictions.csv"

# Where the live training scripts live, post-reorg
EVAL_DIR = ROOT / "model-training" / "eval"
REGRESSION_DIR = ROOT / "model-training" / "regression"

# Default checkpoint — fold_3 of the imagenet-backbone finetune
DEFAULT_CHECKPOINT = (
    ROOT / "model-training" / "finetune-runs" / "imagenet" / "n800" / "fold_3" / "best_model.pt"
)

BATCH_SIZE = 64
NUM_WORKERS = 4


def _auto_image_dir() -> Path | None:
    """Walk up from this file looking for `urban-mosaic/washington-square`.

    Worktrees and the main checkout can live at different depths from the data
    directory; this lets `build_predictions.py` work without --image-dir as long
    as the dataset is somewhere on the path back toward $HOME.
    """
    cur = Path(__file__).resolve().parent
    for _ in range(8):  # cap search depth
        cand = cur / "urban-mosaic" / "washington-square"
        if cand.is_dir():
            return cand
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _load_source(source: str, limit: int | None):
    """Return list of (image_id, lat, lon, heading) tuples from the chosen split."""
    rows = []
    if source == "train_images":
        path = SPLITS / "efficientnet_train_images.csv"
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                try:
                    lat = float(r["snapped_lat"])
                    lon = float(r["snapped_lon"])
                except (KeyError, ValueError):
                    continue
                rows.append((r["image"], lat, lon, None))
    elif source == "test_split":
        path = SPLITS / "test_split.csv"
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                day = (r.get("day_image") or "").strip()
                if not day:
                    continue
                try:
                    lat = float(r["day_lat"])
                    lon = float(r["day_lon"])
                except (KeyError, ValueError):
                    continue
                try:
                    heading = float(r["day_heading"])
                except (KeyError, ValueError):
                    heading = None
                rows.append((day, lat, lon, heading))
    else:
        raise ValueError(f"unknown --source {source!r}")

    if limit is not None:
        rows = rows[:limit]
    return rows


def _mock_predict(rows):
    """Fake predictions with sharp, location-tied structure.

    We bucket each prediction into a coarse grid cell, hash the cell, and use the
    hash as a per-cell brightness offset. That gives the demo well-defined dark
    pockets and bright corridors instead of a smooth gradient, so the routing layer
    actually picks different paths for shortest vs. well-lit.
    """
    rng = random.Random(0)
    # WSP center, approximately
    lat0, lon0 = 40.7308, -73.9973
    out = []
    for image_id, lat, lon, heading in rows:
        # ~50 m grid cells (0.0005° lat ≈ 55 m at this latitude)
        cell_lat = round((lat - lat0) / 0.0005)
        cell_lon = round((lon - lon0) / 0.0005)
        # per-cell offset in [-3, 3] z-score units — deterministic from cell id
        cell_seed = (cell_lat * 1009 + cell_lon * 1013) % 997
        cell_offset = ((cell_seed / 997.0) - 0.5) * 6.0
        # add a coarse bright corridor along east-west arteries (e.g., 8th St): cells
        # near lat0 get a +2 boost
        corridor_boost = 2.0 if abs(cell_lat) <= 1 else 0.0
        noise = rng.gauss(0, 0.5)
        pred = cell_offset + corridor_boost + noise
        out.append(pred)
    return out


def _real_predict(rows, image_dir: Path, checkpoint: Path):
    """Run the trained checkpoint on the given (image_id, lat, lon, heading) rows.

    Expects a `finetune_brightness.py`-style checkpoint:

        {model_state_dict, backbone, metric, image_size}

    The model is `EfficientNetRegressor` (1 output head); predictions come back
    in whatever train-set z-score space the fold was trained on. No de-normalisation.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # Import the model class and the small dataset/name-map helpers.
    sys.path.insert(0, str(REGRESSION_DIR))
    sys.path.insert(0, str(EVAL_DIR))
    from finetune_brightness import EfficientNetRegressor  # noqa: E402
    from eval_brightness_checkpoint import SimpleDataset, load_name_map  # noqa: E402

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[build_predictions] device={device}")

    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    img_size = int(ckpt.get("image_size", 224))
    metric = ckpt.get("metric", "?")
    backbone = ckpt.get("backbone", "?")
    print(f"[build_predictions] checkpoint: backbone={backbone}  metric={metric}  image_size={img_size}")

    name_map = load_name_map(image_dir)

    keep_rows, paths = [], []
    for r in rows:
        image_id = r[0]
        flat = name_map.get(image_id)
        img_path = image_dir / flat if flat else image_dir / image_id
        if not img_path.exists():
            continue
        keep_rows.append(r)
        paths.append(str(img_path))

    print(f"[build_predictions] images resolved on disk: {len(paths)} / {len(rows)}")
    if not paths:
        raise SystemExit(f"No images found on disk under {image_dir}; check the path.")

    model = EfficientNetRegressor().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = SimpleDataset(paths, tf)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    preds = []
    n_done = 0
    n_total = len(paths)
    with torch.no_grad():
        for imgs in dl:
            imgs = imgs.to(device)
            out = model(imgs)              # shape: (batch,) — head already squeezes
            preds.append(out.cpu().numpy())
            n_done += imgs.shape[0]
            if n_done % (BATCH_SIZE * 10) == 0:
                print(f"[build_predictions] {n_done} / {n_total} images …")
    preds = np.concatenate(preds, axis=0)
    return keep_rows, preds.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="generate synthetic predictions; skip model")
    ap.add_argument("--source", choices=["train_images", "test_split"], default="train_images")
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="directory of day images. Auto-detected (walks up looking for "
                         "urban-mosaic/washington-square) unless given explicitly.")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                    help="path to a finetune_brightness.py-style .pt checkpoint")
    ap.add_argument("--limit", type=int, default=None, help="cap number of rows for development")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    DEMO_DATA.mkdir(parents=True, exist_ok=True)
    rows = _load_source(args.source, args.limit)
    print(f"[build_predictions] source={args.source} rows={len(rows)}")

    if args.mock:
        preds = _mock_predict(rows)
        kept = rows
    else:
        image_dir = args.image_dir or _auto_image_dir()
        if image_dir is None:
            raise SystemExit(
                "Could not locate urban-mosaic/washington-square automatically; "
                "pass --image-dir explicitly."
            )
        print(f"[build_predictions] image_dir={image_dir}")
        kept, preds = _real_predict(rows, image_dir, args.checkpoint)

    n_written = 0
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "lat", "lon", "heading", "predicted_brightness"])
        for (image_id, lat, lon, heading), pred in zip(kept, preds):
            if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(pred)):
                continue
            w.writerow([
                image_id,
                f"{lat:.7f}",
                f"{lon:.7f}",
                "" if heading is None else f"{heading:.2f}",
                f"{pred:.4f}",
            ])
            n_written += 1
    print(f"[build_predictions] wrote {n_written} rows → {args.out}")


if __name__ == "__main__":
    main()
