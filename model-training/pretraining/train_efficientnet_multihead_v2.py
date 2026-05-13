# -*- coding: utf-8 -*-
"""train_efficientnet_multihead_v2.py

Improved EfficientNet-B0 pretraining on DINO labels with bounding box area targets.

Key improvements over v1:
  1. Adds bbox_area_sum targets alongside raw counts — forces backbone to learn
     feature prominence (a large storefront vs a distant one) not just presence.
  2. log1p-normalizes all targets before training — counts and bbox areas are
     heavily right-skewed; log1p compresses the long tail without losing signal.
  3. RandomResizedCrop augmentation — exposes the backbone to partial views of
     the same scene, improving spatial generalization.
  4. Gradient clipping — prevents the occasional exploding gradient from
     destabilizing the shared backbone.
  5. Accepts a flat 224px image directory (with filename_map.csv) for faster
     disk I/O on HPC. Falls back to nested original directory if not provided.

Targets trained (6 total):
  tree                    raw DINO detection count
  streetlight             raw DINO detection count
  storefront              raw DINO detection count
  bbox_area_sum_tree      total pixel area of tree detections
  bbox_area_sum_streetlight  total pixel area of streetlight detections
  bbox_area_sum_storefront   total pixel area of storefront detections

  NOTE: verify these column names match your 13k DINO CSV by running:
    python3 -c "import pandas as pd; print(pd.read_csv('model-training/dino_labels/13k-sample-all.csv').columns.tolist())"

Run:
  python3 model-training/pretraining/train_efficientnet_multihead_v2.py
  python3 model-training/pretraining/train_efficientnet_multihead_v2.py \
      --image-dir pretraining/eff-training-224 \
      --save-path model-training/best_efficientnet_multihead_v2.pt
"""

import argparse
import csv
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Config ────────────────────────────────────────────────────────────────────
_MODEL_TRAINING = Path(__file__).resolve().parent.parent
_ROOT = _MODEL_TRAINING.parent

DINO_CSV    = _MODEL_TRAINING / "dino_labels" / "13k-sample-all.csv"
IMAGE_DIR   = _ROOT / "urban-mosaic" / "washington-square"  # nested fallback
SAVE_PATH   = _MODEL_TRAINING / "best_efficientnet_multihead_v2.pt"
PREDS_DIR   = _MODEL_TRAINING / "val_predictions_multihead_v2"
IMAGE_COL   = "image"

# Count targets + bbox area targets + centroid targets
COUNT_TARGETS    = ["tree", "streetlight", "storefront"]
BBOX_TARGETS     = ["bbox_area_sum_tree", "bbox_area_sum_streetlight", "bbox_area_sum_storefront"]
CENTROID_TARGETS = [
    "bbox_cx_tree",        "bbox_cy_tree",
    "bbox_cx_streetlight", "bbox_cy_streetlight",
    "bbox_cx_storefront",  "bbox_cy_storefront",
]
ALL_POSSIBLE_TARGETS = COUNT_TARGETS + BBOX_TARGETS + CENTROID_TARGETS
# Actual targets used are resolved at runtime based on which columns exist in the CSV
TARGETS = ALL_POSSIBLE_TARGETS

N_SAMPLES    = None
NUM_EPOCHS   = 100
BATCH_SIZE   = 64
LR           = 1e-4
WEIGHT_DECAY = 1e-3
IMG_SIZE     = 224
NUM_WORKERS  = 8
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42
SAVE_PREDS_EVERY = 10
GRAD_CLIP    = 1.0   # max gradient norm

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ── Filename map (flat 224 directory) ─────────────────────────────────────────
def load_filename_map(image_dir: Path) -> dict[str, str]:
    map_path = image_dir / "filename_map.csv"
    if not map_path.exists():
        return {}
    remap: dict[str, str] = {}
    with map_path.open(newline="") as f:
        for row in csv.DictReader(f):
            remap[row["original_day_image"]] = row["resized_filename"]
    return remap


# ── Dataset ───────────────────────────────────────────────────────────────────
class CountDataset(Dataset):
    def __init__(self, df, image_dir, filename_map, targets, transform=None):
        self.df           = df.reset_index(drop=True)
        self.image_dir    = Path(image_dir)
        self.filename_map = filename_map
        self.targets      = targets
        self.transform    = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        original = row[IMAGE_COL]
        flat = self.filename_map.get(original, original)
        img_path = self.image_dir / flat

        try:
            image = Image.open(img_path).convert("RGB")
            image.load()
        except Exception as e:
            print(f"  Warning: skipping {img_path}: {e}")
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            try:
                image = self.transform(image)
            except Exception:
                image = torch.zeros((3, IMG_SIZE, IMG_SIZE))

        labels = torch.tensor(row[self.targets].values.astype(np.float32))
        return image, labels


# ── Transforms ────────────────────────────────────────────────────────────────
# RandomResizedCrop: exposes backbone to partial scene views, improves spatial generalization
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model ─────────────────────────────────────────────────────────────────────
class EfficientNetMultiHead(nn.Module):
    def __init__(self, targets: list):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = base.classifier[1].in_features  # 1280

        self.backbone = base.features
        self.pool = base.avgpool

        self.heads = nn.ModuleDict({
            t: nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, 1),
                nn.ReLU(),  # outputs are non-negative
            )
            for t in targets
        })
        self._targets = targets

    def forward(self, x):
        x = torch.flatten(self.pool(self.backbone(x)), 1)
        return torch.stack([self.heads[t](x).squeeze(1) for t in self._targets], dim=1)


# ── log1p target normalization ────────────────────────────────────────────────
def normalize_targets(df: pd.DataFrame, targets: list) -> pd.DataFrame:
    """Apply log1p to all targets. Compresses right-skewed counts and bbox areas."""
    df = df.copy()
    df[targets] = np.log1p(df[targets].values)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────
def per_target_mae(preds, labels, targets):
    maes = (preds - labels).abs().mean(dim=0)
    return {t: maes[i].item() for i, t in enumerate(targets)}


def save_predictions(val_df, all_preds, targets, epoch):
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    preds_np = all_preds.numpy()
    out = val_df[[IMAGE_COL] + targets].copy().reset_index(drop=True)
    for i, t in enumerate(targets):
        out[f"pred_{t}"] = preds_np[:, i].round(4)
    out_path = PREDS_DIR / f"val_preds_epoch_{epoch:03d}.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved val predictions to {out_path}")


# ── Training ──────────────────────────────────────────────────────────────────
def train(image_dir: Path, save_path: Path):
    torch.cuda.empty_cache()

    df = pd.read_csv(DINO_CSV)

    # Check all requested target columns exist
    missing_cols = [t for t in TARGETS if t not in df.columns]
    if missing_cols:
        print(f"WARNING: these target columns are missing from the CSV: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        print("Falling back to count targets only.")
        active_targets = [t for t in TARGETS if t in df.columns]
    else:
        active_targets = TARGETS

    df[active_targets] = df[active_targets].fillna(0)

    # log1p-normalize targets before training
    df = normalize_targets(df, active_targets)

    filename_map = load_filename_map(image_dir)

    # Drop rows whose image file doesn't exist
    def img_exists(f):
        flat = filename_map.get(f, f)
        return (image_dir / flat).exists()
    df = df[df[IMAGE_COL].apply(img_exists)].reset_index(drop=True)
    print(f"Images found on disk: {len(df)}")

    if N_SAMPLES and N_SAMPLES < len(df):
        df = df.sample(n=N_SAMPLES, random_state=RANDOM_SEED)

    print("\nTarget stats (after log1p):")
    print(df[active_targets].describe().loc[["mean", "std", "max"]].round(3))
    print("\n% zeros per target:")
    print(((df[active_targets] == 0).mean() * 100).round(1).to_string())
    print()

    train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    train_ds = CountDataset(train_df, image_dir, filename_map, active_targets, train_tf)
    val_ds   = CountDataset(val_df,   image_dir, filename_map, active_targets, val_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE != "cpu"))
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE != "cpu"))

    model = EfficientNetMultiHead(targets=active_targets).to(DEVICE)

    START_EPOCH = 0
    best_val_loss = float("inf")
    saved_optimizer_state = None
    saved_scheduler_state = None
    saved_scaler_state = None

    if save_path.exists():
        print(f"--- Found checkpoint at {save_path} ---")
        ckpt = torch.load(save_path, map_location=DEVICE)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            saved_optimizer_state = ckpt.get("optimizer_state_dict")
            saved_scheduler_state = ckpt.get("scheduler_state_dict")
            saved_scaler_state    = ckpt.get("scaler_state_dict")
            START_EPOCH           = ckpt["epoch"] + 1
            best_val_loss         = ckpt["best_val_loss"]
            print(f"Resuming from epoch {START_EPOCH}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler    = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    criterion = nn.HuberLoss()

    if saved_optimizer_state:
        optimizer.load_state_dict(saved_optimizer_state)
        if saved_scheduler_state: scheduler.load_state_dict(saved_scheduler_state)
        if saved_scaler_state and scaler: scaler.load_state_dict(saved_scaler_state)

    log_path = save_path.parent / "training_log_v2.csv"
    if not log_path.exists() or START_EPOCH == 0:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"] + active_targets)
            writer.writeheader()

    for epoch in range(START_EPOCH, NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for images, labels in train_dl:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    preds = model(images)
                    loss  = criterion(preds, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(images)
                loss  = criterion(preds, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        preds = model(images)
                else:
                    preds = model(images)
                val_loss += criterion(preds, labels).item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        scheduler.step()
        train_loss /= len(train_dl)
        val_loss   /= len(val_dl)

        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        mae = per_target_mae(all_preds, all_labels, active_targets)
        mae_str = "  ".join(f"{t}={v:.3f}" for t, v in mae.items())
        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  MAE: {mae_str}")

        row = {"epoch": epoch + 1, "train_loss": round(train_loss, 6),
               "val_loss": round(val_loss, 6), **{t: round(v, 4) for t, v in mae.items()}}
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_val_loss": best_val_loss,
                "targets": active_targets,
            }, save_path)
            print(f"  Saved checkpoint (epoch {epoch+1})")

        if (epoch + 1) % SAVE_PREDS_EVERY == 0:
            save_predictions(val_df, all_preds, active_targets, epoch + 1)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR,
                        help="Path to image directory (flat 224px or original nested)")
    parser.add_argument("--dino-csv", type=Path, default=DINO_CSV,
                        help="Path to DINO labels CSV (use enriched_dino_labels.csv for bbox/centroid targets)")
    parser.add_argument("--save-path", type=Path, default=SAVE_PATH)
    args = parser.parse_args()
    DINO_CSV = args.dino_csv  # override module-level default
    print(f"Device: {DEVICE}")
    print(f"Image dir: {args.image_dir}")
    print(f"DINO CSV:  {args.dino_csv}")
    print(f"Save path: {args.save_path}")
    train(args.image_dir, args.save_path)
