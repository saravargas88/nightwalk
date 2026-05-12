# -*- coding: utf-8 -*-
"""continue_training.py

Loads best_efficientnet_brightness.pt and continues training for more epochs
at a lower learning rate.

Usage:
    python3 model-training/continue_training.py --image-dir nightwalk-images-224
    python3 model-training/continue_training.py --image-dir nightwalk-images-224 --epochs 50 --lr-backbone 1e-5 --lr-head 5e-5
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "splits" / "train_split.csv"
DATA_CSV = ROOT / "brightnessmetricexperiments" / "experiment_outputs" / "paired_dataset_with_brightness.csv"
CHECKPOINT = ROOT / "model-training" / "brightness-regression-run" / "best_efficientnet_brightness.pt"
OUTPUT_DIR = ROOT / "model-training" / "brightness-regression-run"

TARGETS = ["gray_mean", "luma_mean", "value_mean", "gray_mean_zscore"]
IMG_SIZE = 224
BATCH_SIZE = 16
VAL_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 4
SAVE_PRED_EVERY = 10

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class Example:
    image_path: str
    targets: list
    night_photo: str
    day_image: str


class BrightnessDataset(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        image = Image.open(ex.image_path).convert("RGB")
        return self.transform(image), torch.tensor(ex.targets, dtype=torch.float32)


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class EfficientNetBrightnessRegressor(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = base.classifier[1].in_features
        self.features = base.features
        self.avgpool = base.avgpool
        self.head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.head(x)


def load_name_map(image_dir):
    map_path = image_dir / "filename_map.csv"
    if not map_path.exists():
        return {}
    remap = {}
    with map_path.open(newline="") as f:
        for row in csv.DictReader(f):
            remap[row["original_day_image"]] = row["resized_filename"]
    return remap


def load_examples(image_dir):
    name_map = load_name_map(image_dir)
    allowed = {(r["night_photo"], r["day_image"]) for r in csv.DictReader(TRAIN_CSV.open(newline=""))}
    brightness = {(r["night_photo"], r["day_image"]): r for r in csv.DictReader(DATA_CSV.open(newline=""))}

    examples = []
    for key in allowed:
        b = brightness.get(key)
        if b is None:
            continue
        rel = b["day_image"]
        flat = name_map.get(rel)
        day_path = image_dir / flat if flat else image_dir / rel
        if not day_path.exists():
            continue
        examples.append(Example(
            image_path=str(day_path),
            targets=[float(b[t]) for t in TARGETS],
            night_photo=b["night_photo"],
            day_image=rel,
        ))
    return examples


def split_examples(examples):
    rng = random.Random(RANDOM_SEED)
    groups = {}
    for ex in examples:
        groups.setdefault(ex.day_image, []).append(ex)
    day_images = list(groups)
    rng.shuffle(day_images)
    split_idx = int(len(day_images) * (1.0 - VAL_SPLIT))
    train_days = set(day_images[:split_idx])
    train_exs, val_exs = [], []
    for day_image, grouped in groups.items():
        if day_image in train_days:
            train_exs.extend(grouped)
        else:
            val_exs.extend(grouped)
    return train_exs, val_exs


def denormalize(preds, mean, std):
    mean_t = torch.tensor(mean, device=preds.device, dtype=preds.dtype)
    std_t = torch.tensor(std, device=preds.device, dtype=preds.dtype)
    return preds * std_t + mean_t


def per_target_r2(preds, labels):
    out = {}
    for i, target in enumerate(TARGETS):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        denom = torch.sum((y_true - y_true.mean()) ** 2)
        if float(denom.item()) < 1e-8:
            out[target] = 0.0
        else:
            out[target] = float((1.0 - torch.sum((y_true - y_pred) ** 2) / denom).item())
    return out


def run(image_dir, num_epochs, lr_backbone, lr_head, weight_decay):
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint from {CHECKPOINT}")

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    target_mean = np.array(ckpt["target_mean"], dtype=np.float32)
    target_std = np.array(ckpt["target_std"], dtype=np.float32)

    model = EfficientNetBrightnessRegressor(num_outputs=len(TARGETS)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Loaded model weights.")

    examples = load_examples(image_dir)
    train_raw, val_raw = split_examples(examples)
    print(f"Train: {len(train_raw)}  Val: {len(val_raw)}")

    # Normalize targets same as original training
    def normalize(exs):
        out = []
        for ex in exs:
            t = ((np.array(ex.targets, dtype=np.float32) - target_mean) / target_std).tolist()
            out.append(Example(ex.image_path, t, ex.night_photo, ex.day_image))
        return out

    train_examples = normalize(train_raw)
    val_examples = normalize(val_raw)

    train_dl = DataLoader(BrightnessDataset(train_examples, train_tf),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dl = DataLoader(BrightnessDataset(val_examples, val_tf),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    optimizer = AdamW([
        {"params": model.features.parameters(), "lr": lr_backbone},
        {"params": model.avgpool.parameters(), "lr": lr_backbone},
        {"params": model.head.parameters(), "lr": lr_head},
    ], weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.HuberLoss()

    best_val_loss = float("inf")
    log_path = OUTPUT_DIR / "continue_training_log.csv"

    with log_path.open("w", newline="") as log_handle:
        writer = csv.DictWriter(log_handle, fieldnames=["epoch", "train_loss", "val_loss", "gray_mean_zscore_r2"])
        writer.writeheader()

        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss = 0.0
            for images, labels in train_dl:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())

            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_dl:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    preds = model(images)
                    val_loss += float(criterion(preds, labels).item())
                    all_preds.append(denormalize(preds, target_mean, target_std))
                    all_labels.append(denormalize(labels, target_mean, target_std))

            scheduler.step()
            train_loss /= max(len(train_dl), 1)
            val_loss /= max(len(val_dl), 1)

            preds_cat = torch.cat(all_preds)
            labels_cat = torch.cat(all_labels)
            r2 = per_target_r2(preds_cat, labels_cat)
            zscore_r2 = r2["gray_mean_zscore"]

            print(f"Epoch {epoch:03d}/{num_epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"gray_mean_zscore R2={zscore_r2:.4f}")

            writer.writerow({"epoch": epoch, "train_loss": round(train_loss, 6),
                             "val_loss": round(val_loss, 6), "gray_mean_zscore_r2": round(zscore_r2, 6)})
            log_handle.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "target_mean": target_mean.tolist(),
                    "target_std": target_std.tolist(),
                    "targets": TARGETS,
                    "image_size": IMG_SIZE,
                }, OUTPUT_DIR / "best_efficientnet_brightness.pt")
                print(f"  Saved best checkpoint (epoch {epoch})")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.image_dir, args.epochs, args.lr_backbone, args.lr_head, args.weight_decay)
