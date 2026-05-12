# -*- coding: utf-8 -*-
"""eval_brightness_checkpoint.py

Evaluates best_efficientnet_brightness.pt on the held-out test split
and prints metrics comparable to finetune_brightness.py results.

Usage:
    python3 model-training/eval_brightness_checkpoint.py --image-dir nightwalk-images-224
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "splits" / "test_split.csv"
BRIGHTNESS_CSV = (
    ROOT / "brightnessmetricexperiments"
    / "experiment_outputs"
    / "paired_dataset_with_brightness.csv"
)
CHECKPOINT = ROOT / "model-training" / "brightness-regression-run" / "best_efficientnet_brightness.pt"
TARGET_METRIC = "gray_mean_zscore"

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


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


class SimpleDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def load_name_map(image_dir):
    map_path = image_dir / "filename_map.csv"
    if not map_path.exists():
        return {}
    remap = {}
    with map_path.open(newline="") as f:
        for row in csv.DictReader(f):
            remap[row["original_day_image"]] = row["resized_filename"]
    return remap


def run(image_dir):
    name_map = load_name_map(image_dir)

    # Load test split + brightness CSV
    test_rows = list(csv.DictReader(TEST_CSV.open(newline="")))
    brightness = {
        (r["night_photo"], r["day_image"]): r
        for r in csv.DictReader(BRIGHTNESS_CSV.open(newline=""))
    }

    paths, targets = [], []
    for row in test_rows:
        di = row.get("day_image", "").strip()
        if not di:
            continue
        b = brightness.get((row["night_photo"], di))
        if b is None:
            continue
        flat = name_map.get(di)
        img_path = image_dir / flat if flat else image_dir / di
        if not img_path.exists():
            continue
        paths.append(str(img_path))
        targets.append(float(b[TARGET_METRIC]))

    print(f"Test examples found: {len(paths)}")

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    target_mean = np.array(ckpt["target_mean"], dtype=np.float32)
    target_std = np.array(ckpt["target_std"], dtype=np.float32)
    trained_targets = ckpt["targets"]
    target_idx = trained_targets.index(TARGET_METRIC)

    model = EfficientNetBrightnessRegressor(num_outputs=len(trained_targets)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = SimpleDataset(paths, tf)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds = []
    with torch.no_grad():
        for imgs in dl:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            all_preds.append(out.cpu().numpy())

    preds_norm = np.concatenate(all_preds, axis=0)[:, target_idx]
    preds = preds_norm * target_std[target_idx] + target_mean[target_idx]
    y = np.array(targets)

    mae = float(np.abs(y - preds).mean())
    rmse = float(np.sqrt(((y - preds) ** 2).mean()))
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    print(f"\nTest results for {TARGET_METRIC}:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")
    print(f"\nBaseline (previous best): MAE=0.486  R2=0.460")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"Device: {DEVICE}")
    run(args.image_dir)
