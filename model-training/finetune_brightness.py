"""finetune_brightness.py

Fine-tunes EfficientNet-B0 to predict nighttime brightness (regression)
from daytime images, using one of three backbone initializations:

  --backbone imagenet       Pure ImageNet weights (baseline)
  --backbone dino_counts    Weights from train_efficientnet_multihead.pt (Script 1)
  --backbone ssl            Weights from pretrain_selfsupervised.py (Script 3)

Supports a label-size sweep (--n-train) to study how many paired
examples are needed.

Outputs (under model-training/finetune-runs/<backbone>/<n_train>/fold_<k>/):
  best_model.pt
  training_log.csv
  val_predictions_epoch_*.csv

Usage:
    python finetune_brightness.py --backbone imagenet --n-train 800
    python finetune_brightness.py --backbone dino_counts --n-train 200 --metric gray_mean_zscore
    python finetune_brightness.py --backbone ssl --n-train 100 --folds 5
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "splits" / "train_split.csv"
BRIGHTNESS_CSV = ROOT / "brightnessmetricexperiments" / "experiment_outputs" / "paired_dataset_with_brightness.csv"
DAY_IMAGE_ROOT = ROOT / "urban-mosaic" / "washington-square"

DINO_CHECKPOINT = ROOT / "model-training" / "best_efficientnet_multihead.pt"
SSL_CHECKPOINT = ROOT / "model-training" / "ssl-pretrain" / "best_ssl_backbone.pt"

OUTPUT_BASE = ROOT / "model-training" / "finetune-runs"

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR_HEAD = 3e-4
LR_BACKBONE = 3e-5
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 8
SAVE_PRED_EVERY = 5
RANDOM_SEED = 42

AVAILABLE_METRICS = [
    "gray_mean",
    "gray_median",
    "gray_trimmed_mean",
    "gray_p90",
    "luma_mean",
    "value_mean",
    "gray_mean_over_std",
    "gray_mean_zscore",
    "gray_mean_robust_zscore",
]
DEFAULT_METRIC = "gray_mean_zscore"

LABEL_SIZE_OPTIONS = [50, 100, 200, 400, 800]
DEFAULT_N_TRAIN = 800
DEFAULT_FOLDS = 5

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ── Data ──────────────────────────────────────────────────────────────────────
@dataclass
class Example:
    image_path: str
    day_image: str
    night_photo: str
    target: float


class BrightnessDataset(Dataset):
    def __init__(self, examples: list[Example], transform: transforms.Compose):
        self.examples = examples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        image = Image.open(ex.image_path).convert("RGB")
        return self.transform(image), torch.tensor(ex.target, dtype=torch.float32)


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


def load_examples(metric: str) -> list[Example]:
    """Join train_split.csv with brightness CSV on night_photo / day_image."""
    train_df = pd.read_csv(TRAIN_CSV)
    brightness_df = pd.read_csv(BRIGHTNESS_CSV)

    # Keep only valid pairs
    train_df = train_df[train_df["day_image"].notna() & (train_df["day_image"].str.strip() != "")]

    # Merge brightness values in
    merged = train_df.merge(
        brightness_df[["night_photo", "day_image", metric]],
        on=["night_photo", "day_image"],
        how="inner",
    )

    examples = []
    for _, row in merged.iterrows():
        day_path = DAY_IMAGE_ROOT / row["day_image"]
        if not day_path.exists():
            continue
        examples.append(
            Example(
                image_path=str(day_path),
                day_image=row["day_image"],
                night_photo=row["night_photo"],
                target=float(row[metric]),
            )
        )
    print(f"Loaded {len(examples)} valid examples with brightness metric '{metric}'")
    return examples


def sample_n_train(
    examples: list[Example],
    n_train: int,
    seed: int,
) -> list[Example]:
    """Sample n_train examples, grouped by day_image for consistency."""
    rng = random.Random(seed)
    if n_train >= len(examples):
        return examples
    sampled = rng.sample(examples, n_train)
    return sampled


def make_kfold_splits(
    examples: list[Example],
    n_folds: int,
    seed: int,
) -> list[tuple[list[Example], list[Example]]]:
    """
    K-fold split grouped by day_image so the same scene
    never appears in both train and val within a fold.
    """
    rng = random.Random(seed)

    # Group by day_image
    groups: dict[str, list[Example]] = {}
    for ex in examples:
        groups.setdefault(ex.day_image, []).append(ex)

    day_images = list(groups.keys())
    rng.shuffle(day_images)

    fold_size = len(day_images) // n_folds
    folds = []
    for k in range(n_folds):
        val_days = set(day_images[k * fold_size: (k + 1) * fold_size])
        train_exs = [ex for ex in examples if ex.day_image not in val_days]
        val_exs = [ex for ex in examples if ex.day_image in val_days]
        folds.append((train_exs, val_exs))
    return folds


# ── Model ─────────────────────────────────────────────────────────────────────
class EfficientNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = base.classifier[1].in_features  # 1280
        self.features = base.features
        self.avgpool = base.avgpool
        self.head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.avgpool(self.features(x)), 1)
        return self.head(x).squeeze(1)


# ── Backbone loading ──────────────────────────────────────────────────────────
def load_backbone(model: EfficientNetRegressor, backbone: str) -> None:
    if backbone == "imagenet":
        print("Using pure ImageNet backbone (no additional pretraining)")
        return

    if backbone == "dino_counts":
        if not DINO_CHECKPOINT.exists():
            raise FileNotFoundError(f"DINO counts checkpoint not found: {DINO_CHECKPOINT}")
        state = torch.load(DINO_CHECKPOINT, map_location="cpu")
        # Script 1 saves raw state_dict with keys like backbone.* or features.*
        remapped = {}
        for key, value in state.items():
            if key.startswith("backbone."):
                new_key = "features." + key[len("backbone."):]
                remapped[new_key] = value
            elif key.startswith("features."):
                remapped[key] = value
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        # Expect only head keys to be missing
        backbone_missing = [k for k in missing if not k.startswith("head.")]
        if backbone_missing:
            print(f"  WARNING: {len(backbone_missing)} backbone keys missing: {backbone_missing[:5]}")
        print(f"  Loaded DINO counts backbone from {DINO_CHECKPOINT.name}")
        print(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")
        return

    if backbone == "ssl":
        if not SSL_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"SSL checkpoint not found: {SSL_CHECKPOINT}\n"
                "Run pretrain_selfsupervised.py first."
            )
        state = torch.load(SSL_CHECKPOINT, map_location="cpu")
        # SSL checkpoint saves {"features": ..., "avgpool": ...}
        missing_f, unexpected_f = model.features.load_state_dict(state["features"], strict=True)
        missing_p, unexpected_p = model.avgpool.load_state_dict(state["avgpool"], strict=True)
        print(f"  Loaded SSL backbone from {SSL_CHECKPOINT.name}")
        return

    raise ValueError(f"Unknown backbone: {backbone}. Choose from: imagenet, dino_counts, ssl")


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    mae = float((preds - targets).abs().mean().item())
    rmse = float(((preds - targets) ** 2).mean().sqrt().item())
    # R² 
    ss_res = float(((targets - preds) ** 2).sum().item())
    ss_tot = float(((targets - targets.mean()) ** 2).sum().item())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ── Prediction saving ─────────────────────────────────────────────────────────
def save_predictions(
    examples: list[Example],
    preds: torch.Tensor,
    epoch: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"val_predictions_epoch_{epoch:03d}.csv"
    preds_np = preds.cpu().numpy()
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["night_photo", "day_image", "true_target", "pred_target"])
        writer.writeheader()
        for ex, pred in zip(examples, preds_np):
            writer.writerow({
                "night_photo": ex.night_photo,
                "day_image": ex.day_image,
                "true_target": round(ex.target, 6),
                "pred_target": round(float(pred), 6),
            })


# ── Single fold training ──────────────────────────────────────────────────────
def train_fold(
    train_examples: list[Example],
    val_examples: list[Example],
    backbone: str,
    output_dir: Path,
    metric: str,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = BrightnessDataset(train_examples, train_tf)
    val_ds = BrightnessDataset(val_examples, val_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE != "cpu"))
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE != "cpu"))

    model = EfficientNetRegressor().to(DEVICE)
    load_backbone(model, backbone)

    criterion = nn.HuberLoss()
    optimizer = AdamW(
        [
            {"params": model.features.parameters(), "lr": LR_BACKBONE},
            {"params": model.avgpool.parameters(), "lr": LR_BACKBONE},
            {"params": model.head.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}

    log_path = output_dir / "training_log.csv"
    with log_path.open("w", newline="") as log_handle:
        writer = csv.DictWriter(
            log_handle,
            fieldnames=["epoch", "train_loss", "val_loss", "mae", "rmse", "r2"],
        )
        writer.writeheader()

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for images, targets in train_dl:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                preds = model(images)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            all_preds, all_targets = [], []
            with torch.no_grad():
                for images, targets in val_dl:
                    images, targets = images.to(DEVICE), targets.to(DEVICE)
                    preds = model(images)
                    val_loss += criterion(preds, targets).item()
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.cpu())

            scheduler.step()
            train_loss /= max(len(train_dl), 1)
            val_loss /= max(len(val_dl), 1)

            all_preds_t = torch.cat(all_preds)
            all_targets_t = torch.cat(all_targets)
            metrics = compute_metrics(all_preds_t, all_targets_t)

            print(
                f"  Epoch {epoch:03d}/{NUM_EPOCHS}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"mae={metrics['mae']:.4f}  rmse={metrics['rmse']:.4f}  r2={metrics['r2']:.4f}"
            )

            writer.writerow({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                **{k: round(v, 6) for k, v in metrics.items()},
            })
            log_handle.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = metrics
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "backbone": backbone,
                        "metric": metric,
                        "image_size": IMG_SIZE,
                    },
                    output_dir / "best_model.pt",
                )

            if epoch % SAVE_PRED_EVERY == 0:
                save_predictions(val_examples, all_preds_t, epoch, output_dir)

    return best_metrics


# ── Main training entry ───────────────────────────────────────────────────────
def train(
    backbone: str,
    n_train: int,
    metric: str,
    n_folds: int,
    seed: int,
) -> None:
    print(f"\n{'='*60}")
    print(f"Backbone: {backbone}  |  n_train: {n_train}  |  metric: {metric}  |  folds: {n_folds}")
    print(f"{'='*60}")

    all_examples = load_examples(metric)
    sampled = sample_n_train(all_examples, n_train, seed)
    print(f"Using {len(sampled)} examples for this run")

    folds = make_kfold_splits(sampled, n_folds, seed)

    fold_metrics = []
    run_dir = OUTPUT_BASE / backbone / f"n{n_train}"

    for fold_idx, (train_exs, val_exs) in enumerate(folds):
        fold_dir = run_dir / f"fold_{fold_idx}"
        print(f"\n── Fold {fold_idx + 1}/{n_folds}  (train={len(train_exs)}, val={len(val_exs)}) ──")
        metrics = train_fold(train_exs, val_exs, backbone, fold_dir, metric)
        fold_metrics.append(metrics)
        print(f"  Best → mae={metrics['mae']:.4f}  rmse={metrics['rmse']:.4f}  r2={metrics['r2']:.4f}")

    # Aggregate across folds
    summary_path = run_dir / "fold_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "mae", "rmse", "r2"])
        writer.writeheader()
        for i, m in enumerate(fold_metrics):
            writer.writerow({"fold": i, **{k: round(v, 6) for k, v in m.items()}})

    mean_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in ["mae", "rmse", "r2"]}
    std_metrics = {k: np.std([m[k] for m in fold_metrics]) for k in ["mae", "rmse", "r2"]}

    print(f"\n── {n_folds}-fold summary ──")
    for k in ["mae", "rmse", "r2"]:
        print(f"  {k}: {mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}")

    return mean_metrics, std_metrics


# ── Arg parsing ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune brightness regression.")
    parser.add_argument(
        "--backbone",
        choices=["imagenet", "dino_counts", "ssl"],
        default="dino_counts",
        help="Which backbone initialization to use.",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=DEFAULT_N_TRAIN,
        help="Number of training examples to use (label size sweep).",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=AVAILABLE_METRICS,
        help="Brightness metric to regress.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_FOLDS,
        help="Number of cross-validation folds.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    args = parse_args()
    train(
        backbone=args.backbone,
        n_train=args.n_train,
        metric=args.metric,
        n_folds=args.folds,
        seed=args.seed,
    )