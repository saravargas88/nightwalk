# -*- coding: utf-8 -*-
"""finetune_brightness.py

Fine-tunes EfficientNet-B0 to predict nighttime brightness (regression)
from daytime images, using one of three backbone initializations:

  --backbone imagenet       Pure ImageNet weights (baseline)
  --backbone dino_counts    Weights from train_efficientnet_multihead.pt (Script 1)
  --backbone ssl            Weights from pretrain_selfsupervised.py (Script 3)

After all folds finish, evaluates the best checkpoint from each fold on the
held-out test split and saves test_predictions.csv + test_metrics.csv.

Outputs (under model-training/finetune-runs/<backbone>/n<n_train>/):
  fold_<k>/
    best_model.pt
    training_log.csv
    val_predictions_epoch_*.csv
    test_predictions.csv    ← test split predictions for this fold's best model
    test_metrics.csv        ← test split metrics for this fold's best model
  fold_summary.csv          ← val metrics aggregated across folds
  test_summary.csv          ← test metrics aggregated across folds

Usage:
    python finetune_brightness.py --backbone imagenet
    python finetune_brightness.py --backbone dino_counts --metric gray_mean_zscore
    python finetune_brightness.py --backbone ssl --folds 5
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

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
ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV = ROOT / "splits" / "train_split.csv"
TEST_CSV = ROOT / "splits" / "test_split.csv"
BRIGHTNESS_CSV = (
    ROOT / "brightnessmetricexperiments"
    / "experiment_outputs"
    / "paired_dataset_with_brightness.csv"
)
DAY_IMAGE_ROOT = ROOT / "brightnessmetricexperiments" / "nightwalk-images-224"

DINO_CHECKPOINT = ROOT / "model-training" / "best_efficientnet_multihead.pt"
SSL_CHECKPOINT = ROOT / "model-training" / "ssl-pretrain" / "best_ssl_backbone.pt"
OUTPUT_BASE = ROOT / "model-training" / "finetune-runs"

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE = 16
NUM_EPOCHS = 40
LR_HEAD = 3e-4
LR_BACKBONE = 3e-5
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 8
SAVE_PRED_EVERY = 5
RANDOM_SEED = 42

# Backbone-specific overrides applied in train_fold when backbone == "dino_counts"
DINO_LR_BACKBONE = 1e-5   # lower LR — backbone already has useful representations
DINO_WARMUP_EPOCHS = 10   # freeze backbone, train head-only first

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


def _load_filename_map(image_dir: Path) -> dict[str, str]:
    """Load original_day_image -> resized_filename map if present."""
    map_path = image_dir / "filename_map.csv"
    if not map_path.exists():
        return {}
    remap: dict[str, str] = {}
    with map_path.open(newline="") as f:
        for row in csv.DictReader(f):
            remap[row["original_day_image"]] = row["resized_filename"]
    return remap


def _load_examples_from_split(split_csv: Path, metric: str) -> list[Example]:
    """Join a split CSV with the brightness CSV and return valid Examples."""
    split_df = pd.read_csv(split_csv)
    brightness_df = pd.read_csv(BRIGHTNESS_CSV)

    split_df = split_df[
        split_df["day_image"].notna() & (split_df["day_image"].str.strip() != "")
    ]

    merged = split_df.merge(
        brightness_df[["night_photo", "day_image", metric]],
        on=["night_photo", "day_image"],
        how="inner",
    )

    name_map = _load_filename_map(DAY_IMAGE_ROOT)

    examples = []
    for _, row in merged.iterrows():
        original = row["day_image"]
        flat = name_map.get(original, original)
        day_path = DAY_IMAGE_ROOT / flat
        if not day_path.exists():
            continue
        examples.append(
            Example(
                image_path=str(day_path),
                day_image=original,
                night_photo=row["night_photo"],
                target=float(row[metric]),
            )
        )
    return examples


def load_train_examples(metric: str) -> list[Example]:
    examples = _load_examples_from_split(TRAIN_CSV, metric)
    print(f"Loaded {len(examples)} train examples with metric '{metric}'")
    return examples


def load_test_examples(metric: str) -> list[Example]:
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test split not found: {TEST_CSV}\n"
            "Run prepare_splits.py first."
        )
    examples = _load_examples_from_split(TEST_CSV, metric)
    print(f"Loaded {len(examples)} test examples with metric '{metric}'")
    return examples


def sample_n_train(examples: list[Example], n_train: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    if n_train >= len(examples):
        return examples
    return rng.sample(examples, n_train)


def make_kfold_splits(
    examples: list[Example],
    n_folds: int,
    seed: int,
) -> list[tuple[list[Example], list[Example]]]:
    """K-fold split grouped by day_image to prevent scene leakage."""
    rng = random.Random(seed)
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
        print("  Using pure ImageNet backbone (no additional pretraining)")
        return

    if backbone == "dino_counts":
        if not DINO_CHECKPOINT.exists():
            raise FileNotFoundError(f"DINO counts checkpoint not found: {DINO_CHECKPOINT}")
        state = torch.load(DINO_CHECKPOINT, map_location="cpu")
        state = state["model_state_dict"]
        remapped = {}
        for key, value in state.items():
            if key.startswith("backbone."):
                remapped["features." + key[len("backbone."):]] = value
            elif key.startswith("features."):
                remapped[key] = value
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        backbone_missing = [k for k in missing if not k.startswith("head.")]
        if backbone_missing:
            print(f"  WARNING: {len(backbone_missing)} backbone keys missing: {backbone_missing[:5]}")
        print(f"  Loaded DINO counts backbone — missing={len(missing)} unexpected={len(unexpected)}")
        return

    if backbone == "ssl":
        if not SSL_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"SSL checkpoint not found: {SSL_CHECKPOINT}\n"
                "Run pretrain_selfsupervised.py first."
            )
        state = torch.load(SSL_CHECKPOINT, map_location="cpu")
        model.features.load_state_dict(state["features"], strict=True)
        model.avgpool.load_state_dict(state["avgpool"], strict=True)
        print(f"  Loaded SSL backbone from {SSL_CHECKPOINT.name}")
        return

    raise ValueError(f"Unknown backbone: {backbone}. Choose from: imagenet, dino_counts, ssl")


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    mae = float((preds - targets).abs().mean().item())
    rmse = float(((preds - targets) ** 2).mean().sqrt().item())
    ss_res = float(((targets - preds) ** 2).sum().item())
    ss_tot = float(((targets - targets.mean()) ** 2).sum().item())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ── Prediction saving ─────────────────────────────────────────────────────────
def save_predictions(
    examples: list[Example],
    preds: torch.Tensor,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds_np = preds.cpu().numpy()
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["night_photo", "day_image", "true_target", "pred_target"],
        )
        writer.writeheader()
        for ex, pred in zip(examples, preds_np):
            writer.writerow({
                "night_photo": ex.night_photo,
                "day_image": ex.day_image,
                "true_target": round(ex.target, 6),
                "pred_target": round(float(pred), 6),
            })


# ── Test evaluation ───────────────────────────────────────────────────────────
def evaluate_on_test(
    test_examples: list[Example],
    checkpoint_path: Path,
    output_dir: Path,
    backbone: str,
    metric: str,
) -> dict[str, float]:
    """Load the best checkpoint for a fold and evaluate on the test split."""
    model = EfficientNetRegressor().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = BrightnessDataset(test_examples, val_tf)
    test_dl = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE != "cpu"),
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_dl:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            preds = model(images)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds_t = torch.cat(all_preds)
    all_targets_t = torch.cat(all_targets)
    metrics = compute_metrics(all_preds_t, all_targets_t)

    # Save predictions
    save_predictions(
        test_examples,
        all_preds_t,
        output_dir / "test_predictions.csv",
    )

    # Save metrics
    metrics_path = output_dir / "test_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backbone", "metric", "mae", "rmse", "r2"])
        writer.writeheader()
        writer.writerow({
            "backbone": backbone,
            "metric": metric,
            **{k: round(v, 6) for k, v in metrics.items()},
        })

    return metrics


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

    # dino_counts backbone: use lower LR + warmup freeze to prevent overfitting
    effective_lr_backbone = DINO_LR_BACKBONE if backbone == "dino_counts" else LR_BACKBONE
    warmup_epochs = DINO_WARMUP_EPOCHS if backbone == "dino_counts" else 0

    criterion = nn.HuberLoss()
    optimizer = AdamW(
        [
            {"params": model.features.parameters(), "lr": effective_lr_backbone},
            {"params": model.avgpool.parameters(), "lr": effective_lr_backbone},
            {"params": model.head.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    # Cosine anneal only over the epochs where the backbone is actually training
    scheduler = CosineAnnealingLR(optimizer, T_max=max(NUM_EPOCHS - warmup_epochs, 1))

    best_val_loss = float("inf")
    best_val_metrics: dict[str, float] = {}
    checkpoint_path = output_dir / "best_model.pt"

    log_path = output_dir / "training_log.csv"
    with log_path.open("w", newline="") as log_handle:
        writer = csv.DictWriter(
            log_handle,
            fieldnames=["epoch", "train_loss", "val_loss", "mae", "rmse", "r2"],
        )
        writer.writeheader()

        for epoch in range(1, NUM_EPOCHS + 1):
            # Warmup: backbone frozen for first N epochs, then thawed
            if epoch == 1 and warmup_epochs > 0:
                for p in model.features.parameters():
                    p.requires_grad_(False)
                print(f"  Backbone frozen for warmup ({warmup_epochs} epochs)")
            elif epoch == warmup_epochs + 1 and warmup_epochs > 0:
                for p in model.features.parameters():
                    p.requires_grad_(True)
                print(f"  Backbone unfrozen at epoch {epoch}")

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

            if epoch > warmup_epochs:
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
                best_val_metrics = metrics
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "backbone": backbone,
                        "metric": metric,
                        "image_size": IMG_SIZE,
                    },
                    checkpoint_path,
                )

            if epoch % SAVE_PRED_EVERY == 0:
                save_predictions(
                    val_examples,
                    all_preds_t,
                    output_dir / f"val_predictions_epoch_{epoch:03d}.csv",
                )

    return best_val_metrics, checkpoint_path


# ── Main training entry ───────────────────────────────────────────────────────
def train(
    backbone: str,
    n_train: int,
    metric: str,
    n_folds: int,
    seed: int,
    run_tag: str = "",
) -> tuple[dict, dict, dict, dict]:
    print(f"\n{'='*60}")
    print(f"Backbone: {backbone}  |  n_train: {n_train}  |  metric: {metric}  |  folds: {n_folds}")
    if run_tag:
        print(f"Run tag: {run_tag}")
    print(f"{'='*60}")

    train_examples = load_train_examples(metric)
    test_examples = load_test_examples(metric)

    sampled = sample_n_train(train_examples, n_train, seed)
    print(f"Using {len(sampled)} training examples for this run")

    folds = make_kfold_splits(sampled, n_folds, seed)
    folder_name = f"n{n_train}" + (f"_{run_tag}" if run_tag else "")
    run_dir = OUTPUT_BASE / backbone / folder_name

    fold_val_metrics = []
    fold_test_metrics = []

    for fold_idx, (train_exs, val_exs) in enumerate(folds):
        fold_dir = run_dir / f"fold_{fold_idx}"
        print(f"\n── Fold {fold_idx + 1}/{n_folds}  (train={len(train_exs)}, val={len(val_exs)}) ──")

        val_metrics, checkpoint_path = train_fold(
            train_exs, val_exs, backbone, fold_dir, metric
        )
        fold_val_metrics.append(val_metrics)
        print(f"  Val best  → mae={val_metrics['mae']:.4f}  rmse={val_metrics['rmse']:.4f}  r2={val_metrics['r2']:.4f}")

        # Evaluate best checkpoint from this fold on the test split
        print(f"  Evaluating fold {fold_idx} best model on test split...")
        test_metrics = evaluate_on_test(
            test_examples, checkpoint_path, fold_dir, backbone, metric
        )
        fold_test_metrics.append(test_metrics)
        print(f"  Test      → mae={test_metrics['mae']:.4f}  rmse={test_metrics['rmse']:.4f}  r2={test_metrics['r2']:.4f}")

    # ── Aggregate val metrics across folds ────────────────────────────────────
    val_summary_path = run_dir / "fold_summary.csv"
    with val_summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "mae", "rmse", "r2"])
        writer.writeheader()
        for i, m in enumerate(fold_val_metrics):
            writer.writerow({"fold": i, **{k: round(v, 6) for k, v in m.items()}})

    # ── Aggregate test metrics across folds ───────────────────────────────────
    test_summary_path = run_dir / "test_summary.csv"
    with test_summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "mae", "rmse", "r2"])
        writer.writeheader()
        for i, m in enumerate(fold_test_metrics):
            writer.writerow({"fold": i, **{k: round(v, 6) for k, v in m.items()}})

    val_mean = {k: float(np.mean([m[k] for m in fold_val_metrics])) for k in ["mae", "rmse", "r2"]}
    val_std  = {k: float(np.std( [m[k] for m in fold_val_metrics])) for k in ["mae", "rmse", "r2"]}
    test_mean = {k: float(np.mean([m[k] for m in fold_test_metrics])) for k in ["mae", "rmse", "r2"]}
    test_std  = {k: float(np.std( [m[k] for m in fold_test_metrics])) for k in ["mae", "rmse", "r2"]}

    print(f"\n── {n_folds}-fold summary ──────────────────────────────")
    print(f"  {'':8}  {'MAE':>20}  {'RMSE':>20}  {'R²':>20}")
    for split, mean, std in [("Val", val_mean, val_std), ("Test", test_mean, test_std)]:
        print(
            f"  {split:8}  "
            f"{mean['mae']:.4f} ± {std['mae']:.4f}  "
            f"{mean['rmse']:.4f} ± {std['rmse']:.4f}  "
            f"{mean['r2']:.4f} ± {std['r2']:.4f}"
        )

    return val_mean, val_std, test_mean, test_std


# ── Arg parsing ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune brightness regression.")
    parser.add_argument(
        "--backbone",
        choices=["imagenet", "dino_counts", "ssl"],
        default="dino_counts",
    )
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--metric", default=DEFAULT_METRIC, choices=AVAILABLE_METRICS)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr-backbone", type=float, default=None,
                        help="Override backbone LR (defaults: dino_counts=1e-5, others=3e-5)")
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Override warmup freeze epochs (default: dino_counts=10, others=0)")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Suffix appended to the output folder (e.g. 'warmup10_lr1e5'). "
                             "Lets you keep multiple runs side by side for comparison.")
    return parser.parse_args()


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    args = parse_args()

    # Apply CLI overrides to module-level constants so train_fold picks them up
    NUM_EPOCHS = args.epochs
    if args.lr_backbone is not None:
        DINO_LR_BACKBONE = args.lr_backbone
    if args.warmup_epochs is not None:
        DINO_WARMUP_EPOCHS = args.warmup_epochs

    train(
        backbone=args.backbone,
        n_train=args.n_train,
        metric=args.metric,
        n_folds=args.folds,
        seed=args.seed,
        run_tag=args.run_tag,
    )