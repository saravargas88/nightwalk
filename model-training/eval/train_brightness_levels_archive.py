"""Train EfficientNet-B0 to predict binned night brightness levels from day images.

This is a classification version of the brightness pipeline. It is useful when
the exact scalar brightness target is noisy but broader illumination levels are
still learnable.

By default this script:
1. Reads the matched day-image / night-brightness dataset
2. Builds quantile bins from a chosen brightness metric
3. Initializes EfficientNet-B0 from best_efficientnet_multihead.pt
4. Trains a classification head to predict brightness level
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
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


ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CSV = ROOT / "brightnessmetricexperiments" / "experiment_outputs" / "paired_dataset_with_brightness.csv"
DAY_IMAGE_ROOT = ROOT / "urban-mosaic" / "washington-square"
CHECKPOINT_PATH = ROOT / "model-training" / "best_efficientnet_multihead.pt"
OUTPUT_DIR = ROOT / "model-training" / "brightness-level-run"

IMAGE_COL = "day_image"
DEFAULT_TARGET_METRIC = "gray_mean_zscore"
DEFAULT_NUM_BINS = 4
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

BATCH_SIZE = 16
NUM_EPOCHS = 20
LR_HEAD = 3e-4
LR_BACKBONE = 3e-5
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224
VAL_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 4
SAVE_PRED_EVERY = 5
DEFAULT_FREEZE_BACKBONE = False
DEFAULT_USE_CLASS_WEIGHTS = True

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class Example:
    image_path: str
    day_image: str
    night_photo: str
    raw_target: float
    label: int


class BrightnessLevelDataset(Dataset):
    def __init__(self, examples: list[Example], transform: transforms.Compose):
        self.examples = examples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        image = Image.open(ex.image_path).convert("RGB")
        return self.transform(image), torch.tensor(ex.label, dtype=torch.long)


class EfficientNetBrightnessClassifier(nn.Module):
    def __init__(self, num_classes: int):
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
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.head(x)


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


def read_rows() -> list[dict[str, str]]:
    with DATA_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_quantile_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    probs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(values, probs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges


def assign_bin(value: float, edges: np.ndarray) -> int:
    idx = int(np.searchsorted(edges[1:-1], value, side="right"))
    return max(0, min(idx, len(edges) - 2))


def build_examples(target_metric: str, num_bins: int) -> tuple[list[Example], np.ndarray]:
    rows = read_rows()
    target_values = np.array([float(row[target_metric]) for row in rows], dtype=np.float32)
    edges = make_quantile_edges(target_values, num_bins)

    examples: list[Example] = []
    for row in rows:
        day_path = DAY_IMAGE_ROOT / row[IMAGE_COL]
        if not day_path.exists():
            continue
        raw_target = float(row[target_metric])
        label = assign_bin(raw_target, edges)
        examples.append(
            Example(
                image_path=str(day_path),
                day_image=row[IMAGE_COL],
                night_photo=row["night_photo"],
                raw_target=raw_target,
                label=label,
            )
        )
    return examples, edges


def split_examples(examples: list[Example]) -> tuple[list[Example], list[Example]]:
    rng = random.Random(RANDOM_SEED)
    groups: dict[str, list[Example]] = {}
    for ex in examples:
        groups.setdefault(ex.day_image, []).append(ex)

    day_images = list(groups)
    rng.shuffle(day_images)
    split_idx = int(len(day_images) * (1.0 - VAL_SPLIT))
    train_days = set(day_images[:split_idx])

    train_examples: list[Example] = []
    val_examples: list[Example] = []
    for day_image, grouped in groups.items():
        if day_image in train_days:
            train_examples.extend(grouped)
        else:
            val_examples.extend(grouped)
    return train_examples, val_examples


def load_pretrained_backbone(model: EfficientNetBrightnessClassifier) -> None:
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            remapped[key] = value
        elif key.startswith("features."):
            remapped[key] = value
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"Loaded checkpoint backbone from {CHECKPOINT_PATH.name}")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")


def compute_class_weights(examples: list[Example], num_classes: int) -> torch.Tensor:
    counts = Counter(ex.label for ex in examples)
    total = sum(counts.values())
    weights = []
    for cls in range(num_classes):
        count = max(counts.get(cls, 0), 1)
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return float((preds == labels).float().mean().item())


def balanced_accuracy(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    recalls = []
    for cls in range(num_classes):
        mask = labels == cls
        if int(mask.sum().item()) == 0:
            continue
        recalls.append(float((preds[mask] == cls).float().mean().item()))
    return float(sum(recalls) / max(len(recalls), 1))


def save_predictions(examples: list[Example], probs: torch.Tensor, preds: torch.Tensor, epoch: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"val_predictions_epoch_{epoch:03d}.csv"
    probs_np = probs.cpu().numpy()
    preds_np = preds.cpu().numpy()
    with out_path.open("w", newline="") as handle:
        fieldnames = ["night_photo", "day_image", "raw_target", "true_bin", "pred_bin"]
        fieldnames += [f"prob_bin_{i}" for i in range(probs_np.shape[1])]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ex, prob_row, pred in zip(examples, probs_np, preds_np):
            row = {
                "night_photo": ex.night_photo,
                "day_image": ex.day_image,
                "raw_target": round(ex.raw_target, 4),
                "true_bin": ex.label,
                "pred_bin": int(pred),
            }
            for i, prob in enumerate(prob_row):
                row[f"prob_bin_{i}"] = round(float(prob), 6)
            writer.writerow(row)
    print(f"  Saved predictions to {out_path}")


def save_metadata(
    edges: np.ndarray,
    train_examples: list[Example],
    val_examples: list[Example],
    target_metric: str,
    num_bins: int,
    freeze_backbone: bool,
    use_class_weights: bool,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "run_metadata.txt"
    train_counts = Counter(ex.label for ex in train_examples)
    val_counts = Counter(ex.label for ex in val_examples)
    lines = [
        f"checkpoint={CHECKPOINT_PATH}",
        f"target_metric={target_metric}",
        f"num_bins={num_bins}",
        f"freeze_backbone={freeze_backbone}",
        f"use_class_weights={use_class_weights}",
        f"bin_edges={edges.tolist()}",
        f"train_counts={dict(sorted(train_counts.items()))}",
        f"val_counts={dict(sorted(val_counts.items()))}",
    ]
    path.write_text("\n".join(lines))


def train(target_metric: str, num_bins: int, freeze_backbone: bool, use_class_weights: bool) -> None:
    examples, edges = build_examples(target_metric, num_bins)
    train_examples, val_examples = split_examples(examples)
    print(f"Loaded examples: {len(examples)}")
    print(f"Train: {len(train_examples)}  Val: {len(val_examples)}")
    print(f"Metric: {target_metric}  bins: {num_bins}")
    print(f"Bin edges: {edges.tolist()}")
    print(f"Train label counts: {dict(sorted(Counter(ex.label for ex in train_examples).items()))}")
    print(f"Val label counts: {dict(sorted(Counter(ex.label for ex in val_examples).items()))}")
    save_metadata(edges, train_examples, val_examples, target_metric, num_bins, freeze_backbone, use_class_weights)

    train_ds = BrightnessLevelDataset(train_examples, train_tf)
    val_ds = BrightnessLevelDataset(val_examples, val_tf)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE != "cpu"),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE != "cpu"),
    )

    model = EfficientNetBrightnessClassifier(num_classes=num_bins).to(DEVICE)
    load_pretrained_backbone(model)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_examples, num_bins).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        [
            {"params": [p for p in model.features.parameters() if p.requires_grad], "lr": LR_BACKBONE},
            {"params": model.avgpool.parameters(), "lr": LR_BACKBONE},
            {"params": model.head.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_score = -float("inf")
    history_path = OUTPUT_DIR / "training_log.csv"
    with history_path.open("w", newline="") as log_handle:
        writer = csv.DictWriter(
            log_handle,
            fieldnames=["epoch", "train_loss", "val_loss", "accuracy", "balanced_accuracy"],
        )
        writer.writeheader()

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for images, labels in train_dl:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())

            model.eval()
            val_loss = 0.0
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_dl:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    logits = model(images)
                    val_loss += float(criterion(logits, labels).item())
                    all_logits.append(logits)
                    all_labels.append(labels)

            scheduler.step()
            train_loss /= max(len(train_dl), 1)
            val_loss /= max(len(val_dl), 1)

            logits = torch.cat(all_logits, dim=0)
            labels = torch.cat(all_labels, dim=0)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            acc = accuracy(preds, labels)
            bal_acc = balanced_accuracy(preds, labels, num_bins)

            print(
                f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"acc={acc:.4f}  bal_acc={bal_acc:.4f}"
            )

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "accuracy": round(acc, 6),
                    "balanced_accuracy": round(bal_acc, 6),
                }
            )
            log_handle.flush()

            score = bal_acc
            if score > best_score:
                best_score = score
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "target_metric": target_metric,
                        "num_bins": num_bins,
                        "bin_edges": edges.tolist(),
                        "image_size": IMG_SIZE,
                    },
                    OUTPUT_DIR / "best_efficientnet_brightness_levels.pt",
                )
                print(f"  Saved best checkpoint to {OUTPUT_DIR / 'best_efficientnet_brightness_levels.pt'}")

            if epoch % SAVE_PRED_EVERY == 0:
                save_predictions(val_examples, probs, preds, epoch)

    print(f"Done. Best balanced accuracy: {best_score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet brightness-level classifier.")
    parser.add_argument(
        "--metric",
        default=DEFAULT_TARGET_METRIC,
        choices=AVAILABLE_METRICS,
        help="Brightness metric to bin into classes.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_NUM_BINS,
        choices=[3, 4, 5],
        help="Number of quantile bins for brightness classification.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the EfficientNet feature extractor and train only the classifier head.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted cross-entropy.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Using device: {DEVICE}")
    train(
        target_metric=args.metric,
        num_bins=args.bins,
        freeze_backbone=args.freeze_backbone or DEFAULT_FREEZE_BACKBONE,
        use_class_weights=(not args.no_class_weights) and DEFAULT_USE_CLASS_WEIGHTS,
    )
