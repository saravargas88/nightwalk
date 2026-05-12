"""Train EfficientNet-B0 4-class brightness classifier.

Bins are derived from gray_mean quartiles computed on the training split only
(no data leakage).  Labels: 0=very_dark  1=dark  2=bright  3=very_bright

Backbones:
  imagenet    -- fresh ImageNet weights (default)
  dino_counts -- load compatible layers from best_efficientnet_multihead.pt

Usage:
    python train_efficientnet_brightness_class.py
    python train_efficientnet_brightness_class.py \\
        --backbone dino_counts --epochs 45 --n-train 600 \\
        --output-dir brightness-class-runs/dino_counts/n600
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

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CSV = ROOT / "brightnessmetricexperiments" / "experiment_outputs" / "paired_dataset_with_brightness.csv"
TRAIN_CSV = ROOT / "splits" / "train_split.csv"
TEST_CSV = ROOT / "splits" / "test_split.csv"
DAY_IMAGE_ROOT = ROOT / "urban-mosaic" / "washington-square"
DINO_CHECKPOINT = ROOT / "model-training" / "best_efficientnet_multihead.pt"
DEFAULT_OUTPUT_DIR = ROOT / "model-training" / "brightness-class-runs" / "default"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
NUM_CLASSES = 4
CLASS_NAMES = ["very_dark", "dark", "bright", "very_bright"]
BATCH_SIZE = 16
DEFAULT_EPOCHS = 45
LR_HEAD = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224
VAL_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 4

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class Example:
    image_path: str
    gray_mean: float
    label: int          # set after bin edges are computed
    night_photo: str
    day_image: str


class BrightnessClassDataset(Dataset):
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
    def __init__(self, num_classes: int = NUM_CLASSES):
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


def load_name_map(image_dir: Path) -> dict:
    map_path = image_dir / "filename_map.csv"
    if not map_path.exists():
        return {}
    remap = {}
    with map_path.open(newline="") as f:
        for row in csv.DictReader(f):
            remap[row["original_day_image"]] = row["resized_filename"]
    return remap


def load_examples(image_dir: Path, split_csv: Path) -> list[Example]:
    name_map = load_name_map(image_dir)
    split_rows = list(csv.DictReader(split_csv.open(newline="")))
    allowed = {(r["night_photo"], r["day_image"]) for r in split_rows}
    brightness = {
        (r["night_photo"], r["day_image"]): r
        for r in csv.DictReader(DATA_CSV.open(newline=""))
    }
    examples: list[Example] = []
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
            gray_mean=float(b["gray_mean"]),
            label=-1,
            night_photo=b["night_photo"],
            day_image=rel,
        ))
    return examples


def compute_bin_edges(examples: list[Example]) -> list[float]:
    scores = [ex.gray_mean for ex in examples]
    q25, q50, q75 = float(np.percentile(scores, 25)), float(np.percentile(scores, 50)), float(np.percentile(scores, 75))
    return [q25, q50, q75]


def assign_labels(examples: list[Example], edges: list[float]) -> list[Example]:
    labeled = []
    for ex in examples:
        label = sum(ex.gray_mean > e for e in edges)
        labeled.append(Example(
            image_path=ex.image_path,
            gray_mean=ex.gray_mean,
            label=label,
            night_photo=ex.night_photo,
            day_image=ex.day_image,
        ))
    return labeled


def split_examples(examples: list[Example], seed: int = RANDOM_SEED) -> tuple[list[Example], list[Example]]:
    rng = random.Random(seed)
    groups: dict[str, list[Example]] = {}
    for ex in examples:
        groups.setdefault(ex.day_image, []).append(ex)
    day_images = list(groups)
    rng.shuffle(day_images)
    split_idx = int(len(day_images) * (1.0 - VAL_SPLIT))
    train_days = set(day_images[:split_idx])
    train_ex, val_ex = [], []
    for day_image, grouped in groups.items():
        if day_image in train_days:
            train_ex.extend(grouped)
        else:
            val_ex.extend(grouped)
    return train_ex, val_ex


def load_dino_backbone(model: EfficientNetBrightnessClassifier) -> None:
    state = torch.load(DINO_CHECKPOINT, map_location="cpu")
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("backbone.") or key.startswith("features."):
            remapped[key] = value
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"Loaded dino_counts backbone from {DINO_CHECKPOINT.name}")
    print(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> list[float]:
    preds = logits.argmax(dim=1)
    accs = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            accs.append(float("nan"))
        else:
            accs.append(float((preds[mask] == labels[mask]).float().mean().item()))
    return accs


def save_predictions(
    examples: list[Example],
    logits: torch.Tensor,
    epoch: int,
    output_dir: Path,
) -> None:
    preds = logits.argmax(dim=1).cpu().numpy()
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    out_path = output_dir / f"val_predictions_epoch_{epoch:03d}.csv"
    with out_path.open("w", newline="") as f:
        fieldnames = ["night_photo", "day_image", "gray_mean", "actual_label", "pred_label"] + [
            f"prob_{c}" for c in CLASS_NAMES
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ex, pred, prob_row in zip(examples, preds, probs):
            writer.writerow({
                "night_photo": ex.night_photo,
                "day_image": ex.day_image,
                "gray_mean": round(ex.gray_mean, 4),
                "actual_label": CLASS_NAMES[ex.label],
                "pred_label": CLASS_NAMES[int(pred)],
                **{f"prob_{CLASS_NAMES[i]}": round(float(prob_row[i]), 4) for i in range(NUM_CLASSES)},
            })


def train(
    image_dir: Path,
    backbone: str,
    n_train: int | None,
    num_epochs: int,
    lr_backbone: float,
    lr_head: float,
    seed: int,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_train_ex = load_examples(image_dir, TRAIN_CSV)
    train_raw, val_raw = split_examples(all_train_ex, seed)

    if n_train is not None and n_train < len(train_raw):
        rng = random.Random(seed)
        train_raw = rng.sample(train_raw, n_train)

    # Compute bin edges from train set only
    edges = compute_bin_edges(train_raw)
    print(f"Bin edges (gray_mean quartiles): {[round(e, 2) for e in edges]}")
    print(f"  0=very_dark ≤{edges[0]:.1f}  1=dark ≤{edges[1]:.1f}  2=bright ≤{edges[2]:.1f}  3=very_bright")

    # Save bin edges so eval scripts can reuse them
    edges_path = output_dir / "bin_edges.csv"
    with edges_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["q25", "q50", "q75"])
        w.writeheader()
        w.writerow({"q25": round(edges[0], 4), "q50": round(edges[1], 4), "q75": round(edges[2], 4)})

    train_examples = assign_labels(train_raw, edges)
    val_examples = assign_labels(val_raw, edges)

    print(f"Train: {len(train_examples)}  Val: {len(val_examples)}")
    train_dist = np.bincount([ex.label for ex in train_examples], minlength=NUM_CLASSES)
    print(f"Train class distribution: {dict(zip(CLASS_NAMES, train_dist.tolist()))}")

    train_ds = BrightnessClassDataset(train_examples, train_tf)
    val_ds = BrightnessClassDataset(val_examples, val_tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE != "cpu"))
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE != "cpu"))

    model = EfficientNetBrightnessClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    if backbone == "dino_counts":
        load_dino_backbone(model)

    optimizer = AdamW(
        [
            {"params": model.features.parameters(), "lr": lr_backbone},
            {"params": model.avgpool.parameters(), "lr": lr_backbone},
            {"params": model.head.parameters(), "lr": lr_head},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    log_path = output_dir / "training_log.csv"
    log_fields = (
        ["epoch", "train_loss", "val_loss", "val_acc"]
        + [f"val_acc_{c}" for c in CLASS_NAMES]
    )

    with log_path.open("w", newline="") as log_handle:
        writer = csv.DictWriter(log_handle, fieldnames=log_fields)
        writer.writeheader()

        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss = 0.0
            for images, labels in train_dl:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())

            model.eval()
            val_loss = 0.0
            all_logits, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_dl:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    logits = model(images)
                    val_loss += float(criterion(logits, labels).item())
                    all_logits.append(logits)
                    all_labels.append(labels)

            scheduler.step()
            train_loss /= max(len(train_dl), 1)
            val_loss /= max(len(val_dl), 1)

            logits_cat = torch.cat(all_logits, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
            val_acc = accuracy(logits_cat, labels_cat)
            per_cls = per_class_accuracy(logits_cat, labels_cat, NUM_CLASSES)

            per_cls_str = "  ".join(f"{CLASS_NAMES[i]}={per_cls[i]:.2f}" for i in range(NUM_CLASSES))
            print(
                f"Epoch {epoch:03d}/{num_epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"val_acc={val_acc:.3f}  [{per_cls_str}]"
            )

            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                **{f"val_acc_{CLASS_NAMES[i]}": round(per_cls[i], 6) if not np.isnan(per_cls[i]) else "" for i in range(NUM_CLASSES)},
            }
            writer.writerow(row)
            log_handle.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "bin_edges": edges,
                        "class_names": CLASS_NAMES,
                        "backbone": backbone,
                        "image_size": IMG_SIZE,
                    },
                    output_dir / "best_efficientnet_brightness_class.pt",
                )

            if epoch % 5 == 0:
                save_predictions(val_examples, logits_cat, epoch, output_dir)

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    return {"best_val_acc": best_val_acc, "bin_edges": edges}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 4-class brightness classifier.")
    parser.add_argument("--image-dir", type=Path, default=DAY_IMAGE_ROOT)
    parser.add_argument("--backbone", choices=["imagenet", "dino_counts"], default="imagenet")
    parser.add_argument("--n-train", type=int, default=None, help="Cap training examples (ablation)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr-backbone", type=float, default=LR_BACKBONE)
    parser.add_argument("--lr-head", type=float, default=LR_HEAD)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device: {DEVICE}  backbone: {args.backbone}  epochs: {args.epochs}  lr_backbone: {args.lr_backbone}  lr_head: {args.lr_head}")
    train(
        image_dir=args.image_dir,
        backbone=args.backbone,
        n_train=args.n_train,
        num_epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        seed=args.seed,
        output_dir=args.output_dir,
    )
