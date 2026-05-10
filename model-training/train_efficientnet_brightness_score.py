"""Train EfficientNet-B0 brightness regressor from the existing count checkpoint.

This script:
1. Reads matched day images plus brightness targets from brightnessmetricexperiments outputs
2. Initializes an EfficientNet-B0 backbone
3. Loads compatible weights from model-training/best_efficientnet_small.pt
4. Replaces the old count head with a brightness regression head
5. Trains on day images to predict multiple night-brightness targets

The main goal is to test whether the pretrained day-feature backbone transfers
better than the count-only linear models for night brightness prediction.
"""

from __future__ import annotations

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
DATA_CSV = ROOT / "brightnessmetricexperiments" / "experiment_outputs" / "paired_dataset_with_brightness.csv"
DAY_IMAGE_ROOT = ROOT / "urban-mosaic" / "washington-square"
CHECKPOINT_PATH = ROOT / "model-training" / "best_efficientnet_multihead.pt"
OUTPUT_DIR = ROOT / "model-training" / "brightness-regression-run"

IMAGE_COL = "day_image"
TARGETS = [
    "gray_mean",
    "luma_mean",
    "value_mean",
    "gray_mean_zscore",
]

BATCH_SIZE = 16
NUM_EPOCHS = 25
LR_HEAD = 3e-4
LR_BACKBONE = 3e-5
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224
VAL_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 4
SAVE_PRED_EVERY = 5

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class Example:
    image_path: str
    targets: list[float]
    night_photo: str
    day_image: str


class BrightnessDataset(Dataset):
    def __init__(self, examples: list[Example], transform: transforms.Compose):
        self.examples = examples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        image = Image.open(ex.image_path).convert("RGB")
        return self.transform(image), torch.tensor(ex.targets, dtype=torch.float32)


class EfficientNetBrightnessRegressor(nn.Module):
    def __init__(self, num_outputs: int):
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


def load_examples() -> list[Example]:
    with DATA_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    examples: list[Example] = []
    for row in rows:
        day_path = DAY_IMAGE_ROOT / row[IMAGE_COL]
        if not day_path.exists():
            continue
        targets = [float(row[target]) for target in TARGETS]
        examples.append(
            Example(
                image_path=str(day_path),
                targets=targets,
                night_photo=row["night_photo"],
                day_image=row[IMAGE_COL],
            )
        )
    return examples


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
    for day_image, grouped_examples in groups.items():
        if day_image in train_days:
            train_examples.extend(grouped_examples)
        else:
            val_examples.extend(grouped_examples)
    return train_examples, val_examples


def compute_target_stats(examples: list[Example]) -> tuple[np.ndarray, np.ndarray]:
    targets = np.array([ex.targets for ex in examples], dtype=np.float32)
    mean = targets.mean(axis=0)
    std = targets.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def normalize_examples(examples: list[Example], mean: np.ndarray, std: np.ndarray) -> list[Example]:
    normalized: list[Example] = []
    for ex in examples:
        targets = ((np.array(ex.targets, dtype=np.float32) - mean) / std).tolist()
        normalized.append(
            Example(
                image_path=ex.image_path,
                targets=targets,
                night_photo=ex.night_photo,
                day_image=ex.day_image,
            )
        )
    return normalized


def load_pretrained_backbone(model: EfficientNetBrightnessRegressor) -> None:
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


def denormalize(preds: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=preds.device, dtype=preds.dtype)
    std_t = torch.tensor(std, device=preds.device, dtype=preds.dtype)
    return preds * std_t + mean_t


def per_target_mae(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    maes = (preds - labels).abs().mean(dim=0)
    return {target: float(maes[i].item()) for i, target in enumerate(TARGETS)}


def per_target_r2(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, target in enumerate(TARGETS):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        denom = torch.sum((y_true - y_true.mean()) ** 2)
        if float(denom.item()) < 1e-8:
            out[target] = 0.0
        else:
            out[target] = float((1.0 - torch.sum((y_true - y_pred) ** 2) / denom).item())
    return out


def save_predictions(
    examples: list[Example],
    preds: torch.Tensor,
    labels: torch.Tensor,
    epoch: int,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"val_predictions_epoch_{epoch:03d}.csv"
    with out_path.open("w", newline="") as handle:
        fieldnames = ["night_photo", "day_image"]
        for target in TARGETS:
            fieldnames.extend([f"actual_{target}", f"pred_{target}"])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        for ex, pred_row, label_row in zip(examples, preds_np, labels_np):
            row = {
                "night_photo": ex.night_photo,
                "day_image": Path(ex.image_path).relative_to(DAY_IMAGE_ROOT).as_posix(),
            }
            for i, target in enumerate(TARGETS):
                row[f"actual_{target}"] = round(float(label_row[i]), 4)
                row[f"pred_{target}"] = round(float(pred_row[i]), 4)
            writer.writerow(row)
    print(f"  Saved predictions to {out_path}")


def train() -> None:
    examples = load_examples()
    train_raw, val_raw = split_examples(examples)
    print(f"Loaded examples: {len(examples)}")
    print(f"Train: {len(train_raw)}  Val: {len(val_raw)}")

    target_mean, target_std = compute_target_stats(train_raw)
    train_examples = normalize_examples(train_raw, target_mean, target_std)
    val_examples = normalize_examples(val_raw, target_mean, target_std)

    train_ds = BrightnessDataset(train_examples, train_tf)
    val_ds = BrightnessDataset(val_examples, val_tf)

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

    model = EfficientNetBrightnessRegressor(num_outputs=len(TARGETS)).to(DEVICE)
    load_pretrained_backbone(model)

    optimizer = AdamW(
        [
            {"params": model.features.parameters(), "lr": LR_BACKBONE},
            {"params": model.avgpool.parameters(), "lr": LR_BACKBONE},
            {"params": model.head.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.HuberLoss()

    best_score = float("inf")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history_path = OUTPUT_DIR / "training_log.csv"
    with history_path.open("w", newline="") as log_handle:
        writer = csv.DictWriter(
            log_handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                *[f"mae_{target}" for target in TARGETS],
                *[f"r2_{target}" for target in TARGETS],
            ],
        )
        writer.writeheader()

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for images, labels in train_dl:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                preds = model(images)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())

            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_dl:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    preds = model(images)
                    val_loss += float(criterion(preds, labels).item())
                    all_preds.append(preds)
                    all_labels.append(labels)

            scheduler.step()
            train_loss /= max(len(train_dl), 1)
            val_loss /= max(len(val_dl), 1)

            preds_norm = torch.cat(all_preds, dim=0)
            labels_norm = torch.cat(all_labels, dim=0)
            preds_real = denormalize(preds_norm, target_mean, target_std)
            labels_real = denormalize(labels_norm, target_mean, target_std)

            mae = per_target_mae(preds_real, labels_real)
            r2 = per_target_r2(preds_real, labels_real)
            mae_str = "  ".join(f"{k}={v:.2f}" for k, v in mae.items())
            r2_str = "  ".join(f"{k}={v:.3f}" for k, v in r2.items())
            print(
                f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"MAE: {mae_str}  R2: {r2_str}"
            )

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    **{f"mae_{target}": round(mae[target], 6) for target in TARGETS},
                    **{f"r2_{target}": round(r2[target], 6) for target in TARGETS},
                }
            )
            log_handle.flush()

            score = float(np.mean([mae[target] for target in TARGETS]))
            if score < best_score:
                best_score = score
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "target_mean": target_mean.tolist(),
                        "target_std": target_std.tolist(),
                        "targets": TARGETS,
                        "image_size": IMG_SIZE,
                    },
                    OUTPUT_DIR / "best_efficientnet_brightness.pt",
                )
                print(f"  Saved best checkpoint to {OUTPUT_DIR / 'best_efficientnet_brightness.pt'}")

            if epoch % SAVE_PRED_EVERY == 0:
                save_predictions(val_raw, preds_real, labels_real, epoch)

    print(f"Done. Best mean target MAE: {best_score:.4f}")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train()
