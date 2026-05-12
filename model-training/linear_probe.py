"""linear_probe.py

Extracts frozen EfficientNet-B0 embeddings (1280-dim) from daytime images
and trains Ridge regression to predict nighttime brightness.

Optionally concatenates tabular features already in the brightness CSV
(DINO counts, bbox areas) alongside the embedding.

Results use the same k-fold split and metric as finetune_brightness.py
so numbers are directly comparable.

Usage:
    python linear_probe.py --backbone imagenet
    python linear_probe.py --backbone dino_counts --extra-features
    python linear_probe.py --backbone ssl --metric gray_mean_zscore --extra-features
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "splits" / "train_split.csv"
TEST_CSV = ROOT / "splits" / "test_split.csv"
BRIGHTNESS_CSV = (
    ROOT / "brightnessmetricexperiments"
    / "experiment_outputs"
    / "paired_dataset_with_brightness.csv"
)
DAY_IMAGE_ROOT = ROOT / "urban-mosaic" / "washington-square"

DINO_CHECKPOINT = ROOT / "model-training" / "best_efficientnet_multihead.pt"
SSL_CHECKPOINT = ROOT / "model-training" / "ssl-pretrain" / "best_ssl_backbone.pt"
OUTPUT_BASE = ROOT / "model-training" / "linear-probe-runs"

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
RANDOM_SEED = 42
DEFAULT_METRIC = "gray_mean_zscore"
DEFAULT_FOLDS = 5
RIDGE_ALPHA = 1.0

TABULAR_FEATURES = [
    "dino_count_tree",
    "dino_count_streetlight",
    "dino_count_storefront",
    "bbox_area_sum_tree",
    "bbox_area_sum_streetlight",
    "bbox_area_sum_storefront",
    "bbox_area_sum_total",
]

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
    tabular: list[float] = field(default_factory=list)


class ImageDataset(Dataset):
    def __init__(self, examples: list[Example], transform: transforms.Compose):
        self.examples = examples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ex = self.examples[idx]
        image = Image.open(ex.image_path).convert("RGB")
        return self.transform(image), idx


val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_examples(split_csv: Path, metric: str) -> list[Example]:
    split_df = pd.read_csv(split_csv)
    brightness_df = pd.read_csv(BRIGHTNESS_CSV)

    split_df = split_df[
        split_df["day_image"].notna() & (split_df["day_image"].str.strip() != "")
    ]

    tabular_cols = [c for c in TABULAR_FEATURES if c in brightness_df.columns]
    merge_cols = ["night_photo", "day_image", metric] + tabular_cols
    merged = split_df.merge(brightness_df[merge_cols], on=["night_photo", "day_image"], how="inner")

    examples = []
    for _, row in merged.iterrows():
        day_path = DAY_IMAGE_ROOT / row["day_image"]
        if not day_path.exists():
            continue
        examples.append(Example(
            image_path=str(day_path),
            day_image=row["day_image"],
            night_photo=row["night_photo"],
            target=float(row[metric]),
            tabular=[float(row[c]) for c in tabular_cols],
        ))
    return examples


def make_kfold_splits(
    examples: list[Example], n_folds: int, seed: int
) -> list[tuple[list[Example], list[Example]]]:
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
class EfficientNetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.avgpool = base.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.avgpool(self.features(x)), 1)


def load_backbone(model: EfficientNetEmbedder, backbone: str) -> None:
    if backbone == "imagenet":
        print("  Using ImageNet backbone")
        return

    if backbone == "dino_counts":
        if not DINO_CHECKPOINT.exists():
            raise FileNotFoundError(f"Checkpoint not found: {DINO_CHECKPOINT}")
        state = torch.load(DINO_CHECKPOINT, map_location="cpu")
        state = state["model_state_dict"]
        remapped = {}
        for key, value in state.items():
            if key.startswith("backbone."):
                remapped["features." + key[len("backbone."):]] = value
            elif key.startswith("features."):
                remapped[key] = value
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        print(f"  Loaded dino_counts backbone — missing={len(missing)} unexpected={len(unexpected)}")
        return

    if backbone == "ssl":
        if not SSL_CHECKPOINT.exists():
            raise FileNotFoundError(f"Checkpoint not found: {SSL_CHECKPOINT}")
        state = torch.load(SSL_CHECKPOINT, map_location="cpu")
        model.features.load_state_dict(state["features"], strict=True)
        model.avgpool.load_state_dict(state["avgpool"], strict=True)
        print(f"  Loaded SSL backbone")
        return

    raise ValueError(f"Unknown backbone: {backbone}")


# ── Embedding extraction ───────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model: EfficientNetEmbedder, examples: list[Example]) -> np.ndarray:
    model.eval()
    ds = ImageDataset(examples, val_tf)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    all_embeddings = []
    for images, _ in dl:
        images = images.to(DEVICE)
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ── Main ──────────────────────────────────────────────────────────────────────
def run(backbone: str, metric: str, n_folds: int, seed: int, extra_features: bool) -> None:
    print(f"\n{'='*60}")
    print(f"Backbone: {backbone}  |  metric: {metric}  |  extra_features: {extra_features}")
    print(f"{'='*60}")

    print("Loading data...")
    train_examples = load_examples(TRAIN_CSV, metric)
    test_examples = load_examples(TEST_CSV, metric)
    print(f"Train: {len(train_examples)}  Test: {len(test_examples)}")

    print(f"Building EfficientNet embedder on {DEVICE}...")
    embedder = EfficientNetEmbedder().to(DEVICE)
    load_backbone(embedder, backbone)
    for param in embedder.parameters():
        param.requires_grad = False

    print("Extracting embeddings for all train examples...")
    all_train_emb = extract_embeddings(embedder, train_examples)
    print("Extracting embeddings for test examples...")
    test_emb = extract_embeddings(embedder, test_examples)
    print(f"Embedding shape: {all_train_emb.shape}")

    if extra_features and train_examples[0].tabular:
        train_tab = np.array([ex.tabular for ex in train_examples])
        test_tab = np.array([ex.tabular for ex in test_examples])
        all_train_X = np.concatenate([all_train_emb, train_tab], axis=1)
        test_X = np.concatenate([test_emb, test_tab], axis=1)
        print(f"Feature shape with tabular: {all_train_X.shape}")
    else:
        all_train_X = all_train_emb
        test_X = test_emb

    all_train_y = np.array([ex.target for ex in train_examples])
    test_y = np.array([ex.target for ex in test_examples])

    folds = make_kfold_splits(train_examples, n_folds, seed)

    run_dir = OUTPUT_BASE / backbone / ("with_tabular" if extra_features else "emb_only")
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_val_metrics = []
    fold_test_metrics = []

    for fold_idx, (train_exs, val_exs) in enumerate(folds):
        train_idx = [i for i, ex in enumerate(train_examples) if ex.day_image in {e.day_image for e in train_exs}]
        val_idx   = [i for i, ex in enumerate(train_examples) if ex.day_image in {e.day_image for e in val_exs}]

        X_train, y_train = all_train_X[train_idx], all_train_y[train_idx]
        X_val,   y_val   = all_train_X[val_idx],   all_train_y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(test_X)

        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(X_train_s, y_train)

        val_preds  = ridge.predict(X_val_s)
        test_preds = ridge.predict(X_test_s)

        val_metrics  = compute_metrics(y_val, val_preds)
        test_metrics = compute_metrics(test_y, test_preds)
        fold_val_metrics.append(val_metrics)
        fold_test_metrics.append(test_metrics)

        print(
            f"  Fold {fold_idx}  "
            f"val  mae={val_metrics['mae']:.4f} r2={val_metrics['r2']:.4f}  |  "
            f"test mae={test_metrics['mae']:.4f} r2={test_metrics['r2']:.4f}"
        )

    val_summary_path = run_dir / "fold_summary.csv"
    with val_summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "mae", "rmse", "r2"])
        writer.writeheader()
        for i, m in enumerate(fold_val_metrics):
            writer.writerow({"fold": i, **{k: round(v, 6) for k, v in m.items()}})

    test_summary_path = run_dir / "test_summary.csv"
    with test_summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "mae", "rmse", "r2"])
        writer.writeheader()
        for i, m in enumerate(fold_test_metrics):
            writer.writerow({"fold": i, **{k: round(v, 6) for k, v in m.items()}})

    val_mean  = {k: float(np.mean([m[k] for m in fold_val_metrics]))  for k in ["mae", "rmse", "r2"]}
    val_std   = {k: float(np.std( [m[k] for m in fold_val_metrics]))  for k in ["mae", "rmse", "r2"]}
    test_mean = {k: float(np.mean([m[k] for m in fold_test_metrics])) for k in ["mae", "rmse", "r2"]}
    test_std  = {k: float(np.std( [m[k] for m in fold_test_metrics])) for k in ["mae", "rmse", "r2"]}

    print(f"\n── {n_folds}-fold summary ──")
    print(f"  {'':6}  {'MAE':>20}  {'RMSE':>20}  {'R²':>20}")
    for split, mean, std in [("Val", val_mean, val_std), ("Test", test_mean, test_std)]:
        print(
            f"  {split:6}  "
            f"{mean['mae']:.4f} ± {std['mae']:.4f}  "
            f"{mean['rmse']:.4f} ± {std['rmse']:.4f}  "
            f"{mean['r2']:.4f} ± {std['r2']:.4f}"
        )

    print(f"\nSaved results to {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probe on EfficientNet embeddings.")
    parser.add_argument("--backbone", choices=["imagenet", "dino_counts", "ssl"], default="imagenet")
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--extra-features", action="store_true",
                        help="Concatenate DINO counts + bbox areas alongside the embedding.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device: {DEVICE}")
    run(
        backbone=args.backbone,
        metric=args.metric,
        n_folds=args.folds,
        seed=args.seed,
        extra_features=args.extra_features,
    )
