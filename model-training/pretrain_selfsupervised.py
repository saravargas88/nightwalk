"""pretrain_selfsupervised.py

Self-supervised pretraining of EfficientNet-B0 using SimCLR (via lightly)
on the 13k daytime images from urban-mosaic/washington-square.

Outputs:
  model-training/ssl-pretrain/best_ssl_backbone.pt   ← backbone weights only
  model-training/ssl-pretrain/training_log.csv

Usage:
    python pretrain_selfsupervised.py
    python pretrain_selfsupervised.py --epochs 100 --batch-size 256
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

import pandas as pd
from torch.utils.data import Dataset as TorchDataset


try:
    import lightly
    from lightly.data import LightlyDataset
    from lightly.loss import NTXentLoss
    from lightly.models.modules import SimCLRProjectionHead
except ImportError:
    raise ImportError(
        "lightly is required for self-supervised pretraining.\n"
        "Install it with: pip install lightly"
    )

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "urban-mosaic" / "washington-square"
OUTPUT_DIR = ROOT / "model-training" / "ssl-pretrain"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_IMG_SIZE = 224
DEFAULT_NUM_WORKERS = 8
DEFAULT_SEED = 42
PROJECTION_DIM = 128

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ── SimCLR augmentation ───────────────────────────────────────────────────────
# Strong augmentation is important for SimCLR — two views must be different
# enough that the model can't trivially match them without learning semantics
def make_simclr_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * img_size) | 1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Model ─────────────────────────────────────────────────────────────────────
class SimCLREfficientNet(nn.Module):
    def __init__(self, projection_dim: int = PROJECTION_DIM):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = base.classifier[1].in_features  # 1280

        self.features = base.features
        self.avgpool = base.avgpool
        self.projection_head = SimCLRProjectionHead(
            input_dim=in_features,
            hidden_dim=in_features,
            output_dim=projection_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.avgpool(self.features(x)), 1)
        return self.projection_head(x)

    def save_backbone(self, path: Path) -> None:
        """Save only the backbone weights (features + avgpool) for downstream use."""
        torch.save(
            {
                "features": self.features.state_dict(),
                "avgpool": self.avgpool.state_dict(),
            },
            path,
        )


def train(epochs: int, batch_size: int, lr: float, weight_decay: float, img_size: int, num_workers: int) -> None:
    torch.manual_seed(DEFAULT_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transform = make_simclr_transform(img_size)

    # LightlyDataset handles producing two augmented views per image automatically
    train_csv = ROOT / "splits" / "efficientnet_train_images.csv"
    train_df = pd.read_csv(train_csv)
    valid_paths = [
        IMAGE_DIR / row["image"]
        for _, row in train_df.iterrows()
        if (IMAGE_DIR / row["image"]).exists()
    ]
    print(f"Found {len(valid_paths)} valid images from efficientnet_train_images.csv")

    class PathListDataset(TorchDataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            # return two views like lightly expects
            return [self.transform(img), self.transform(img)], 0, str(self.paths[idx])

    dataset = PathListDataset(valid_paths, transform)


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE != "cpu"),
        drop_last=True,
    )

    model = SimCLREfficientNet(projection_dim=PROJECTION_DIM).to(DEVICE)
    criterion = NTXentLoss(temperature=0.07)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    log_path = OUTPUT_DIR / "training_log.csv"

    with log_path.open("w", newline="") as log_handle:
        writer = csv.DictWriter(log_handle, fieldnames=["epoch", "loss"])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                views, _, _ = batch
                x0, x1 = views[0].to(DEVICE), views[1].to(DEVICE)

                optimizer.zero_grad()
                z0 = model(x0)
                z1 = model(x1)
                loss = criterion(z0, z1)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            epoch_loss /= max(n_batches, 1)

            print(f"Epoch {epoch:03d}/{epochs}  loss={epoch_loss:.4f}")
            writer.writerow({"epoch": epoch, "loss": round(epoch_loss, 6)})
            log_handle.flush()

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                backbone_path = OUTPUT_DIR / "best_ssl_backbone.pt"
                model.save_backbone(backbone_path)
                print(f"  Saved best backbone → {backbone_path}")

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"Backbone weights saved to {OUTPUT_DIR / 'best_ssl_backbone.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-supervised SimCLR pretraining on daytime images.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )