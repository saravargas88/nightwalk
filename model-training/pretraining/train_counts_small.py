"""train_small.py
Fine-tunes EfficientNet-B0 (fully unfrozen) to predict
tree, streetlight, storefront counts from ~1500 images.
Designed to run comfortably on CPU or Apple Silicon (MPS).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent.parent
IMAGE_ROOT = ROOT / "urban-mosaic" / "washington-square"
CSV_PATH   = ROOT / "dino_experiments" / "dino_counts" / "dino_counts_informed_prompt_3.csv"
PREDS_DIR  = Path(__file__).resolve().parent.parent / "val_predictions"
TARGETS    = ["tree", "streetlight", "storefront"]
N_SAMPLES  = 1500          # set to None to use full dataset
BATCH_SIZE = 16
NUM_EPOCHS = 50
LR         = 1e-4          # lower LR since full backbone is unfrozen
IMG_SIZE   = 224
NUM_WORKERS = 4
VAL_SPLIT  = 0.2
RANDOM_SEED = 42
SAVE_PREDS_EVERY = 5       # save val predictions every N epochs

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── Dataset ───────────────────────────────────────────────────────────────────
class CountDataset(Dataset):
    def __init__(self, df, image_root, targets, transform=None):
        self.df         = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.targets    = targets
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.image_root / row["image"]
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.targets].values.astype(np.float32))
        return image, labels

# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(p=0.1),
    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    # transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Model ─────────────────────────────────────────────────────────────────────
class EfficientNetRegressor(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Fully unfrozen — entire network trains end-to-end
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_outputs),
            nn.ReLU(),           # counts are non-negative
        )

    def forward(self, x):
        return self.backbone(x)

# ── Per-target MAE ────────────────────────────────────────────────────────────
def per_target_mae(preds, labels, targets):
    maes = (preds - labels).abs().mean(dim=0)
    return {t: maes[i].item() for i, t in enumerate(targets)}

# ── Save val predictions to CSV ───────────────────────────────────────────────
def save_predictions(val_df, all_preds, targets, epoch):
    PREDS_DIR.mkdir(exist_ok=True)
    preds_np = all_preds.numpy()
    out = val_df[["image"] + targets].copy().reset_index(drop=True)
    for i, t in enumerate(targets):
        out[f"pred_{t}"] = preds_np[:, i].round(2)
    out_path = PREDS_DIR / f"val_preds_epoch_{epoch:03d}.csv"
    out.to_csv(out_path, index=False)
    print(f"  ✓ Saved val predictions to {out_path}")

# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    df = pd.read_csv(CSV_PATH)
    df[TARGETS] = df[TARGETS].fillna(0)
    if N_SAMPLES and N_SAMPLES < len(df):
        df = df.sample(n=N_SAMPLES, random_state=RANDOM_SEED)
        print(f"Subsampled to {N_SAMPLES} examples")

    print("\nLabel stats:")
    print(df[TARGETS].describe().loc[["mean", "std", "max"]])
    zero_pct = (df[TARGETS] == 0).mean() * 100
    print("\n% zeros per target:")
    print(zero_pct.round(1).to_string())
    print()

    train_df, val_df = train_test_split(
        df, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    train_ds = CountDataset(train_df, IMAGE_ROOT, TARGETS, train_tf)
    val_ds   = CountDataset(val_df,   IMAGE_ROOT, TARGETS, val_tf)

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE != "cpu")
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE != "cpu")
    )

    model     = EfficientNetRegressor(num_outputs=len(TARGETS)).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-3)  # all params unfrozen
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.HuberLoss()

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for images, labels in train_dl:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds     = model(images)
                val_loss += criterion(preds, labels).item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        train_loss /= len(train_dl)
        val_loss   /= len(val_dl)
        scheduler.step()

        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        mae        = per_target_mae(all_preds, all_labels, TARGETS)
        mae_str    = "  ".join(f"{t}={v:.2f}" for t, v in mae.items())

        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"MAE: {mae_str}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_efficientnet_small.pt")
            print(f"  ✓ Saved best model")

        # Save val predictions every N epochs
        if (epoch + 1) % SAVE_PREDS_EVERY == 0:
            save_predictions(val_df, all_preds, TARGETS, epoch + 1)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print("Model saved to best_efficientnet_small.pt")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train()