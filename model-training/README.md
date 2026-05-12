# model-training

This directory contains all training, evaluation, and baseline scripts for **NightWalk** — predicting nighttime street luminance from daytime Street View Images (SVI).

---

## Directory layout

```
model-training/
├── run_experiments.py          ← regression ablation orchestrator (backbone × n_train)
├── run_experiments_new.py      ← classification ablation orchestrator (backbone × n_train)
│
├── pretraining/                ← train the backbone before brightness fine-tuning
│   ├── train_efficientnet_multihead.py   HPC script: trains EfficientNet on DINO counts
│   │                                     (tree / streetlight / storefront). Produces
│   │                                     best_efficientnet_multihead.pt ("dino_counts" backbone)
│   ├── pretrain_selfsupervised.py        SimCLR self-supervised pretraining on 13k day images.
│   │                                     Produces ssl-pretrain/best_ssl_backbone.pt
│   └── train_counts_small.py             Older / lighter version of multihead training.
│                                         Kept as reference; superseded by multihead.
│
├── regression/                 ← predict a continuous brightness score
│   ├── train_brightness_score.py  Multi-target regressor (gray_mean, luma_mean, value_mean,
│   │                               gray_mean_zscore simultaneously). Simple 80/20 split.
│   │                               **This is the script that reached R²=0.47 at 45 epochs.**
│   │                               Outputs → brightness-regression-run/
│   └── finetune_brightness.py     Single-target regressor with 5-fold cross-validation.
│                                   Called by run_experiments.py. Supports imagenet /
│                                   dino_counts / ssl backbone. Outputs → finetune-runs/
│
├── classification/             ← predict 1-of-4 brightness bins
│   └── train_brightness_class.py  4-class classifier (very_dark / dark / bright / very_bright).
│                                   Bins are quartiles of gray_mean computed from the train
│                                   split only (no leakage). 45 epochs, slow LR.
│                                   Called by run_experiments_new.py.
│                                   Outputs → brightness-class-runs/
│
└── eval/                       ← evaluation and analysis
    ├── eval_brightness_checkpoint.py   Evaluate train_brightness_score.py checkpoint on test split.
    ├── linear_probe.py                 Frozen EfficientNet embedding + Ridge regression baseline.
    │                                   Supports --extra-features to add DINO counts + bbox areas.
    │                                   Directly comparable to finetune_brightness.py results.
    ├── visualize_training.py           Plots per-target MAE curves from multihead training logs.
    ├── predict_night_brightness.py     Early-stage inference script (kept for reference).
    └── train_brightness_levels_archive.py  Older classification script superseded by
                                            classification/train_brightness_class.py.
                                            Safe to delete once results are confirmed.
```

---

## Data flow

```
urban-mosaic/washington-square/     ~100k daytime SVI images (13k used for backbone)
        ↓
pretraining/train_efficientnet_multihead.py
        ↓
best_efficientnet_multihead.pt      "dino_counts" backbone checkpoint
        ↓
regression/train_brightness_score.py        (or finetune_brightness.py for k-fold)
        ↓
brightness-regression-run/best_efficientnet_brightness.pt
        ↓
eval/eval_brightness_checkpoint.py  → test set metrics
```

Day-night pairs live in:
```
splits/train_split.csv    ~780 pairs
splits/test_split.csv     ~200 pairs
brightnessmetricexperiments/experiment_outputs/paired_dataset_with_brightness.csv
    → gray_mean, luma_mean, value_mean, gray_mean_zscore + bbox/DINO features per pair
```

---

## Running experiments

### 1. Backbone pretraining (do once)
```bash
# Train on DINO count labels — run on HPC
python model-training/pretraining/train_efficientnet_multihead.py

# Or train SimCLR SSL backbone
python model-training/pretraining/pretrain_selfsupervised.py --epochs 100
```

### 2. Regression ablation (Table 1 in paper)
```bash
# Full sweep: imagenet × dino_counts × ssl, across n_train sizes, 5-fold CV
python model-training/run_experiments.py

# Quick subset
python model-training/run_experiments.py --backbones imagenet dino_counts --skip-ssl-pretrain
```
Outputs → `model-training/finetune-runs/`  
Summary → `model-training/results_summary.csv`

### 3. Best regression model (what reached R²=0.47)
```bash
python model-training/regression/train_brightness_score.py \
    --epochs 45 --lr-backbone 1e-5 --lr-head 1e-4
```
Evaluate on test split:
```bash
python model-training/eval/eval_brightness_checkpoint.py \
    --image-dir urban-mosaic/washington-square
```

### 4. Linear probe baseline (Table 1 baselines)
```bash
# Embedding only
python model-training/eval/linear_probe.py --backbone imagenet

# Embedding + DINO counts + bbox features
python model-training/eval/linear_probe.py --backbone dino_counts --extra-features
```

### 5. Classification ablation (Table 2 in paper)
```bash
# Full sweep: imagenet × dino_counts, full / 600 / 400 training examples
python model-training/run_experiments_new.py --epochs 45

# Single condition
python model-training/run_experiments_new.py --backbones dino_counts --n-trains full
```
Outputs → `model-training/brightness-class-runs/`  
Summary → `model-training/brightness_class_results.csv`

---

## Checkpoints and outputs

| File | What it is |
|------|-----------|
| `best_efficientnet_multihead.pt` | "dino_counts" backbone (tree/streetlight/storefront counts) |
| `ssl-pretrain/best_ssl_backbone.pt` | SimCLR SSL backbone |
| `brightness-regression-run/best_efficientnet_brightness.pt` | Best multi-target regression model |
| `finetune-runs/<backbone>/n<N>/fold_<k>/best_model.pt` | Per-fold checkpoints from k-fold ablation |
| `brightness-class-runs/<backbone>/n<tag>/best_efficientnet_brightness_class.pt` | Classification checkpoints |

---

## Key numbers so far

| Model | Backbone | Epochs | R² (gray_mean_zscore) |
|-------|----------|--------|-----------------------|
| Linear (DINO counts) | — | — | ~0.06 |
| EfficientNet multi-target | dino_counts | 45 | **0.47** |

---

## What still needs to run

See paper experiment plan below. The critical missing pieces are:
1. `run_experiments.py` full sweep → k-fold results for all backbones (Table 1)
2. `linear_probe.py --extra-features` → linear baseline with bbox (Table 1 baseline row)
3. `run_experiments_new.py` → classification ablation (Table 2)
4. `train_brightness_score.py` with `--backbone imagenet` for fair comparison
