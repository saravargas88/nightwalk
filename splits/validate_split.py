"""validate_splits.py

Checks integrity of the train/test splits against finetune_pairs.csv.

Verifications:
  1. No duplicate night_photo entries within train_split.csv
  2. No duplicate night_photo entries within test_split.csv
  3. No overlap between train and test splits (same night_photo in both)
  4. Every valid pair from finetune_pairs.csv appears in either train or test
  5. No invalid pairs (rows without day_image) leaked into either split

Usage:
    python splits/validate_splits.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "splits"
FINETUNE_CSV = SPLITS_DIR / "finetune_pairs.csv"
TRAIN_CSV = SPLITS_DIR / "train_split.csv"
TEST_CSV = SPLITS_DIR / "test_split.csv"

ID_COL = "night_photo"


def check(condition: bool, label: str) -> bool:
    status = "✓ PASS" if condition else "✗ FAIL"
    print(f"  {status}  {label}")
    return condition


def main() -> None:
    print("Loading CSVs...")
    finetune = pd.read_csv(FINETUNE_CSV)
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # Valid pairs = rows in finetune_pairs that actually have a day_image
    valid_pairs = finetune[
        finetune["day_image"].notna() & (finetune["day_image"].str.strip() != "")
    ]

    valid_ids = set(valid_pairs[ID_COL])
    train_ids = set(train[ID_COL])
    test_ids = set(test[ID_COL])

    print(f"\n  finetune_pairs.csv total rows:  {len(finetune)}")
    print(f"  Valid pairs (has day_image):     {len(valid_pairs)}")
    print(f"  train_split.csv rows:            {len(train)}")
    print(f"  test_split.csv rows:             {len(test)}")

    print("\n── Checks ───────────────────────────────────────────────")
    results = []

    # 1. No duplicates within train
    train_dupe_mask = train[ID_COL].duplicated(keep=False)
    train_dupes = int(train_dupe_mask.sum())
    results.append(check(train_dupes == 0, f"No duplicates in train_split ({train_dupes} found)"))
    if train_dupes > 0:
        dupe_vals = sorted(train.loc[train_dupe_mask, ID_COL].unique().tolist())
        print(f"    Duplicated ids: {dupe_vals}")

    # 2. No duplicates within test
    test_dupe_mask = test[ID_COL].duplicated(keep=False)
    test_dupes = int(test_dupe_mask.sum())
    results.append(check(test_dupes == 0, f"No duplicates in test_split ({test_dupes} found)"))
    if test_dupes > 0:
        dupe_vals = sorted(test.loc[test_dupe_mask, ID_COL].unique().tolist())
        print(f"    Duplicated ids: {dupe_vals}")

    # 3. No overlap between train and test
    overlap = train_ids & test_ids
    results.append(check(len(overlap) == 0, f"No train/test overlap ({len(overlap)} overlapping ids found)"))
    if overlap:
        print(f"    Overlapping ids: {sorted(overlap)}")

    # 4. All valid pairs accounted for
    combined_ids = train_ids | test_ids
    missing = valid_ids - combined_ids
    extra = combined_ids - valid_ids
    results.append(check(len(missing) == 0, f"All valid pairs present in splits ({len(missing)} missing)"))
    if missing:
        print(f"    Missing ids (first 10): {sorted(missing)[:10]}")
    results.append(check(len(extra) == 0, f"No extra ids in splits not from valid pairs ({len(extra)} unexpected)"))
    if extra:
        print(f"    Unexpected ids (first 10): {sorted(extra)[:10]}")

    # 5. No invalid pairs (no day_image) in either split
    invalid_ids = set(finetune[~finetune[ID_COL].isin(valid_ids)][ID_COL])
    invalid_in_train = train_ids & invalid_ids
    invalid_in_test = test_ids & invalid_ids
    results.append(check(len(invalid_in_train) == 0, f"No unpaired night images in train ({len(invalid_in_train)} found)"))
    if invalid_in_train:
        print(f"    Unpaired ids in train: {sorted(invalid_in_train)}")
    results.append(check(len(invalid_in_test) == 0, f"No unpaired night images in test ({len(invalid_in_test)} found)"))
    if invalid_in_test:
        print(f"    Unpaired ids in test: {sorted(invalid_in_test)}")

    # ── Final verdict ──────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────")
    n_passed = sum(results)
    n_total = len(results)
    if all(results):
        print(f"  All {n_total} checks passed. Splits are clean.")
    else:
        print(f"  {n_passed}/{n_total} checks passed. Fix issues above before training.")


if __name__ == "__main__":
    main()