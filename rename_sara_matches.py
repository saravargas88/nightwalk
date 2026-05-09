"""
Renames all night images in sara-matched/ to have a 'sara-' prefix
and updates the night_photo column in matches_sara.csv to match.
"""

import csv
import shutil
from pathlib import Path

folder = Path("sara-matched")
csv_path = folder / "matches_sara.csv"

# Read existing CSV
with open(csv_path) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# Rename images and update CSV rows
for row in rows:
    old_name = row["night_photo"]
    new_name = "sara-" + old_name

    old_path = folder / old_name
    new_path = folder / new_name

    if old_path.exists():
        shutil.move(str(old_path), str(new_path))
        print(f"Renamed: {old_name} → {new_name}")
    elif new_path.exists():
        print(f"Already renamed: {new_name}")
    else:
        print(f"WARNING: file not found: {old_name}")

    row["night_photo"] = new_name

# Write updated CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone. CSV updated at {csv_path}")
