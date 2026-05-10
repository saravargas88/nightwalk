"""
Remap matches.csv sequential names (1.JPG) to IMG_XXXX names via XMP timestamps.

Usage:
    python remap_matches.py

Reads:
    matches.csv
    night-photos-all/*.xmp

Outputs:
    matches_remapped.csv   — night_photo updated to IMG_XXXX where possible,
                             kept as N.JPG where not found (original export)
"""

import pandas as pd
from pathlib import Path
from xml.etree import ElementTree as ET
from datetime import datetime

PHOTO_DIR  = Path("night-photos-all")
INPUT_CSV  = Path("matches.csv")
OUTPUT_CSV = Path("matches_remapped.csv")

# ── Build timestamp → IMG_XXXX index from XMP files ──────────────────────────

ts_to_img = {}   # key: "2026:04:28 20:07:12" → "IMG_4883"

for xmp in sorted(PHOTO_DIR.glob("*.xmp")):
    tree = ET.parse(xmp)
    ts = None
    for el in tree.getroot().iter():
        if el.tag.endswith("DateCreated"):
            ts = el.text
            break
    if not ts:
        continue
    dt = datetime.fromisoformat(ts)
    key = dt.strftime("%Y:%m:%d %H:%M:%S")
    # First-one-wins for duplicates
    if key not in ts_to_img:
        ts_to_img[key] = xmp.stem   # e.g. "IMG_4883"

print(f"Indexed {len(ts_to_img)} unique XMP timestamps")

# ── Remap ─────────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
remapped = 0
kept = 0

def remap_name(row):
    global remapped, kept
    ts = str(row["night_taken"])
    if ts in ts_to_img:
        remapped += 1
        return ts_to_img[ts] + ".JPG"
    else:
        kept += 1
        return row["night_photo"]   # keep original N.JPG name

df["night_photo"] = df.apply(remap_name, axis=1)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Remapped to IMG_XXXX: {remapped}")
print(f"Kept as original:     {kept}")
print(f"Saved → {OUTPUT_CSV}")