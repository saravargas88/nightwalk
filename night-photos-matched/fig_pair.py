"""fig_pair.py
Side-by-side day/night pair figure for the midterm report.
Uses 122.JPG — tree=4, streetlight=5, storefront=2, actual=86.8, pred=86.8
"""
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE       = Path(__file__).parent
URBAN      = HERE.parent / "urban-mosaic" / "washington-square"
NIGHT_DIR  = HERE / "night-photos-matched"

# ── Pair data (from regression_results.csv) ───────────────────────────────────
NIGHT_FILE = "122.JPG"
DAY_REL    = ("0/20161114/8b82a42e5a149e22/"
              "cds-8b82a42e5a149e22-20161114-1047.raw/3/"
              "dr5rsnxm7bnk-dr5rsnxkdpj2-cds-8b82a42e5a149e22-20161114-1047-37.jpg")
COUNTS     = {"Trees": 4, "Streetlights": 5, "Storefronts": 2}
ACTUAL     = 86.829
PREDICTED  = 86.778

# ── Load images ───────────────────────────────────────────────────────────────
day_img   = Image.open(URBAN / DAY_REL).convert("RGB")
night_img = Image.open(NIGHT_DIR / NIGHT_FILE).convert("RGB")

# Crop to same height ratio (landscape → square-ish)
def center_crop(img, ratio=0.65):
    w, h = img.size
    new_h = int(h * ratio)
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))

day_img   = center_crop(day_img,   ratio=0.6)
night_img = center_crop(night_img, ratio=0.6)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax_day, ax_night) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#FFFFFF")

for ax in (ax_day, ax_night):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# Day image
ax_day.imshow(np.array(day_img))
ax_day.set_title("Daytime (Street View)", color="black", fontsize=13, pad=8, fontweight="bold")

# DINO count badges — top-left stack
COLORS = {"Trees": "#4caf50", "Streetlights": "#ffd54f", "Storefronts": "#ef9a9a"}
ICONS  = {"Trees": "🌳", "Streetlights": "💡", "Storefronts": "🏪"}
for i, (label, count) in enumerate(COUNTS.items()):
    y_pos = 0.97 - i * 0.12
    ax_day.text(
        0.03, y_pos,
        f"{ICONS[label]}  {label}: {count}",
        transform=ax_day.transAxes,
        fontsize=11, color="white", fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS[label],
                  alpha=0.82, edgecolor="none"),
    )

# Night image
ax_night.imshow(np.array(night_img))
ax_night.set_title("Nighttime Image", color="black", fontsize=13, pad=8, fontweight="bold")

# Brightness box — bottom center
brightness_text = (
    f"Predicted brightness:  {PREDICTED:.1f}\n"
    f"Actual brightness:       {ACTUAL:.1f}\n"
    f"Error:                          {abs(PREDICTED - ACTUAL):.1f}"
)
ax_night.text(
    0.5, 0.06,
    brightness_text,
    transform=ax_night.transAxes,
    fontsize=10.5, color="white",
    va="bottom", ha="center",
    family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#222222", alpha=0.88, edgecolor="#666666"),
)

plt.suptitle(
    "Day–Night Pair  ·  DINO Counts (day) → Predicted Night Brightness",
    color="black", fontsize=14, fontweight="bold", y=1.01,
)
plt.tight_layout(pad=1.2)

out = HERE / "fig_pair.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
