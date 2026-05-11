"""regression.py
Predict nighttime grayscale luminance from DINO feature counts.

Input:  dino_counts/dino_counts_informed_prompt_3-pairs.csv
Output: regression_results.csv   — per-image predictions vs ground truth
        regression_report.txt    — model coefficients + evaluation metrics
        regression_scatter.png   — predicted vs actual scatter plot
"""
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
NIGHT_DIR    = HERE / "night-photos-matched"
COUNTS_CSV   = HERE / "dino_counts" / "dino_counts_informed_prompt_3-pairs.csv"
OUT_CSV      = HERE / "regression_results.csv"
OUT_REPORT   = HERE / "regression_report.txt"
OUT_SCATTER  = HERE / "regression_scatter.png"

FEATURES = ["tree", "streetlight", "storefront"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def grayscale_mean(path: Path) -> float:
    img = Image.open(path).convert("L")
    return float(np.array(img).mean())

# ── Load DINO counts ──────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(COUNTS_CSV)))
print(f"Loaded {len(rows)} rows from {COUNTS_CSV.name}")

# ── Compute ground truth grayscale for each night photo ───────────────────────
valid, skipped = [], []
for row in rows:
    night_path = NIGHT_DIR / row["night_photo"]
    if not night_path.exists():
        skipped.append(row["night_photo"])
        continue
    row["grey"] = grayscale_mean(night_path)
    valid.append(row)

if skipped:
    print(f"Skipped {len(skipped)} missing night photos: {skipped[:5]}")
print(f"Using {len(valid)} matched pairs")

X = np.array([[int(r[f]) for f in FEATURES] for r in valid], dtype=float)
y = np.array([r["grey"] for r in valid], dtype=float)

# ── Models ────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge":            Ridge(alpha=1.0),
    "RandomForest":     RandomForestRegressor(n_estimators=200, random_state=42),
}

loo = LeaveOneOut()
report_lines = []

best_name, best_r2, best_preds = None, -np.inf, None

for name, model in models.items():
    X_in = X_scaled if name != "RandomForest" else X
    preds = cross_val_predict(model, X_in, y, cv=loo)
    r2   = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))

    report_lines.append(f"\n── {name} (LOO-CV) ──────────────────────────")
    report_lines.append(f"  R²:   {r2:.4f}")
    report_lines.append(f"  RMSE: {rmse:.4f}")

    # fit on full data for coefficients
    model.fit(X_in, y)
    if hasattr(model, "coef_"):
        for feat, coef in zip(FEATURES, model.coef_):
            report_lines.append(f"  {feat:>12}: {coef:+.4f}")
        report_lines.append(f"  {'intercept':>12}: {model.intercept_:+.4f}")
    elif hasattr(model, "feature_importances_"):
        for feat, imp in zip(FEATURES, model.feature_importances_):
            report_lines.append(f"  {feat:>12} importance: {imp:.4f}")

    if r2 > best_r2:
        best_r2, best_name, best_preds = r2, name, preds

# ── Save results CSV ──────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["night_photo", "day_image", *FEATURES, "grey_actual", "grey_predicted"])
    writer.writeheader()
    for row, pred in zip(valid, best_preds):
        writer.writerow({
            "night_photo":    row["night_photo"],
            "day_image":      row["image"],
            **{f: row[f] for f in FEATURES},
            "grey_actual":    round(row["grey"], 3),
            "grey_predicted": round(pred, 3),
        })

# ── Save report ───────────────────────────────────────────────────────────────
header = [
    f"Regression Report — DINO counts → night grayscale luminance",
    f"Ground truth: mean grayscale pixel value of night photo",
    f"Features: {FEATURES}",
    f"N: {len(valid)}",
    f"Best model: {best_name}  (R²={best_r2:.4f})",
]
full_report = "\n".join(header + report_lines)
OUT_REPORT.write_text(full_report)
print(full_report)

# ── Scatter plot (best model) ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y, best_preds, alpha=0.6, edgecolors="k", linewidths=0.4)
mn, mx = min(y.min(), best_preds.min()), max(y.max(), best_preds.max())
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="perfect fit")
ax.set_xlabel("Actual grayscale (night photo)")
ax.set_ylabel("Predicted grayscale")
ax.set_title(f"{best_name} — LOO-CV  R²={best_r2:.3f}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_SCATTER, dpi=150)
print(f"\nSaved scatter plot → {OUT_SCATTER}")
print(f"Saved predictions  → {OUT_CSV}")
print(f"Saved report       → {OUT_REPORT}")
