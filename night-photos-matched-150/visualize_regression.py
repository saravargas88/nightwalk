"""
Comprehensive visualization of regression results and feature breakdown.
Input:  ../final_regression_results.csv
Output: regression_viz.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("../final_regression_results.csv")

actual    = df["grey_actual"].values
predicted = df["grey_predicted"].values
residuals = actual - predicted
r2   = r2_score(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── 1. Scatter: predicted vs actual ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
sc = ax1.scatter(actual, predicted, alpha=0.7, c=residuals, cmap="RdBu",
                 edgecolors="none", s=20, vmin=-50, vmax=50)
mn, mx = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
ax1.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="Perfect fit")
cb = plt.colorbar(sc, ax=ax1)
cb.set_label("Residual (actual − pred)", rotation=270, labelpad=15)
ax1.set_xlabel("Actual grayscale (night)")
ax1.set_ylabel("Predicted grayscale")
ax1.set_title(f"Predicted vs Actual\nR²={r2:.3f}  RMSE={rmse:.2f}")
ax1.legend(fontsize=8)

# ── Save scatter as standalone PNG ────────────────────────────────────────────
fig_scatter, ax_s = plt.subplots(figsize=(7, 7))
sc2 = ax_s.scatter(actual, predicted, alpha=0.7, c=residuals, cmap="RdBu",
                   edgecolors="none", s=25, vmin=-50, vmax=50)
ax_s.plot([mn, mx], [mn, mx], "k--", linewidth=1.5, label="Perfect fit")
cb2 = plt.colorbar(sc2, ax=ax_s)
cb2.set_label("Residual (actual − pred)", rotation=270, labelpad=15)
ax_s.set_xlabel("Actual grayscale (night)", fontsize=12)
ax_s.set_ylabel("Predicted grayscale", fontsize=12)
ax_s.set_title(f"Predicted vs Actual\nR²={r2:.3f}  RMSE={rmse:.2f}", fontsize=13)
ax_s.legend(fontsize=9)
fig_scatter.tight_layout()
fig_scatter.savefig("regression_scatter_standalone.png", dpi=150, bbox_inches="tight")
print("Saved → regression_scatter_standalone.png")
plt.close(fig_scatter)

# ── 2. Residuals vs actual ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(actual, residuals, alpha=0.4, s=20, color="steelblue", edgecolors="none")
ax2.axhline(0, color="red", linewidth=1, linestyle="--")
ax2.set_xlabel("Actual grayscale (night)")
ax2.set_ylabel("Residual (actual − predicted)")
ax2.set_title("Residuals vs Actual\n(systematic bias visible if curved)")

# ── 3. Residual distribution ──────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(residuals, bins=40, color="steelblue", edgecolor="white", linewidth=0.4)
ax3.axvline(0, color="red", linewidth=1, linestyle="--")
ax3.axvline(residuals.mean(), color="orange", linewidth=1.5, linestyle="-",
            label=f"Mean={residuals.mean():.2f}")
ax3.set_xlabel("Residual")
ax3.set_ylabel("Count")
ax3.set_title("Residual Distribution")
ax3.legend(fontsize=8)

# ── 4–6. Feature vs actual brightness (one per feature) ───────────────────────
features = ["tree", "streetlight", "storefront"]
colors   = ["#4CAF50", "#FF9800", "#2196F3"]

for i, (feat, col) in enumerate(zip(features, colors)):
    ax = fig.add_subplot(gs[1, i])
    vals = df[feat].values
    jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(vals))
    ax.scatter(vals + jitter, actual, alpha=0.6, s=15, color=col, edgecolors="none")

    # mean actual brightness per count bin
    for v in sorted(set(vals)):
        mask = vals == v
        if mask.sum() > 2:
            ax.plot(v, actual[mask].mean(), "k^", markersize=6, zorder=5)

    # regression line
    m, b = np.polyfit(vals, actual, 1)
    x_line = np.linspace(vals.min(), vals.max(), 100)
    ax.plot(x_line, m * x_line + b, color="red", linewidth=2, linestyle="--", label=f"y={m:+.2f}x+{b:.1f}")
    ax.legend(fontsize=7)

    corr = np.corrcoef(vals, actual)[0, 1]
    ax.set_xlabel(f"{feat.capitalize()} count (DINO)")
    ax.set_ylabel("Actual grayscale (night)" if i == 0 else "")
    ax.set_title(f"{feat.capitalize()} vs Night Brightness\nr={corr:.3f}")

fig.suptitle(f"Regression Analysis — DINO Counts → Night Brightness  (N={len(df)})",
             fontsize=14, fontweight="bold", y=1.01)

out = "regression_viz.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
