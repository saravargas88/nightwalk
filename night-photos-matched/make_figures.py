"""make_figures.py
Generates report-quality figures from regression_results.csv.

Outputs (all in the same directory):
  fig_scatter.png        — predicted vs actual, colored by |error|
  fig_features.png       — per-feature scatter (3 subplots)
  fig_residuals.png      — residuals vs predicted
  fig_coefficients.png   — Ridge coefficient bar chart
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

HERE = Path(__file__).parent

FEATURES = ["tree", "streetlight", "storefront"]
FEATURE_LABELS = {"tree": "Trees", "streetlight": "Streetlights", "storefront": "Storefronts"}
PALETTE = {"pos": "#2166ac", "neg": "#d6604d", "neutral": "#4d4d4d"}
GREY = "#888888"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(HERE / "regression_results.csv")
df["error"] = df["grey_predicted"] - df["grey_actual"]
df["abs_error"] = df["error"].abs()

y_actual = df["grey_actual"].values
y_pred   = df["grey_predicted"].values
r2       = r2_score(y_actual, y_pred)
rmse     = np.sqrt(mean_squared_error(y_actual, y_pred))
n        = len(df)

# Refit Ridge on full data to get coefficients
X = df[FEATURES].values.astype(float)
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)
ridge  = Ridge(alpha=1.0).fit(X_sc, y_actual)

# ── 1. Main scatter ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5.5))

norm  = mcolors.Normalize(vmin=0, vmax=df["abs_error"].quantile(0.95))
cmap  = plt.cm.YlOrRd
sc    = ax.scatter(y_actual, y_pred,
                   c=df["abs_error"], cmap=cmap, norm=norm,
                   s=40, edgecolors="white", linewidths=0.4, zorder=3)

lo = min(y_actual.min(), y_pred.min()) - 3
hi = max(y_actual.max(), y_pred.max()) + 3
ax.plot([lo, hi], [lo, hi], "--", color=GREY, linewidth=1.2, label="Ideal (y = x)", zorder=2)

cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
cb.set_label("Absolute error", fontsize=10)

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Actual night brightness (mean grayscale)")
ax.set_ylabel("Predicted night brightness")
ax.set_title(f"Ridge Regression — LOO-CV\n"
             f"$R^2$ = {r2:.3f}   RMSE = {rmse:.1f}   N = {n}")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "fig_scatter.png", dpi=180)
plt.close(fig)
print("Saved fig_scatter.png")

# ── 2. Per-feature scatter ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for ax, feat in zip(axes, FEATURES):
    x = df[feat].values
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(x))

    ax.scatter(x + jitter, y_actual,
               alpha=0.55, s=30, color=PALETTE["pos"],
               edgecolors="white", linewidths=0.3)

    # Trend line
    m, b = np.polyfit(x, y_actual, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, m * xs + b, color=PALETTE["neg"], linewidth=1.8, zorder=4)

    r2_feat = r2_score(y_actual, m * x + b)
    ax.set_xlabel(f"DINO count — {FEATURE_LABELS[feat]}")
    ax.set_title(f"{FEATURE_LABELS[feat]}\n$R^2$ = {r2_feat:.3f}   slope = {m:+.1f}")
    ax.set_xticks(sorted(set(x.astype(int))))

axes[0].set_ylabel("Actual night brightness (mean grayscale)")
fig.suptitle("Per-feature relationship: DINO counts vs. night brightness",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(HERE / "fig_features.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved fig_features.png")

# ── 3. Residual plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4))

ax.axhline(0, color=GREY, linewidth=1.2, linestyle="--", zorder=2)
ax.scatter(y_pred, df["error"],
           c=df["abs_error"], cmap=cmap, norm=norm,
           s=35, edgecolors="white", linewidths=0.4, zorder=3)

ax.set_xlabel("Predicted night brightness")
ax.set_ylabel("Residual (predicted − actual)")
ax.set_title(f"Residuals vs. Predicted\nRMSE = {rmse:.1f}   N = {n}")
fig.tight_layout()
fig.savefig(HERE / "fig_residuals.png", dpi=180)
plt.close(fig)
print("Saved fig_residuals.png")

# ── 4. Coefficient bar chart ──────────────────────────────────────────────────
coefs = ridge.coef_
colors = [PALETTE["neg"] if c < 0 else PALETTE["pos"] for c in coefs]
labels = [FEATURE_LABELS[f] for f in FEATURES]

fig, ax = plt.subplots(figsize=(5, 3.2))
bars = ax.barh(labels, coefs, color=colors, edgecolor="white", height=0.5)
ax.axvline(0, color=GREY, linewidth=1.0)

for bar, val in zip(bars, coefs):
    xpos = val + (0.3 if val >= 0 else -0.3)
    ha   = "left" if val >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}", va="center", ha=ha, fontsize=10)

ax.set_xlabel("Standardized coefficient")
ax.set_title("Ridge Regression Coefficients\n(standardized features → night brightness)")
fig.tight_layout()
fig.savefig(HERE / "fig_coefficients.png", dpi=180)
plt.close(fig)
print("Saved fig_coefficients.png")

print("\nAll figures saved.")
