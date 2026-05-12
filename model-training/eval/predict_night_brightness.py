import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt


pairs    = pd.read_csv("paired_fixed.csv")
features = pd.read_csv("../dino_experiments/dino_counts/dino_counts_informed_prompt_3-pairs.csv")

merged = features.merge(pairs[["day_image", "night_grey"]], 
                        left_on="image", right_on="day_image", how="inner")

print(f"Matched rows: {len(merged)}")

X = merged[["tree", "streetlight", "storefront"]].values
y = merged["night_grey"].values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
print(f"R²:  {r2_score(y, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y, y_pred):.4f}")
print("\nCoefficients:")
for name, coef in zip(["tree", "streetlight", "storefront"], model.coef_):
    print(f"  {name:>12}: {coef:.4f}")
print(f"  {'intercept':>12}: {model.intercept_:.4f}")

print(merged["night_grey"].describe())
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# for ax, feature in zip(axes, ["tree", "streetlight", "storefront"]):
#     ax.scatter(merged[feature], merged["night_grey"], alpha=0.6)
#     ax.set_xlabel(feature)
#     ax.set_ylabel("night_grey")
#     ax.set_title(f"{feature} vs nighttime brightness")

# plt.tight_layout()
# plt.savefig("feature_contributions.png")
# plt.show()