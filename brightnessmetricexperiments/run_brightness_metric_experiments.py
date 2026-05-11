"""Run brightness metric and regression experiments for matched night/day pairs.

This script:
1. Loads matched night/day pairs from ``all-matches.csv``.
2. Joins DINO feature counts and bounding-box detections for each matched day image.
3. Computes several brightness targets from the extracted night photos.
4. Benchmarks multiple regression models and a logistic classifier with CV.
5. Writes compact CSV/Markdown outputs into ``experiment_outputs``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MATCHES_CSV = Path(__file__).resolve().parent.parent / "splits" /"all-matches.csv"
COUNTS_CSV = BASE_DIR / "dino-counts-trainset.csv"
BBOX_JSON = BASE_DIR / "brounding-boxes-trainset.json"
IMAGE_ROOT = BASE_DIR / "extracted_night_images" / "images"
OUTPUT_DIR = BASE_DIR / "experiment_outputs"

LABELS = ("tree", "streetlight", "storefront")
EPS = 1e-8
RNG_SEED = 42
N_FOLDS = 5
ANALYSIS_MAX_SIDE = 512


@dataclass
class RegressionResult:
    target: str
    feature_set: str
    model: str
    weighting: str
    r2: float
    rmse: float
    mae: float
    spearman: float


@dataclass
class LogisticResult:
    target: str
    feature_set: str
    model: str
    weighting: str
    accuracy: float
    balanced_accuracy: float
    auc: float
    log_loss: float


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_matches() -> list[dict[str, str]]:
    rows = read_csv_rows(MATCHES_CSV)
    matched = [row for row in rows if row["skipped"] == "False" and row["day_image"]]

    # A few night images appear more than once. Keep the closest day match.
    by_night: dict[str, dict[str, str]] = {}
    for row in matched:
        night = row["night_photo"]
        best = by_night.get(night)
        if best is None:
            by_night[night] = row
            continue
        current_distance = float(row["distance_m"] or "inf")
        best_distance = float(best["distance_m"] or "inf")
        if current_distance < best_distance:
            by_night[night] = row

    return sorted(by_night.values(), key=lambda row: row["night_photo"])


def load_counts() -> dict[str, dict[str, float]]:
    rows = read_csv_rows(COUNTS_CSV)
    accum: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        image = row["image"]
        accum.setdefault(image, []).append({label: float(row[label]) for label in LABELS})

    counts: dict[str, dict[str, float]] = {}
    for image, entries in accum.items():
        counts[image] = {
            f"dino_count_{label}": float(np.mean([entry[label] for entry in entries]))
            for label in LABELS
        }
    return counts


def load_bbox_features() -> dict[str, dict[str, float]]:
    raw = json.loads(BBOX_JSON.read_text())
    features: dict[str, dict[str, float]] = {}

    for image, detections in raw.items():
        row: dict[str, float] = {"bbox_total_boxes": float(len(detections))}
        total_area = 0.0
        for label in LABELS:
            row[f"bbox_count_{label}"] = 0.0
            row[f"bbox_area_sum_{label}"] = 0.0
            row[f"bbox_area_mean_{label}"] = 0.0
            row[f"bbox_area_max_{label}"] = 0.0

        for det in detections:
            label = det.get("category") or det.get("label")
            if label not in LABELS:
                continue
            box = det["box"]
            width = max(0.0, float(box[2]) - float(box[0]))
            height = max(0.0, float(box[3]) - float(box[1]))
            area = width * height
            total_area += area
            row[f"bbox_count_{label}"] += 1.0
            row[f"bbox_area_sum_{label}"] += area
            row[f"bbox_area_max_{label}"] = max(row[f"bbox_area_max_{label}"], area)

        for label in LABELS:
            count = row[f"bbox_count_{label}"]
            if count > 0:
                row[f"bbox_area_mean_{label}"] = row[f"bbox_area_sum_{label}"] / count

        row["bbox_area_sum_total"] = total_area
        features[image] = row

    return features


def build_image_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in IMAGE_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            index[path.name] = path
    return index


def compute_brightness_metrics(path: Path) -> dict[str, float]:
    image = Image.open(path).convert("RGB")
    image.thumbnail((ANALYSIS_MAX_SIDE, ANALYSIS_MAX_SIDE))
    rgb = np.asarray(image, dtype=np.float32)
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    flat = gray.reshape(-1)
    trimmed = flat[(flat >= np.quantile(flat, 0.1)) & (flat <= np.quantile(flat, 0.9))]
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    value = rgb.max(axis=2)
    gray_std = float(gray.std())
    return {
        "gray_mean": float(gray.mean()),
        "gray_median": float(np.median(gray)),
        "gray_trimmed_mean": float(trimmed.mean()),
        "gray_p90": float(np.quantile(flat, 0.9)),
        "luma_mean": float(luma.mean()),
        "value_mean": float(value.mean()),
        "gray_mean_over_std": float(gray.mean() / (gray_std + EPS)),
        "gray_std": gray_std,
    }


def add_normalized_targets(rows: list[dict[str, float | str]]) -> None:
    gray_mean = np.array([float(row["gray_mean"]) for row in rows], dtype=float)
    mean = float(gray_mean.mean())
    std = float(gray_mean.std() + EPS)
    median = float(np.median(gray_mean))
    mad = float(np.median(np.abs(gray_mean - median)) + EPS)

    for row in rows:
        gm = float(row["gray_mean"])
        row["gray_mean_zscore"] = (gm - mean) / std
        row["gray_mean_robust_zscore"] = (gm - median) / (1.4826 * mad + EPS)
        row["bright_binary_median_split"] = 1 if gm >= median else 0
        row["bright_binary_mean_plus_half_std"] = 1 if gm >= mean + 0.5 * std else 0


def mean_std_scale_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < EPS] = 1.0
    return mean, scale


def mean_std_scale_transform(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - mean) / scale


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


def fit_ols(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    X1 = add_intercept(X)
    if sample_weight is None:
        return np.linalg.pinv(X1) @ y
    root_w = np.sqrt(sample_weight)
    Xw = X1 * root_w[:, None]
    yw = y * root_w
    return np.linalg.pinv(Xw) @ yw


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    X1 = add_intercept(X)
    if sample_weight is not None:
        root_w = np.sqrt(sample_weight)
        X1 = X1 * root_w[:, None]
        y = y * root_w
    penalty = np.eye(X1.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(X1.T @ X1 + penalty, X1.T @ y)


def fit_huber(
    X: np.ndarray,
    y: np.ndarray,
    delta: float = 1.5,
    max_iter: int = 50,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    X1 = add_intercept(X)
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=float)
    root_base = np.sqrt(sample_weight)
    beta = np.linalg.pinv(X1 * root_base[:, None]) @ (y * root_base)
    for _ in range(max_iter):
        residuals = y - X1 @ beta
        scale = np.median(np.abs(residuals)) / 0.6745
        scale = float(max(scale, EPS))
        scaled = residuals / scale
        weights = np.ones_like(scaled)
        mask = np.abs(scaled) > delta
        weights[mask] = delta / np.abs(scaled[mask])
        total_weight = weights * sample_weight
        root_w = np.sqrt(total_weight)
        Xw = X1 * root_w[:, None]
        yw = y * root_w
        beta_next = np.linalg.pinv(Xw) @ yw
        if np.max(np.abs(beta_next - beta)) < 1e-6:
            beta = beta_next
            break
        beta = beta_next
    return beta


def predict_linear(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return add_intercept(X) @ beta


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def fit_logistic_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    lr: float = 0.1,
    max_iter: int = 4000,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    X1 = add_intercept(X)
    beta = np.zeros(X1.shape[1], dtype=float)
    penalty = np.ones_like(beta) * alpha
    penalty[0] = 0.0
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=float)
    norm = float(np.sum(sample_weight))

    for _ in range(max_iter):
        probs = sigmoid(X1 @ beta)
        gradient = (X1.T @ ((probs - y) * sample_weight)) / norm + penalty * beta / norm
        beta_next = beta - lr * gradient
        if np.max(np.abs(beta_next - beta)) < 1e-7:
            beta = beta_next
            break
        beta = beta_next
    return beta


def predict_logistic_proba(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return sigmoid(add_intercept(X) @ beta)


def make_folds(n_samples: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    return [fold for fold in np.array_split(order, n_folds) if len(fold) > 0]


def make_stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = [[] for _ in range(n_folds)]
    for idx, value in enumerate(pos):
        folds[idx % n_folds].append(int(value))
    for idx, value in enumerate(neg):
        folds[idx % n_folds].append(int(value))
    return [np.array(sorted(fold), dtype=int) for fold in folds if fold]


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom < EPS:
        return 0.0
    return 1.0 - float(np.sum((y_true - y_pred) ** 2) / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ar = rankdata(a)
    br = rankdata(b)
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = math.sqrt(float((ar**2).sum() * (br**2).sum()))
    if denom < EPS:
        return 0.0
    return float((ar @ br) / denom)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    tpr = float(np.mean(y_pred[positives] == 1)) if positives.any() else 0.0
    tnr = float(np.mean(y_pred[negatives] == 0)) if negatives.any() else 0.0
    return 0.5 * (tpr + tnr)


def binary_log_loss(y_true: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)))


def roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    pos = probs[y_true == 1]
    neg = probs[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg))
        wins += 0.5 * float(np.sum(p == neg))
    return wins / (len(pos) * len(neg))


def cross_validated_regression(
    X: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray],
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    preds = np.zeros_like(y, dtype=float)
    folds = make_folds(len(y), N_FOLDS, RNG_SEED)
    all_idx = np.arange(len(y))

    for test_idx in folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_idx[train_mask]

        x_mean, x_scale = mean_std_scale_fit(X[train_idx])
        X_train = mean_std_scale_transform(X[train_idx], x_mean, x_scale)
        X_test = mean_std_scale_transform(X[test_idx], x_mean, x_scale)

        train_weight = None if sample_weight is None else sample_weight[train_idx]
        beta = fit_fn(X_train, y[train_idx], train_weight)
        preds[test_idx] = predict_linear(X_test, beta)

    return preds


def cross_validated_logistic(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    probs = np.zeros_like(y, dtype=float)
    folds = make_stratified_folds(y, N_FOLDS, RNG_SEED)
    all_idx = np.arange(len(y))

    for test_idx in folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_idx[train_mask]

        x_mean, x_scale = mean_std_scale_fit(X[train_idx])
        X_train = mean_std_scale_transform(X[train_idx], x_mean, x_scale)
        X_test = mean_std_scale_transform(X[test_idx], x_mean, x_scale)

        train_weight = None if sample_weight is None else sample_weight[train_idx]
        beta = fit_logistic_ridge(X_train, y[train_idx], sample_weight=train_weight)
        probs[test_idx] = predict_logistic_proba(X_test, beta)

    return probs


def as_feature_matrix(rows: list[dict[str, float | str]], feature_names: list[str]) -> np.ndarray:
    return np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=float)


def as_target_vector(rows: list[dict[str, float | str]], target_name: str) -> np.ndarray:
    return np.array([float(row[target_name]) for row in rows], dtype=float)


def regression_feature_sets() -> dict[str, list[str]]:
    count_features = [f"dino_count_{label}" for label in LABELS]
    bbox_area_features = [f"bbox_area_sum_{label}" for label in LABELS]
    bbox_max_features = [f"bbox_area_max_{label}" for label in LABELS]
    bbox_mean_features = [f"bbox_area_mean_{label}" for label in LABELS]
    bbox_count_features = [f"bbox_count_{label}" for label in LABELS]
    return {
        "counts_only": count_features,
        "counts_plus_box_area": count_features + bbox_area_features + bbox_max_features,
        "counts_plus_box_area_and_density": (
            count_features
            + bbox_count_features
            + bbox_area_features
            + bbox_mean_features
            + ["bbox_total_boxes", "bbox_area_sum_total"]
        ),
        "bbox_only": bbox_count_features + bbox_area_features + bbox_mean_features + ["bbox_total_boxes"],
    }


def make_streetlight_weightings(rows: list[dict[str, float | str]]) -> dict[str, np.ndarray | None]:
    counts = np.array([float(row["dino_count_streetlight"]) for row in rows], dtype=float)
    box_counts = np.array([float(row["bbox_count_streetlight"]) for row in rows], dtype=float)
    nonzero = counts > 0
    stronger = counts >= 2
    box_nonzero = box_counts > 0

    weightings: dict[str, np.ndarray | None] = {"unweighted": None}
    for factor in (2.0, 3.0, 5.0):
        w = np.ones(len(rows), dtype=float)
        w[nonzero] = factor
        weightings[f"streetlight_count_gt0_x{int(factor)}"] = w

    binned = np.ones(len(rows), dtype=float)
    binned[counts == 1] = 2.0
    binned[(counts >= 2) & (counts <= 3)] = 3.0
    binned[counts >= 4] = 5.0
    weightings["streetlight_count_binned"] = binned

    hybrid = np.ones(len(rows), dtype=float)
    hybrid[box_nonzero] = 2.0
    hybrid[stronger] = 4.0
    weightings["streetlight_hybrid_count_bbox"] = hybrid
    return weightings


def evaluate_regressions(rows: list[dict[str, float | str]]) -> tuple[list[RegressionResult], dict[str, np.ndarray]]:
    feature_sets = regression_feature_sets()
    weightings = make_streetlight_weightings(rows)
    targets = [
        "gray_mean",
        "gray_median",
        "gray_trimmed_mean",
        "gray_p90",
        "luma_mean",
        "value_mean",
        "gray_mean_over_std",
        "gray_mean_zscore",
        "gray_mean_robust_zscore",
    ]
    models: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray]] = {
        "ols": fit_ols,
        "ridge": lambda X, y, w: fit_ridge(X, y, alpha=1.0, sample_weight=w),
        "huber": lambda X, y, w: fit_huber(X, y, sample_weight=w),
    }

    results: list[RegressionResult] = []
    prediction_cache: dict[str, np.ndarray] = {}

    for target in targets:
        y = as_target_vector(rows, target)
        for feature_set_name, feature_names in feature_sets.items():
            X = as_feature_matrix(rows, feature_names)
            for weighting_name, sample_weight in weightings.items():
                for model_name, fit_fn in models.items():
                    preds = cross_validated_regression(X, y, fit_fn, sample_weight=sample_weight)
                    key = f"{target}::{feature_set_name}::{weighting_name}::{model_name}"
                    prediction_cache[key] = preds
                    results.append(
                        RegressionResult(
                            target=target,
                            feature_set=feature_set_name,
                            model=model_name,
                            weighting=weighting_name,
                            r2=r2_score(y, preds),
                            rmse=rmse(y, preds),
                            mae=mae(y, preds),
                            spearman=spearman_corr(y, preds),
                        )
                    )

    return results, prediction_cache


def evaluate_logistic(rows: list[dict[str, float | str]]) -> tuple[list[LogisticResult], dict[str, np.ndarray]]:
    feature_sets = regression_feature_sets()
    weightings = make_streetlight_weightings(rows)
    targets = ["bright_binary_median_split", "bright_binary_mean_plus_half_std"]

    results: list[LogisticResult] = []
    probability_cache: dict[str, np.ndarray] = {}

    for target in targets:
        y = as_target_vector(rows, target).astype(int)
        for feature_set_name, feature_names in feature_sets.items():
            X = as_feature_matrix(rows, feature_names)
            for weighting_name, sample_weight in weightings.items():
                probs = cross_validated_logistic(X, y, sample_weight=sample_weight)
                preds = (probs >= 0.5).astype(int)
                key = f"{target}::{feature_set_name}::{weighting_name}::logistic_ridge"
                probability_cache[key] = probs
                results.append(
                    LogisticResult(
                        target=target,
                        feature_set=feature_set_name,
                        model="logistic_ridge",
                        weighting=weighting_name,
                        accuracy=accuracy(y, preds),
                        balanced_accuracy=balanced_accuracy(y, preds),
                        auc=roc_auc(y, probs),
                        log_loss=binary_log_loss(y, probs),
                    )
                )

    return results, probability_cache


def format_float(value: float) -> str:
    return f"{value:.4f}"


def save_regression_plot(
    rows: list[dict[str, float | str]],
    best_result: RegressionResult,
    prediction_cache: dict[str, np.ndarray],
) -> None:
    key = f"{best_result.target}::{best_result.feature_set}::{best_result.weighting}::{best_result.model}"
    y = as_target_vector(rows, best_result.target)
    preds = prediction_cache[key]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, preds, alpha=0.65, edgecolors="k", linewidths=0.3)
    mn = min(float(y.min()), float(preds.min()))
    mx = max(float(y.max()), float(preds.max()))
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.0, color="red")
    ax.set_xlabel(f"Actual {best_result.target}")
    ax.set_ylabel("Predicted")
    ax.set_title(
        f"Best regression: {best_result.model} / {best_result.feature_set}\n"
        f"{best_result.target}  R^2={best_result.r2:.3f}"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "best_regression_scatter.png", dpi=150)
    plt.close(fig)


def build_dataset_rows() -> list[dict[str, float | str]]:
    matches = load_matches()
    counts = load_counts()
    bbox = load_bbox_features()
    image_index = build_image_index()

    rows: list[dict[str, float | str]] = []
    for match in matches:
        night_photo = match["night_photo"]
        day_image = match["day_image"]
        night_path = image_index[night_photo]
        brightness = compute_brightness_metrics(night_path)

        row: dict[str, float | str] = {
            "night_photo": night_photo,
            "night_image_path": str(night_path),
            "day_image": day_image,
            "distance_m": float(match["distance_m"]),
        }
        row.update(brightness)
        row.update(counts[day_image])
        row.update(bbox[day_image])
        rows.append(row)

    add_normalized_targets(rows)
    return rows


def save_outputs(
    dataset_rows: list[dict[str, float | str]],
    regression_results: list[RegressionResult],
    logistic_results: list[LogisticResult],
    prediction_cache: dict[str, np.ndarray],
    probability_cache: dict[str, np.ndarray],
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    dataset_fieldnames = list(dataset_rows[0].keys())
    write_csv(OUTPUT_DIR / "paired_dataset_with_brightness.csv", dataset_fieldnames, dataset_rows)

    regression_rows = [
        {
            "target": result.target,
            "feature_set": result.feature_set,
            "weighting": result.weighting,
            "model": result.model,
            "r2": format_float(result.r2),
            "rmse": format_float(result.rmse),
            "mae": format_float(result.mae),
            "spearman": format_float(result.spearman),
        }
        for result in sorted(regression_results, key=lambda item: (-item.r2, item.rmse))
    ]
    write_csv(
        OUTPUT_DIR / "regression_summary.csv",
        ["target", "feature_set", "weighting", "model", "r2", "rmse", "mae", "spearman"],
        regression_rows,
    )

    logistic_rows = [
        {
            "target": result.target,
            "feature_set": result.feature_set,
            "weighting": result.weighting,
            "model": result.model,
            "accuracy": format_float(result.accuracy),
            "balanced_accuracy": format_float(result.balanced_accuracy),
            "auc": format_float(result.auc),
            "log_loss": format_float(result.log_loss),
        }
        for result in sorted(logistic_results, key=lambda item: (-item.auc, -item.balanced_accuracy))
    ]
    write_csv(
        OUTPUT_DIR / "logistic_summary.csv",
        ["target", "feature_set", "weighting", "model", "accuracy", "balanced_accuracy", "auc", "log_loss"],
        logistic_rows,
    )

    best_regression = max(regression_results, key=lambda item: item.r2)
    best_regression_key = f"{best_regression.target}::{best_regression.feature_set}::{best_regression.weighting}::{best_regression.model}"
    best_regression_preds = prediction_cache[best_regression_key]
    best_prediction_rows = []
    for row, pred in zip(dataset_rows, best_regression_preds):
        best_prediction_rows.append(
            {
                "night_photo": row["night_photo"],
                "day_image": row["day_image"],
                "actual_target": format_float(float(row[best_regression.target])),
                "predicted_target": format_float(float(pred)),
                "target_name": best_regression.target,
                "feature_set": best_regression.feature_set,
                "weighting": best_regression.weighting,
                "model": best_regression.model,
            }
        )
    write_csv(
        OUTPUT_DIR / "best_regression_predictions.csv",
        ["night_photo", "day_image", "actual_target", "predicted_target", "target_name", "feature_set", "weighting", "model"],
        best_prediction_rows,
    )

    best_logistic = max(logistic_results, key=lambda item: item.auc)
    best_logistic_key = f"{best_logistic.target}::{best_logistic.feature_set}::{best_logistic.weighting}::{best_logistic.model}"
    best_logistic_probs = probability_cache[best_logistic_key]
    best_logistic_rows = []
    for row, prob in zip(dataset_rows, best_logistic_probs):
        actual = int(float(row[best_logistic.target]))
        best_logistic_rows.append(
            {
                "night_photo": row["night_photo"],
                "day_image": row["day_image"],
                "actual_class": actual,
                "predicted_probability": format_float(float(prob)),
                "predicted_class": int(prob >= 0.5),
                "target_name": best_logistic.target,
                "feature_set": best_logistic.feature_set,
                "weighting": best_logistic.weighting,
                "model": best_logistic.model,
            }
        )
    write_csv(
        OUTPUT_DIR / "best_logistic_predictions.csv",
        [
            "night_photo",
            "day_image",
            "actual_class",
            "predicted_probability",
            "predicted_class",
            "target_name",
            "feature_set",
            "weighting",
            "model",
        ],
        best_logistic_rows,
    )

    save_regression_plot(dataset_rows, best_regression, prediction_cache)

    top_regressions = sorted(regression_results, key=lambda item: (-item.r2, item.rmse))[:10]
    top_logistics = sorted(logistic_results, key=lambda item: (-item.auc, -item.balanced_accuracy))[:6]

    gray_mean_results = [item for item in regression_results if item.target == "gray_mean"]
    gray_mean_counts_only = max(
        (item for item in gray_mean_results if item.feature_set == "counts_only" and item.weighting == "unweighted"),
        key=lambda item: item.r2,
    )
    gray_mean_box_augmented = max(
        (item for item in gray_mean_results if item.feature_set == "counts_plus_box_area_and_density" and item.weighting == "unweighted"),
        key=lambda item: item.r2,
    )
    delta_r2 = gray_mean_box_augmented.r2 - gray_mean_counts_only.r2
    delta_rmse = gray_mean_box_augmented.rmse - gray_mean_counts_only.rmse
    weighted_gray_mean = max(
        (item for item in gray_mean_results if item.feature_set == "counts_plus_box_area_and_density" and item.weighting != "unweighted"),
        key=lambda item: item.r2,
    )
    weighted_delta_r2 = weighted_gray_mean.r2 - gray_mean_box_augmented.r2
    weighted_delta_rmse = weighted_gray_mean.rmse - gray_mean_box_augmented.rmse

    report_lines = [
        "# Brightness Metric Experiment Report",
        "",
        f"- Dataset rows used: {len(dataset_rows)}",
        f"- Unique matched night/day pairs after night-photo deduplication: {len(dataset_rows)}",
        f"- Cross-validation: {N_FOLDS}-fold with fixed seed {RNG_SEED}",
        f"- Feature families: counts-only, bbox-only, and bbox-size-augmented variants",
        "",
        "## Quick Takeaways",
        "",
        f"- Best regression overall: `{best_regression.model}` on `{best_regression.target}` with `{best_regression.feature_set}` / `{best_regression.weighting}` "
        f"(R^2={best_regression.r2:.4f}, RMSE={best_regression.rmse:.4f}).",
        f"- Best logistic setup: `{best_logistic.model}` on `{best_logistic.target}` with `{best_logistic.feature_set}` / `{best_logistic.weighting}` "
        f"(AUC={best_logistic.auc:.4f}, balanced accuracy={best_logistic.balanced_accuracy:.4f}).",
        f"- For raw `gray_mean`, adding bbox-size features changed best-model R^2 by {delta_r2:+.4f} "
        f"and RMSE by {delta_rmse:+.4f} versus counts-only.",
        f"- Best lamppost-aware weighting on raw `gray_mean` changed R^2 by {weighted_delta_r2:+.4f} "
        f"and RMSE by {weighted_delta_rmse:+.4f} versus the unweighted bbox-augmented baseline.",
        "- Z-score targets are useful for standardizing interpretation, but because they are an affine rescaling of brightness, "
        "their model ranking should stay almost the same as the underlying raw metric.",
        "",
        "## Top Regression Results",
        "",
        "| target | feature_set | weighting | model | R^2 | RMSE | MAE | Spearman |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in top_regressions:
        report_lines.append(
            f"| {item.target} | {item.feature_set} | {item.weighting} | {item.model} | "
            f"{item.r2:.4f} | {item.rmse:.4f} | {item.mae:.4f} | {item.spearman:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Top Logistic Results",
            "",
            "| target | feature_set | weighting | model | accuracy | balanced_accuracy | AUC | log_loss |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in top_logistics:
        report_lines.append(
            f"| {item.target} | {item.feature_set} | {item.weighting} | {item.model} | "
            f"{item.accuracy:.4f} | {item.balanced_accuracy:.4f} | {item.auc:.4f} | {item.log_loss:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Lamppost-aware weighting is applied only inside training folds, so the evaluation stays out-of-sample.",
            "- Bounding-box size is represented with raw detector-space box areas, since the original day-image dimensions are not stored in the JSON.",
            "- `gray_mean_over_std` is an exploratory local normalization that bakes each image's own contrast scale into the target.",
            "- `bright_binary_mean_plus_half_std` is intentionally harder than the median split because it isolates the brighter tail.",
            "",
            "## Output Files",
            "",
            "- `paired_dataset_with_brightness.csv`",
            "- `regression_summary.csv`",
            "- `logistic_summary.csv`",
            "- `best_regression_predictions.csv`",
            "- `best_logistic_predictions.csv`",
            "- `best_regression_scatter.png`",
        ]
    )
    (OUTPUT_DIR / "experiment_report.md").write_text("\n".join(report_lines))


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    dataset_rows = build_dataset_rows()
    regression_results, prediction_cache = evaluate_regressions(dataset_rows)
    logistic_results, probability_cache = evaluate_logistic(dataset_rows)
    save_outputs(dataset_rows, regression_results, logistic_results, prediction_cache, probability_cache)
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
