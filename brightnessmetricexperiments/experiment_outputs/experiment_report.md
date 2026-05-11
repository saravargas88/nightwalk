# Brightness Metric Experiment Report

- Dataset rows used: 974
- Unique matched night/day pairs after night-photo deduplication: 974
- Cross-validation: 5-fold with fixed seed 42
- Feature families: counts-only, bbox-only, and bbox-size-augmented variants

## Quick Takeaways

- Best regression overall: `ols` on `value_mean` with `counts_plus_box_area_and_density` (R^2=0.1411, RMSE=21.6714).
- Best logistic setup: `logistic_ridge` on `bright_binary_mean_plus_half_std` with `counts_plus_box_area` (AUC=0.6879, balanced accuracy=0.5285).
- For raw `gray_mean`, adding bbox-size features changed best-model R^2 by +0.0414 and RMSE by -0.4709 versus counts-only.
- Z-score targets are useful for standardizing interpretation, but because they are an affine rescaling of brightness, their model ranking should stay almost the same as the underlying raw metric.

## Top Regression Results

| target | feature_set | model | R^2 | RMSE | MAE | Spearman |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| value_mean | counts_plus_box_area_and_density | ols | 0.1411 | 21.6714 | 17.7399 | 0.3659 |
| value_mean | bbox_only | ridge | 0.1407 | 21.6755 | 17.7257 | 0.3627 |
| value_mean | counts_plus_box_area_and_density | ridge | 0.1407 | 21.6758 | 17.7267 | 0.3628 |
| value_mean | bbox_only | ols | 0.1404 | 21.6799 | 17.7309 | 0.3617 |
| value_mean | counts_plus_box_area_and_density | huber | 0.1389 | 21.6987 | 17.6829 | 0.3650 |
| value_mean | bbox_only | huber | 0.1389 | 21.6987 | 17.6829 | 0.3650 |
| value_mean | counts_plus_box_area | ridge | 0.1368 | 21.7245 | 17.6720 | 0.3583 |
| value_mean | counts_plus_box_area | ols | 0.1367 | 21.7267 | 17.6738 | 0.3580 |
| gray_mean_zscore | counts_plus_box_area_and_density | ols | 0.1366 | 0.9292 | 0.7579 | 0.3555 |
| gray_mean_robust_zscore | counts_plus_box_area_and_density | ols | 0.1365 | 0.8321 | 0.6786 | 0.3557 |

## Top Logistic Results

| target | feature_set | model | accuracy | balanced_accuracy | AUC | log_loss |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| bright_binary_mean_plus_half_std | counts_plus_box_area | logistic_ridge | 0.6725 | 0.5285 | 0.6879 | 0.5720 |
| bright_binary_mean_plus_half_std | counts_plus_box_area_and_density | logistic_ridge | 0.6581 | 0.5179 | 0.6841 | 0.5734 |
| bright_binary_median_split | counts_plus_box_area | logistic_ridge | 0.6345 | 0.6345 | 0.6840 | 0.6396 |
| bright_binary_mean_plus_half_std | bbox_only | logistic_ridge | 0.6591 | 0.5195 | 0.6839 | 0.5734 |
| bright_binary_median_split | bbox_only | logistic_ridge | 0.6427 | 0.6427 | 0.6838 | 0.6406 |
| bright_binary_median_split | counts_plus_box_area_and_density | logistic_ridge | 0.6427 | 0.6427 | 0.6837 | 0.6407 |

## Notes

- Bounding-box size is represented with raw detector-space box areas, since the original day-image dimensions are not stored in the JSON.
- `gray_mean_over_std` is an exploratory local normalization that bakes each image's own contrast scale into the target.
- `bright_binary_mean_plus_half_std` is intentionally harder than the median split because it isolates the brighter tail.

## Output Files

- `paired_dataset_with_brightness.csv`
- `regression_summary.csv`
- `logistic_summary.csv`
- `best_regression_predictions.csv`
- `best_logistic_predictions.csv`
- `best_regression_scatter.png`