# Brightness Metric Experiment Report

- Dataset rows used: 974
- Unique matched night/day pairs after night-photo deduplication: 974
- Cross-validation: 5-fold with fixed seed 42
- Feature families: counts-only, bbox-only, and bbox-size-augmented variants

## Quick Takeaways

- Best regression overall: `ols` on `value_mean` with `bbox_only` / `unweighted` (R^2=0.1426, RMSE=21.6423).
- Best logistic setup: `logistic_ridge` on `bright_binary_mean_plus_half_std` with `counts_plus_box_area` / `streetlight_hybrid_count_bbox` (AUC=0.6940, balanced accuracy=0.5806).
- For raw `gray_mean`, adding bbox-size features changed best-model R^2 by +0.0453 and RMSE by -0.5147 versus counts-only.
- Best lamppost-aware weighting on raw `gray_mean` changed R^2 by -0.0004 and RMSE by +0.0043 versus the unweighted bbox-augmented baseline.
- Z-score targets are useful for standardizing interpretation, but because they are an affine rescaling of brightness, their model ranking should stay almost the same as the underlying raw metric.

## Top Regression Results

| target | feature_set | weighting | model | R^2 | RMSE | MAE | Spearman |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| value_mean | bbox_only | unweighted | ols | 0.1426 | 21.6423 | 17.7090 | 0.3679 |
| value_mean | counts_plus_box_area_and_density | unweighted | ols | 0.1426 | 21.6423 | 17.7090 | 0.3679 |
| value_mean | counts_plus_box_area_and_density | unweighted | ridge | 0.1425 | 21.6437 | 17.6929 | 0.3674 |
| value_mean | bbox_only | unweighted | ridge | 0.1424 | 21.6443 | 17.6925 | 0.3674 |
| value_mean | counts_plus_box_area_and_density | streetlight_count_gt0_x2 | ridge | 0.1422 | 21.6467 | 17.6566 | 0.3676 |
| value_mean | bbox_only | streetlight_count_gt0_x2 | ridge | 0.1422 | 21.6473 | 17.6556 | 0.3676 |
| value_mean | bbox_only | streetlight_count_gt0_x2 | ols | 0.1421 | 21.6479 | 17.6747 | 0.3679 |
| value_mean | counts_plus_box_area_and_density | streetlight_count_gt0_x2 | ols | 0.1421 | 21.6479 | 17.6747 | 0.3679 |
| value_mean | counts_plus_box_area_and_density | streetlight_count_gt0_x2 | huber | 0.1421 | 21.6479 | 17.6187 | 0.3704 |
| value_mean | bbox_only | streetlight_count_gt0_x2 | huber | 0.1421 | 21.6479 | 17.6187 | 0.3704 |

## Top Logistic Results

| target | feature_set | weighting | model | accuracy | balanced_accuracy | AUC | log_loss |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| bright_binary_mean_plus_half_std | counts_plus_box_area | streetlight_hybrid_count_bbox | logistic_ridge | 0.6745 | 0.5806 | 0.6940 | 0.5737 |
| bright_binary_mean_plus_half_std | counts_plus_box_area | streetlight_count_gt0_x5 | logistic_ridge | 0.6684 | 0.6030 | 0.6932 | 0.5784 |
| bright_binary_mean_plus_half_std | counts_plus_box_area | streetlight_count_gt0_x3 | logistic_ridge | 0.6766 | 0.5812 | 0.6924 | 0.5734 |
| bright_binary_mean_plus_half_std | bbox_only | streetlight_hybrid_count_bbox | logistic_ridge | 0.6663 | 0.5677 | 0.6921 | 0.5738 |
| bright_binary_mean_plus_half_std | counts_plus_box_area_and_density | streetlight_hybrid_count_bbox | logistic_ridge | 0.6663 | 0.5677 | 0.6917 | 0.5739 |
| bright_binary_mean_plus_half_std | counts_plus_box_area | streetlight_count_binned | logistic_ridge | 0.6797 | 0.5766 | 0.6916 | 0.5733 |

## Notes

- Lamppost-aware weighting is applied only inside training folds, so the evaluation stays out-of-sample.
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