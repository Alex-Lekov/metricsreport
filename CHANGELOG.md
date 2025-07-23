# Changelog

## 2025.7.23

### Added

-   Significantly increased test coverage for the `metricsreport` module, ensuring all functions and edge cases are properly tested.

### Fixed

-   Resolved a potential `ZeroDivisionError` in the `plot_all_count_metrics` function that could occur with a small number of steps.
-   Addressed a `RuntimeWarning` related to memory leaks by ensuring Matplotlib figures are properly closed after use in the HTML report generation.
-   Suppressed the `UndefinedMetricWarning` by setting the `zero_division` parameter to `0` in `classification_report` calls, resulting in cleaner test outputs.
-   Corrected a failing test case with an incorrect assertion.

### Changed

-   Updated project version to `2025.7.23`.

## 2024.7.22

### Fixed

- Fixed UndefinedMetricWarning and RuntimeWarning that occurred during metric calculation and plotting in cases where the model does not predict any positive examples (zero division).
- Suppressed the UndefinedMetricWarning in the `test_y_pred_bin_all_zeros` test in `tests/test_metricsreport.py` for clean test runs.