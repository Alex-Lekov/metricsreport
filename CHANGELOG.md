# Changelog

## 2024.7.22

### Fixed

- Устранены UndefinedMetricWarning и RuntimeWarning, возникающие при расчете метрик и построении графиков в случаях, когда модель не предсказывает ни одного положительного примера (нулевое деление).
- Подавлено предупреждение UndefinedMetricWarning в тесте test_y_pred_bin_all_zeros в tests/test_metricsreport.py для чистого прохождения тестов.