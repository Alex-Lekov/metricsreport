import pytest
import numpy as np
from matplotlib import pyplot as plt

from metricsreport import MetricsReport

@pytest.fixture
def binary_classification_data():
    y_true = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    y_pred = [0.2, 0.8, 0.9, 0.1, 0.6, 0.3, 0.4, 0.7, 0.2, 0.9, 0.8, 0.4, 0.9]
    return y_true, y_pred

@pytest.fixture
def regression_data():
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1.1, 2.3, 3.4, 4.2, 4.8]
    return y_true, y_pred

########### tests ############

def test_metrics_report_instance(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    assert report.task_type == "classification"
    assert np.array_equal(report.y_true, np.array(y_true))
    assert np.array_equal(report.y_pred, np.array(y_pred))
    assert report.threshold == 0.5

def test_determine_task_type(binary_classification_data, regression_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred)
    assert report._determine_task_type(y_true) == "classification"
    
    y_true, y_pred = regression_data
    report = MetricsReport(y_true, y_pred)
    assert report._determine_task_type(y_true) == "regression"

def test_classification_metrics(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    assert isinstance(report.metrics, dict)
    assert set(report.metrics.keys()) == {
        'AP', 'AUC', 'Log Loss', 'MSE', 'Accuracy', 'Precision_weighted', 'MCC', 'TN', 'FP', 'FN', 'TP', 
        'P precision', 'P recall', 'P f1-score', 'P support', 'N precision', 'N recall', 'N f1-score', 'N support'
    }

def test_regression_metrics(regression_data):
    y_true, y_pred = regression_data
    report = MetricsReport(y_true, y_pred)
    assert isinstance(report.metrics, dict)
    assert set(report.metrics.keys()) == {
        'Mean Squared Error', 'Mean Squared Log Error', 'Mean Absolute Error', 'R^2', 
        'Explained Variance Score', 'Max Error', 'Mean Absolute Percentage Error'
    }

def test_classification_metrics_values(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    assert np.isclose(report.metrics['AUC'], 0.7857, atol=1e-4)
    assert np.isclose(report.metrics['Log Loss'], 0.5807, atol=1e-4)
    assert np.isclose(report.metrics['AP'], 0.6635, atol=1e-4)
    assert np.isclose(report.metrics['Accuracy'], 0.7692, atol=1e-4)
    assert np.isclose(report.metrics['Precision_weighted'], 0.7784, atol=1e-4)
    assert report.metrics['TN'] == 5
    assert report.metrics['FP'] == 2
    assert report.metrics['FN'] == 1
    assert report.metrics['TP'] == 5

def test_y_pred_bin_all_zeros():
    y_true = [0, 1, 1, 0]
    y_pred = [0.2, 0.3, 0.3, 0.1]
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    assert report.metrics['TP'] == 0
    assert report.metrics['FP'] == 0
    assert report.metrics['FN'] == 2
    assert report.metrics['TN'] == 2
    
def test_y_true_all_zeros():
    y_true = [0, 0, 0, 0]
    y_pred = [0.2, 0.3, 0.3, 0.1]
    with pytest.raises(ValueError) as exc_info:
        MetricsReport(y_true, y_pred, threshold=0.5)
    assert str(exc_info.value) == "For classification tasks, y_true should contain at least one True value."

########### Plot Tests ############

def test_plot_roc_curve(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_roc_curve()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_precision_recall_curve(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_precision_recall_curve()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_confusion_matrix(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_confusion_matrix()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_class_distribution(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_class_distribution()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_class_hist(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_class_hist()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_all_count_metrics(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_all_count_metrics()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_calibration_curve(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_calibration_curve()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_lift_curve(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_lift_curve()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_ks_statistic(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_ks_statistic()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_precision_recall_vs_threshold(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_precision_recall_vs_threshold()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_tp_fp_with_optimal_threshold(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_tp_fp_with_optimal_threshold()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_cumulative_gains_chart(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_cumulative_gains_chart()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_cap(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_cap()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_residual_plot(regression_data):
    y_true, y_pred = regression_data
    report = MetricsReport(y_true, y_pred)
    plt_obj = report.plot_residual_plot()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_predicted_vs_actual(regression_data):
    y_true, y_pred = regression_data
    report = MetricsReport(y_true, y_pred)
    plt_obj = report.plot_predicted_vs_actual()
    assert isinstance(plt_obj.gcf(), plt.Figure)
