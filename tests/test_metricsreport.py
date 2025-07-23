import pytest
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
import pandas as pd

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

@pytest.mark.filterwarnings("ignore:Precision is ill-defined")
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

def test_plot_roc_curve_invalid_curves(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    with pytest.raises(ValueError) as exc_info:
        report.plot_roc_curve(curves=('invalid',))
    assert str(exc_info.value) == 'curves must contain "micro" or "macro"'

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

def test_plot_precision_recall_vs_threshold_invalid_input(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    report.y_true = np.array([[1,2],[3,4]])
    with pytest.raises(ValueError) as exc_info:
        report.plot_precision_recall_vs_threshold()
    assert str(exc_info.value) == "y_true and probas_pred must have the same length."

def test_plot_tp_fp_with_optimal_threshold(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    plt_obj = report.plot_tp_fp_with_optimal_threshold()
    assert isinstance(plt_obj.gcf(), plt.Figure)

def test_plot_tp_fp_with_optimal_threshold_invalid_input(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    report.y_pred = np.array([1,2])
    with pytest.raises(ValueError) as exc_info:
        report.plot_tp_fp_with_optimal_threshold()
    assert str(exc_info.value) == "y_true and probas_pred must have the same length."

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

def test_target_info_classification(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred)
    report._MetricsReport__target_info()
    assert report.target_info == {
        'Count of samples': 13,
        'Count True class': 6,
        'Count False class': 7,
        'Class balance %': 46.2,
    }

def test_target_info_regression(regression_data):
    y_true, y_pred = regression_data
    report = MetricsReport(y_true, y_pred)
    report._MetricsReport__target_info()
    assert report.target_info == {
        'Count of samples': 5,
        'Mean of target': 3.0,
        'Std of target': 1.41,
        'Min of target': 1,
        'Max of target': 5,
    }

def test_generate_html_report(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred)
    report._MetricsReport__target_info()
    html = report._generate_html_report()
    assert "Metrics Report" in html
    assert "Summary" in html
    assert "Data Information" in html
    assert "Metrics" in html
    assert "Plots" in html

def test_generate_html_report_no_css(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred)
    report._MetricsReport__target_info()
    html = report._generate_html_report(add_css=False)
    assert "<style>" not in html

def test_add_webp_plots_to_html_rows_classification(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred)
    rows = report.add_webp_plots_to_html_rows()
    assert 'data:image/webp;base64,' in rows
    assert 'all_count_metrics' in rows

def test_add_webp_plots_to_html_rows_regression(regression_data):
    y_true, y_pred = regression_data
    report = MetricsReport(y_true, y_pred)
    rows = report.add_webp_plots_to_html_rows()
    assert 'data:image/webp;base64,' in rows
    assert 'residual_plot' in rows

def test_generate_html_rows():
    report = MetricsReport([0,1],[0.1,0.9])
    data = {'int': 1, 'float': 0.5}
    rows = report._MetricsReport__generate_html_rows(data)
    assert '<tr><td>int</td><td>1</td></tr>' in rows
    assert '<tr><td>float</td><td>0.5</td></tr>' in rows

def test_save_report(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred)
    report.save_report(folder='test_report', name='test_report')
    assert os.path.exists('test_report/test_report.html')
    shutil.rmtree('test_report')

def test_print_metrics_classification(binary_classification_data, capsys):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred)
    report.print_metrics()
    captured = capsys.readouterr()
    assert "AP" in captured.out
    assert "AUC" in captured.out

def test_print_metrics_regression(regression_data, capsys):
    y_true, y_pred = regression_data
    report = MetricsReport(y_true, y_pred)
    report.print_metrics()
    captured = capsys.readouterr()
    assert "Mean Squared Error" in captured.out
    assert "R^2" in captured.out

def test_classification_plots(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred)
    report._classification_plots(save=True, folder='test_plots')
    assert os.path.exists('test_plots/plots/all_count_metrics.png')
    shutil.rmtree('test_plots')

def test_regression_plots(regression_data):
    y_true, y_pred = regression_data
    report = MetricsReport(y_true, y_pred)
    report._regression_plots(save=True, folder='test_plots')
    assert os.path.exists('test_plots/plots/residual_plot.png')
    shutil.rmtree('test_plots')

def test_print_report_classification(binary_classification_data, capsys):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred)
    report.print_report()
    captured = capsys.readouterr()
    assert "Classification Report" in captured.out
    assert "Metrics Report" in captured.out
    assert "Lift" in captured.out
    assert "Plots" in captured.out

def test_print_report_regression(regression_data, capsys):
    y_true, y_pred = regression_data
    report = MetricsReport(y_true, y_pred)
    report.print_report()
    captured = capsys.readouterr()
    assert "Metrics Report" in captured.out
    assert "Plots" in captured.out

def test_plot_metrics_classification(binary_classification_data):
    y_true, y_pred = binary_classification_data
    report = MetricsReport(y_true, y_pred)
    report.plot_metrics()

def test_plot_metrics_regression(regression_data):
    y_true, y_pred = regression_data
    report = MetricsReport(y_true, y_pred)
    report.plot_metrics()

@pytest.mark.filterwarnings("ignore:For classification tasks threshold")
def test_y_pred_binary_all_zeros():
    y_true = [0, 1, 1, 0]
    y_pred = [0.1, 0.2, 0.3, 0.4]
    report = MetricsReport(y_true, y_pred, threshold=0.5)
    assert report.metrics['TP'] == 0
    assert report.metrics['FP'] == 0
    assert report.metrics['FN'] == 2
    assert report.metrics['TN'] == 2
