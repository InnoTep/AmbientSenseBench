import pandas as pd
import pytest

# matplotlib is an install extra, so skip rather than fail collection when a base
# install is used.
pytest.importorskip("matplotlib", reason="install .[dev] to run the figure tests")

from ambientsensebench.plot_benchmark import create_benchmark_figures


def test_figure_generator_writes_performance_and_alarm_figures(tmp_path):
    summary = pd.DataFrame(
        {
            "scenario": ["P01", "P01", "P04", "P04"],
            "model_id": ["isolation_forest", "one_class_svm"] * 2,
            "model": ["Isolation Forest", "One-Class SVM"] * 2,
            "auprc_mean": [0.91, 0.97, 0.97, 0.99],
            "auprc_std": [0.01, 0.01, 0.01, 0.01],
            "f1_fixed_threshold_mean": [0.65, 0.25, 0.93, 0.90],
            "f1_fixed_threshold_std": [0.02, 0.02, 0.01, 0.01],
            "n_warnings_total_mean": [64, 203, 290, 316],
            "n_warnings_total_std": [1, 2, 4, 0],
            "n_true_alarms_mean": [31, 31, 256, 259],
            "n_true_alarms_std": [1, 0, 1, 0],
            "n_false_alarms_mean": [33, 172, 34, 57],
            "n_false_alarms_std": [2, 2, 3, 0],
            "mean_lead_time_days_mean": [1, 14, 13, 14],
            "mean_lead_time_days_std": [1, 0, 1, 0],
            "runs": [5, 5, 5, 5],
        }
    )
    summary_path = tmp_path / "summary_results.csv"
    summary.to_csv(summary_path, index=False)

    figures = create_benchmark_figures(summary_path, tmp_path / "figures")

    assert figures["performance"].exists()
    assert figures["performance"].with_suffix(".png").exists()
    assert figures["alarm_burden"].exists()
    assert figures["alarm_burden"].with_suffix(".png").exists()
