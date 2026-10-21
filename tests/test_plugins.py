import numpy as np
import pandas as pd
import pytest

from ambientsensebench import benchmark
from ambientsensebench.wp6_evaluation import FEATURE_COLUMNS


class MeanAbsDeviation:
    """Toy plugin: mean absolute standardised feature value of each day."""

    def fit_score(self, train_data, test_data):
        return np.abs(train_data).mean(axis=1), np.abs(test_data).mean(axis=1)


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(benchmark, "PLUGIN_DETECTORS", {})
    return benchmark


def _scenario_frame():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(365, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame["date"] = pd.date_range("2026-01-01", periods=365).strftime("%Y-%m-%d")
    frame["label"] = "baseline"
    frame.loc[100:160, "label"] = "distress"
    frame.loc[100:160, FEATURE_COLUMNS] += 3.0
    frame["severity"] = (frame["label"] == "distress").astype(float)
    return frame


def test_registered_plugin_runs_under_the_shared_protocol(registry):
    registry.register_detector("mean_abs", "Mean |z|", lambda training_rows: MeanAbsDeviation())

    result = registry.evaluate_detector(_scenario_frame(), "P01", "mean_abs", seed=0)

    assert "mean_abs" in registry.available_models()
    assert result["model"] == "Mean |z|"
    assert result["n_train"] == 40
    assert result["auprc"] > 0.9


def test_plugin_ids_cannot_shadow_existing_detectors(registry):
    for taken in ("bocpd", "ewma"):
        with pytest.raises(ValueError):
            registry.register_detector(taken, "Duplicate", lambda training_rows: MeanAbsDeviation())
