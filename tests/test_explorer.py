import numpy as np
import pandas as pd

from ambientsensebench.explorer import evaluate_isolation_forest
from ambientsensebench.wp6_evaluation import FEATURE_COLUMNS


def test_explorer_scoring_marks_only_test_window_as_eligible_for_alerts():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(365, len(FEATURE_COLUMNS)))
    features = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    features["date"] = pd.date_range("2026-01-01", periods=365).strftime("%Y-%m-%d")
    features["label"] = ["baseline"] * 80 + ["distress"] * 285
    features["severity"] = [0.0] * 80 + [1.0] * 285

    scored, metrics = evaluate_isolation_forest(features)

    assert scored["anomaly_score"].iloc[:40].isna().all()
    assert not scored["alarm"].iloc[:40].any()
    assert len(scored) == 365
    assert metrics["training_days"] == 40
