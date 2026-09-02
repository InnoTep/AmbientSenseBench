"""Data preparation for the local AmbientSenseBench explorer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .generate_scenarios import SCENARIOS, generate_scenario
from .wp6_evaluation import (
    FEATURE_COLUMNS,
    compute_fixed_threshold_from_training_scores,
    convert_wake_time_to_hour,
    prepare_train_test_arrays,
    split_train_test,
)

SCENARIO_DESCRIPTIONS = {
    "P01": "Short disruption followed by recovery",
    "P02": "Sustained disruption with late recovery",
    "P03": "Two recurrent disruptions separated by recovery",
    "P04": "Progressive, monotone change in routine",
}


def evaluate_isolation_forest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Attach baseline-trained Isolation Forest scores and fixed-threshold alarms."""
    df = df.copy()
    df["wake_time"] = df["wake_time"].apply(convert_wake_time_to_hour)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    train_df, test_df = split_train_test(df, "explorer")
    train_data, test_data, _, _ = prepare_train_test_arrays(train_df, test_df)

    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
    )
    model.fit(train_data)

    train_scores = -model.decision_function(train_data)
    test_scores = -model.decision_function(test_data)
    threshold = compute_fixed_threshold_from_training_scores(train_scores)
    test_alerts = test_scores >= threshold

    scored = df.copy()
    scored["anomaly_score"] = np.nan
    scored["alarm"] = False
    scored.loc[test_df.index, "anomaly_score"] = test_scores
    scored.loc[test_df.index, "alarm"] = test_alerts

    labels = test_df["label"].to_numpy()
    alert_labels = labels[test_alerts]
    return scored, {
        "threshold": round(float(threshold), 5),
        "alerts": int(test_alerts.sum()),
        "alerts_on_change_days": int((alert_labels == "distress").sum()),
        "alerts_on_baseline_days": int((alert_labels == "baseline").sum()),
        "training_days": len(train_df),
    }


def _normalise_record(record: dict[str, object]) -> dict[str, object]:
    normalised: dict[str, object] = {}
    for key, value in record.items():
        if isinstance(value, (bool, np.bool_)):
            normalised[key] = bool(value)
        elif pd.isna(value):
            normalised[key] = None
        elif isinstance(value, (np.integer, int)):
            normalised[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            normalised[key] = round(float(value), 5)
        else:
            normalised[key] = value
    return normalised


def generate_explorer_payload(
    scenario_id: str,
    seed: int,
    output_root: Path,
) -> dict[str, object]:
    """Generate a reference scenario and return the data required by the explorer."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_id}")

    generate_scenario(
        scenario_id,
        seed_override=seed,
        output_root=output_root,
    )

    features_path = output_root / scenario_id / "daily_features.csv"
    features = pd.read_csv(features_path)
    scored, metrics = evaluate_isolation_forest(features)

    selected_columns = [
        "date",
        "label",
        "severity",
        *FEATURE_COLUMNS,
        "anomaly_score",
        "alarm",
    ]
    records = [
        _normalise_record(record)
        for record in scored[selected_columns].to_dict(orient="records")
    ]

    return {
        "scenario": scenario_id,
        "description": SCENARIO_DESCRIPTIONS[scenario_id],
        "seed": seed,
        "metrics": metrics,
        "features": FEATURE_COLUMNS,
        "records": records,
    }
