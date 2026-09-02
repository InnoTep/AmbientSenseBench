import os
import pandas as pd

from sklearn.ensemble import IsolationForest

from .wp6_evaluation import (
    SCENARIOS,
    FEATURE_COLUMNS,
    load_scenario,
    split_train_test,
    prepare_train_test_arrays,
    run_model_inference_per_day,
    evaluate_model_outputs,
)


BASE_DIR = "data/scenarios"

RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(RESULTS_DIR, "wp6_isolation_forest_results.csv")


def run_isolation_forest_for_scenario(scenario_id, base_dir="data/scenarios"):
    df = load_scenario(base_dir, scenario_id)
    train_df, test_df = split_train_test(df, scenario_id)

    X_train_scaled, X_test_scaled, y_test, scaler = prepare_train_test_arrays(
        train_df,
        test_df,
    )

    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
    )

    model.fit(X_train_scaled)

    # Higher score must mean more anomalous.
    train_anomaly_scores = -model.decision_function(X_train_scaled)

    (
        test_anomaly_scores,
        inference_times_ms,
        peak_rss_mb,
    ) = run_model_inference_per_day(
        model,
        X_test_scaled,
    )

    result = evaluate_model_outputs(
        scenario_id=scenario_id,
        model_name="Isolation Forest",
        test_df=test_df,
        y_test=y_test,
        train_anomaly_scores=train_anomaly_scores,
        test_anomaly_scores=test_anomaly_scores,
        inference_times_ms=inference_times_ms,
        peak_rss_mb=peak_rss_mb,
        model=model,
        scaler=scaler,
    )

    return result


def main():
    results = []

    for scenario_id in SCENARIOS:
        print(f"Running Isolation Forest for {scenario_id}...")
        result = run_isolation_forest_for_scenario(scenario_id)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nIsolation Forest results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
