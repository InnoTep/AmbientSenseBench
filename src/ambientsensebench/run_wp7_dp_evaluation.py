import os
import pandas as pd

from .run_isolation_forest import run_isolation_forest_for_scenario
from .run_one_class_svm import run_one_class_svm_for_scenario
from .wp6_evaluation import SCENARIOS


RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DP_SCENARIO_DIR = "data/scenarios_dp_epsilon_1"

ISOLATION_FOREST_OUTPUT = os.path.join(
    RESULTS_DIR,
    "wp7_dp_isolation_forest_results.csv",
)

ONE_CLASS_SVM_OUTPUT = os.path.join(
    RESULTS_DIR,
    "wp7_dp_one_class_svm_results.csv",
)

COMBINED_OUTPUT = os.path.join(
    RESULTS_DIR,
    "wp7_dp_anomaly_detection_results.csv",
)


def add_wp7_metadata(result):
    result = result.copy()
    result["wp"] = "WP7"
    result["privacy_condition"] = "DP_epsilon_1"
    result["epsilon"] = 1.0
    return result


def main():
    isolation_forest_results = []
    one_class_svm_results = []

    for scenario_id in SCENARIOS:
        print(f"Running WP7 DP Isolation Forest for {scenario_id}...")
        result = run_isolation_forest_for_scenario(
            scenario_id,
            base_dir=DP_SCENARIO_DIR,
        )
        isolation_forest_results.append(add_wp7_metadata(result))

    for scenario_id in SCENARIOS:
        print(f"Running WP7 DP One-Class SVM for {scenario_id}...")
        result = run_one_class_svm_for_scenario(
            scenario_id,
            base_dir=DP_SCENARIO_DIR,
        )
        one_class_svm_results.append(add_wp7_metadata(result))

    isolation_forest_df = pd.DataFrame(isolation_forest_results)
    one_class_svm_df = pd.DataFrame(one_class_svm_results)

    combined_df = pd.concat(
        [isolation_forest_df, one_class_svm_df],
        ignore_index=True,
    )

    preferred_order = [
        "wp",
        "privacy_condition",
        "epsilon",
        "scenario",
        "model",
        "auprc",
        "f1_fixed_threshold",
        "threshold_rule",
        "fixed_threshold",
        "oracle_f1",
        "oracle_threshold",
        "mean_lead_time_days",
        "n_episodes",
        "n_pre_onset_warnings",
        "n_onset_day_warnings",
        "n_no_warning",
        "episode_lead_times",
        "n_warnings_total",
        "n_true_alarms",
        "n_false_alarms",
        "n_train",
        "n_test",
        "n_distress_test",
        "n_baseline_test",
        "positive_rate_test",
        "median_inference_ms",
        "p95_inference_ms",
        "peak_rss_mb",
        "model_size_kb",
    ]

    existing_preferred = [
        column for column in preferred_order if column in combined_df.columns
    ]
    remaining_columns = [
        column for column in combined_df.columns if column not in existing_preferred
    ]

    combined_df = combined_df[existing_preferred + remaining_columns]

    isolation_forest_df.to_csv(ISOLATION_FOREST_OUTPUT, index=False)
    one_class_svm_df.to_csv(ONE_CLASS_SVM_OUTPUT, index=False)
    combined_df.to_csv(COMBINED_OUTPUT, index=False)

    print("\nCombined WP7 DP anomaly detection results:")
    print(combined_df.to_string(index=False))

    print(f"\nSaved WP7 DP Isolation Forest results to: {ISOLATION_FOREST_OUTPUT}")
    print(f"Saved WP7 DP One-Class SVM results to: {ONE_CLASS_SVM_OUTPUT}")
    print(f"Saved WP7 DP combined results to: {COMBINED_OUTPUT}")


if __name__ == "__main__":
    main()
