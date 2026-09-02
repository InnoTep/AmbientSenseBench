import os
import pandas as pd

from .run_isolation_forest import run_isolation_forest_for_scenario
from .run_one_class_svm import run_one_class_svm_for_scenario
from .wp6_evaluation import SCENARIOS


RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ISOLATION_FOREST_OUTPUT = os.path.join(
    RESULTS_DIR,
    "wp6_isolation_forest_results.csv",
)

ONE_CLASS_SVM_OUTPUT = os.path.join(
    RESULTS_DIR,
    "wp6_one_class_svm_results.csv",
)

COMBINED_OUTPUT = os.path.join(
    RESULTS_DIR,
    "wp6_anomaly_detection_results.csv",
)


def main():
    isolation_forest_results = []
    one_class_svm_results = []

    for scenario_id in SCENARIOS:
        print(f"Running Isolation Forest for {scenario_id}...")
        isolation_forest_results.append(
            run_isolation_forest_for_scenario(scenario_id)
        )

    for scenario_id in SCENARIOS:
        print(f"Running One-Class SVM for {scenario_id}...")
        one_class_svm_results.append(
            run_one_class_svm_for_scenario(scenario_id)
        )

    isolation_forest_df = pd.DataFrame(isolation_forest_results)
    one_class_svm_df = pd.DataFrame(one_class_svm_results)

    combined_df = pd.concat(
        [isolation_forest_df, one_class_svm_df],
        ignore_index=True,
    )

    isolation_forest_df.to_csv(ISOLATION_FOREST_OUTPUT, index=False)
    one_class_svm_df.to_csv(ONE_CLASS_SVM_OUTPUT, index=False)
    combined_df.to_csv(COMBINED_OUTPUT, index=False)

    print("\nCombined WP6 anomaly detection results:")
    print(combined_df.to_string(index=False))

    print(f"\nSaved Isolation Forest results to: {ISOLATION_FOREST_OUTPUT}")
    print(f"Saved One-Class SVM results to: {ONE_CLASS_SVM_OUTPUT}")
    print(f"Saved combined results to: {COMBINED_OUTPUT}")


if __name__ == "__main__":
    main()
