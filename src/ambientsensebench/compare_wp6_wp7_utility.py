"""
WP6 vs WP7 Utility Comparison

This script compares the original non-private WP6 anomaly-detection results
with the WP7 epsilon=1.0 DP-perturbed anomaly-detection results.

Inputs:
- data/results/wp6_anomaly_detection_results.csv
- data/results/wp7_dp_anomaly_detection_results.csv

Output:
- data/results/wp7_dp_utility_comparison.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WP6_FILE = Path("data/results/wp6_anomaly_detection_results.csv")
WP7_FILE = Path("data/results/wp7_dp_anomaly_detection_results.csv")
OUTPUT_FILE = Path("data/results/wp7_dp_utility_comparison.csv")


def safe_ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def normalise_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Standardise column names across WP6/WP7 result files.
    """
    df = df.copy()

    rename_map = {}

    if "f1_fixed_threshold" in df.columns:
        rename_map["f1_fixed_threshold"] = "f1_fixed"

    if "mean_lead_time_days" in df.columns:
        rename_map["mean_lead_time_days"] = "lead_time_days"

    if "n_warnings_total" in df.columns:
        rename_map["n_warnings_total"] = "total_alarms"

    if "n_true_alarms" in df.columns:
        rename_map["n_true_alarms"] = "true_alarms"

    if "n_false_alarms" in df.columns:
        rename_map["n_false_alarms"] = "false_alarms"

    df = df.rename(columns=rename_map)

    required = [
        "scenario",
        "model",
        "auprc",
        "f1_fixed",
        "lead_time_days",
        "total_alarms",
        "true_alarms",
        "false_alarms",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")

    return df[required].copy()


def main():
    if not WP6_FILE.exists():
        raise FileNotFoundError(f"Missing WP6 file: {WP6_FILE}")

    if not WP7_FILE.exists():
        raise FileNotFoundError(f"Missing WP7 file: {WP7_FILE}")

    wp6 = pd.read_csv(WP6_FILE)
    wp7 = pd.read_csv(WP7_FILE)

    wp6 = normalise_columns(wp6, "WP6")
    wp7 = normalise_columns(wp7, "WP7")

    wp6 = wp6.rename(
        columns={
            "auprc": "wp6_auprc",
            "f1_fixed": "wp6_f1_fixed",
            "lead_time_days": "wp6_lead_time_days",
            "total_alarms": "wp6_total_alarms",
            "true_alarms": "wp6_true_alarms",
            "false_alarms": "wp6_false_alarms",
        }
    )

    wp7 = wp7.rename(
        columns={
            "auprc": "wp7_auprc",
            "f1_fixed": "wp7_f1_fixed",
            "lead_time_days": "wp7_lead_time_days",
            "total_alarms": "wp7_total_alarms",
            "true_alarms": "wp7_true_alarms",
            "false_alarms": "wp7_false_alarms",
        }
    )

    comparison = pd.merge(
        wp6,
        wp7,
        on=["scenario", "model"],
        how="inner",
        validate="one_to_one",
    )

    comparison["auprc_drop"] = comparison["wp6_auprc"] - comparison["wp7_auprc"]
    comparison["f1_fixed_drop"] = comparison["wp6_f1_fixed"] - comparison["wp7_f1_fixed"]

    comparison["auprc_retention"] = comparison.apply(
        lambda row: safe_ratio(row["wp7_auprc"], row["wp6_auprc"]),
        axis=1,
    )

    comparison["f1_fixed_retention"] = comparison.apply(
        lambda row: safe_ratio(row["wp7_f1_fixed"], row["wp6_f1_fixed"]),
        axis=1,
    )

    comparison["total_alarm_change"] = (
        comparison["wp7_total_alarms"] - comparison["wp6_total_alarms"]
    )

    comparison["true_alarm_change"] = (
        comparison["wp7_true_alarms"] - comparison["wp6_true_alarms"]
    )

    comparison["false_alarm_change"] = (
        comparison["wp7_false_alarms"] - comparison["wp6_false_alarms"]
    )

    comparison["lead_time_change_days"] = (
        comparison["wp7_lead_time_days"] - comparison["wp6_lead_time_days"]
    )

    numeric_columns = comparison.select_dtypes(include=["number"]).columns
    comparison[numeric_columns] = comparison[numeric_columns].round(4)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_FILE, index=False)

    print(f"[DONE] WP6 vs WP7 utility comparison written to: {OUTPUT_FILE}")
    


if __name__ == "__main__":
    main()