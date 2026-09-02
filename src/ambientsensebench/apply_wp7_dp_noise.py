"""
WP7 Differential Privacy Noise Application

This script applies day-level Laplace differential privacy noise to the
behavioural feature vectors used by the WP6 anomaly-detection pipeline.

Scope:
- Single operational privacy point: epsilon = 1.0
- Day-level feature perturbation
- Labels and severity are preserved only for retrospective evaluation
- The WP6 model-input feature set is reused unchanged
- Output files are written to data/scenarios_dp_epsilon_1/P01-P04/daily_features.csv

This script does not train or evaluate anomaly-detection models.
It only creates the DP-perturbed feature datasets for WP7.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


SCENARIOS = ["P01", "P02", "P03", "P04"]

MODEL_INPUT_FEATURES = [
    "wake_time",
    "sleep_hours",
    "meal_count",
    "social_proxy",
    "night_activity",
    "mobility_score",
    "kitchen_activity_score",
    "room_transition_entropy",
    "evening_routine_consistency",
]


FEATURE_SENSITIVITIES: Dict[str, float] = {
    # Hour-based behavioural features
    "wake_time": 1.0,
    "sleep_hours": 1.0,

    # Count-like daily behavioural features
    "meal_count": 1.0,
    "social_proxy": 1.0,
    "night_activity": 1.0,

    # Normalised or bounded behavioural scores
    "mobility_score": 0.2,
    "kitchen_activity_score": 0.2,
    "room_transition_entropy": 0.2,
    "evening_routine_consistency": 0.2,
}


FEATURE_CLIPPING_BOUNDS: Dict[str, Tuple[float, float]] = {
    "wake_time": (0.0, 24.0),
    "sleep_hours": (0.0, 24.0),

    "meal_count": (0.0, 10.0),
    "social_proxy": (0.0, 20.0),
    "night_activity": (0.0, 30.0),

    "mobility_score": (0.0, 1.5),
    "kitchen_activity_score": (0.0, 1.5),
    "room_transition_entropy": (0.0, 2.0),
    "evening_routine_consistency": (0.0, 1.0),
}


def convert_wake_time_to_decimal_hour(value: object) -> float:
    """
    Convert wake_time to decimal hour.

    The scenario CSV may store wake_time either as:
    - decimal hour, e.g. 7.5
    - HH:MM:SS string, e.g. "07:30:00"

    Returns:
        Decimal hour as float.
    """
    if pd.isna(value):
        raise ValueError("wake_time contains NaN.")

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    value_str = str(value).strip()

    if ":" not in value_str:
        return float(value_str)

    parts = value_str.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid wake_time format: {value}")

    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) > 2 else 0

    return hour + minute / 60.0 + second / 3600.0


def apply_laplace_noise(
    df: pd.DataFrame,
    epsilon: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Apply Laplace DP noise to the WP6 model-input features.

    Args:
        df: Original scenario daily feature dataframe.
        epsilon: Differential privacy epsilon.
        rng: NumPy random generator for reproducibility.

    Returns:
        A new dataframe with DP-perturbed feature values.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    noisy_df = df.copy()

    missing_features = [f for f in MODEL_INPUT_FEATURES if f not in noisy_df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    # Convert wake_time before adding noise so that the DP file is directly usable
    # by the same WP6 evaluation protocol.
    noisy_df["wake_time"] = noisy_df["wake_time"].apply(convert_wake_time_to_decimal_hour)

    for feature in MODEL_INPUT_FEATURES:
        sensitivity = FEATURE_SENSITIVITIES[feature]
        scale = sensitivity / epsilon

        original_values = noisy_df[feature].astype(float).to_numpy()
        noise = rng.laplace(loc=0.0, scale=scale, size=len(noisy_df))

        noisy_values = original_values + noise

        lower, upper = FEATURE_CLIPPING_BOUNDS[feature]
        noisy_values = np.clip(noisy_values, lower, upper)

        noisy_df[feature] = noisy_values

    return noisy_df


def validate_preserved_columns(original_df: pd.DataFrame, noisy_df: pd.DataFrame) -> None:
    """
    Validate that evaluation-only columns are preserved.

    date, label, and severity must not be changed by WP7 noise application.
    They are used only for retrospective evaluation and are not model inputs.
    """
    preserved_columns = ["date", "label", "severity"]

    for column in preserved_columns:
        if column not in original_df.columns:
            raise ValueError(f"Original dataframe is missing required column: {column}")
        if column not in noisy_df.columns:
            raise ValueError(f"Noisy dataframe is missing required column: {column}")

        original_values = original_df[column].astype(str).tolist()
        noisy_values = noisy_df[column].astype(str).tolist()

        if original_values != noisy_values:
            raise AssertionError(f"Column was unexpectedly modified: {column}")


def compute_noise_summary(
    original_df: pd.DataFrame,
    noisy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute a compact feature-level noise summary for reporting and debugging.
    """
    rows = []

    original_for_comparison = original_df.copy()
    original_for_comparison["wake_time"] = original_for_comparison["wake_time"].apply(
        convert_wake_time_to_decimal_hour
    )

    for feature in MODEL_INPUT_FEATURES:
        original_values = original_for_comparison[feature].astype(float)
        noisy_values = noisy_df[feature].astype(float)
        diff = noisy_values - original_values

        rows.append(
            {
                "feature": feature,
                "sensitivity": FEATURE_SENSITIVITIES[feature],
                "mean_original": original_values.mean(),
                "mean_noisy": noisy_values.mean(),
                "mean_absolute_noise": diff.abs().mean(),
                "std_noise": diff.std(ddof=1),
                "min_noisy": noisy_values.min(),
                "max_noisy": noisy_values.max(),
            }
        )

    return pd.DataFrame(rows)


def process_scenario(
    scenario_id: str,
    input_root: Path,
    output_root: Path,
    epsilon: float,
    base_seed: int,
) -> pd.DataFrame:
    """
    Apply WP7 DP perturbation to one scenario.
    """
    input_file = input_root / scenario_id / "daily_features.csv"
    output_dir = output_root / scenario_id
    output_file = output_dir / "daily_features.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Input scenario file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Scenario-specific deterministic seed.
    scenario_offset = int(scenario_id.replace("P", ""))
    rng = np.random.default_rng(base_seed + scenario_offset)

    original_df = pd.read_csv(input_file)
    noisy_df = apply_laplace_noise(original_df, epsilon=epsilon, rng=rng)

    validate_preserved_columns(original_df, noisy_df)

    noisy_df.to_csv(output_file, index=False)

    summary_df = compute_noise_summary(original_df, noisy_df)
    summary_df.insert(0, "scenario", scenario_id)

    print(f"[OK] {scenario_id}: wrote {output_file}")

    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply WP7 epsilon=1.0 day-level DP noise to scenario features."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/scenarios"),
        help="Input root containing P01-P04 scenario folders.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/scenarios_dp_epsilon_1"),
        help="Output root for DP-perturbed scenario folders.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Differential privacy epsilon. WP7 default is 1.0.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7001,
        help="Base random seed for reproducible DP noise.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/results/wp7_dp_noise_summary.csv"),
        help="Output CSV for feature-level DP noise summary.",
    )

    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for scenario_id in SCENARIOS:
        scenario_summary = process_scenario(
            scenario_id=scenario_id,
            input_root=args.input_root,
            output_root=args.output_root,
            epsilon=args.epsilon,
            base_seed=args.seed,
        )
        all_summaries.append(scenario_summary)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_df.to_csv(args.summary_output, index=False)

    print(f"[OK] DP noise summary written to {args.summary_output}")
    print("[DONE] WP7 DP-perturbed scenario generation completed.")


if __name__ == "__main__":
    main()