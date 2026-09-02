import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


SCENARIOS = ["P01", "P02", "P03", "P04"]

BASELINE_MAX_SEVERITY = 0.1
DISTRESS_MIN_SEVERITY = 0.9


FEATURES = [
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


PROFILE_KEY_MAP = {
    "wake_time": {
        "distribution": "normal",
        "mean_key": "wake_mean",
        "std_key": "wake_std",
    },
    "sleep_hours": {
        "distribution": "normal",
        "mean_key": "sleep_mean",
        "std_key": "sleep_std",
    },
    "meal_count": {
        "distribution": "normal",
        "mean_key": "meal_mean",
        "std_key": "meal_std",
    },
    "social_proxy": {
        "distribution": "normal",
        "mean_key": "social_mean",
        "std_key": "social_std",
    },
    "night_activity": {
        "distribution": "poisson",
        "lambda_key": "night_lambda",
    },
    "mobility_score": {
        "distribution": "normal",
        "mean_key": "mobility_mean",
        "std_key": "mobility_std",
    },
    "kitchen_activity_score": {
        "distribution": "normal",
        "mean_key": "kitchen_mean",
        "std_key": "kitchen_std",
    },
    "room_transition_entropy": {
        "distribution": "normal",
        "mean_key": "transition_entropy_mean",
        "std_key": "transition_entropy_std",
    },
    "evening_routine_consistency": {
        "distribution": "normal",
        "mean_key": "evening_consistency_mean",
        "std_key": "evening_consistency_std",
    },
}


def load_profiles(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        profiles = yaml.safe_load(f)

    if "baseline" not in profiles:
        raise KeyError("Missing 'baseline' in profiles.yaml")

    if "distress" not in profiles:
        raise KeyError("Missing 'distress' in profiles.yaml")

    return profiles


def get_reference_distribution(profiles: dict, state: str, feature: str) -> dict:
    """
    Return the reference distribution information for a given state and feature.

    For normal features:
      {
        distribution: normal,
        mean: ...,
        std: ...
      }

    For night_activity:
      {
        distribution: poisson,
        lambda: ...
      }
    """
    if state not in profiles:
        raise KeyError(f"State '{state}' not found in profiles.yaml")

    if feature not in PROFILE_KEY_MAP:
        raise KeyError(f"Feature '{feature}' not found in PROFILE_KEY_MAP")

    state_profile = profiles[state]
    mapping = PROFILE_KEY_MAP[feature]
    distribution = mapping["distribution"]

    if distribution == "normal":
        mean_key = mapping["mean_key"]
        std_key = mapping["std_key"]

        if mean_key not in state_profile:
            raise KeyError(f"Missing key '{mean_key}' for {state}.{feature}")

        if std_key not in state_profile:
            raise KeyError(f"Missing key '{std_key}' for {state}.{feature}")

        mean = float(state_profile[mean_key])
        std = float(state_profile[std_key])

        if std <= 0:
            raise ValueError(f"Reference std must be > 0 for {state}.{feature}. Got {std}")

        return {
            "distribution": "normal",
            "mean": mean,
            "std": std,
        }

    if distribution == "poisson":
        lambda_key = mapping["lambda_key"]

        if lambda_key not in state_profile:
            raise KeyError(f"Missing key '{lambda_key}' for {state}.{feature}")

        lam = float(state_profile[lambda_key])

        if lam <= 0:
            raise ValueError(f"Poisson lambda must be > 0 for {state}.{feature}. Got {lam}")

        return {
            "distribution": "poisson",
            "lambda": lam,
        }

    raise ValueError(f"Unsupported distribution type: {distribution}")

def convert_feature_samples(samples: pd.Series, feature: str) -> np.ndarray:
    """
    Convert feature samples into numeric values for KS validation.

    wake_time is stored as HH:MM:SS, so it must be converted to decimal hour.
    Example:
      06:57:00 -> 6.95
    """
    clean_samples = samples.dropna()

    if feature == "wake_time":
        parsed_time = pd.to_datetime(clean_samples, format="%H:%M:%S", errors="coerce")

        decimal_hour = (
            parsed_time.dt.hour
            + parsed_time.dt.minute / 60.0
            + parsed_time.dt.second / 3600.0
        )

        return decimal_hour.dropna().astype(float).to_numpy()

    return clean_samples.astype(float).to_numpy()

def ks_test_against_reference(
    samples: pd.Series,
    reference: dict,
    feature: str,
) -> tuple[float, float, int]:
    clean_samples = convert_feature_samples(samples, feature)
    n = len(clean_samples)

    if n == 0:
        return np.nan, np.nan, 0

    distribution = reference["distribution"]

    if distribution == "normal":
        mean = reference["mean"]
        std = reference["std"]

        result = stats.kstest(
            clean_samples,
            lambda x: stats.norm.cdf(x, loc=mean, scale=std),
        )

        return float(result.statistic), float(result.pvalue), n

    if distribution == "poisson":
        lam = reference["lambda"]

        count_samples = np.rint(clean_samples).astype(int)

        result = stats.kstest(
            count_samples,
            lambda x: stats.poisson.cdf(x, mu=lam),
        )

        return float(result.statistic), float(result.pvalue), n

    raise ValueError(f"Unsupported distribution type: {distribution}")

def compute_relative_error(empirical_mean: float, reference_mean: float) -> float:
    if pd.isna(empirical_mean) or pd.isna(reference_mean):
        return np.nan

    if reference_mean == 0:
        return np.nan

    return abs(empirical_mean - reference_mean) / abs(reference_mean)


def classify_moment_status(mean_rel_err: float, std_rel_err: float) -> str:
    if pd.isna(mean_rel_err) or pd.isna(std_rel_err):
        return "insufficient_samples"

    if mean_rel_err <= 0.10 and std_rel_err <= 0.20:
        return "Strong"

    if mean_rel_err <= 0.20 and std_rel_err <= 0.30:
        return "Acceptable"

    return "Mean mismatch"


def classify_ks_alignment(ks_statistic: float, mean_rel_err: float) -> str:
    if pd.isna(ks_statistic) or pd.isna(mean_rel_err):
        return "insufficient_samples"

    if ks_statistic <= 0.15 and mean_rel_err <= 0.10:
        return "Strong"

    if ks_statistic <= 0.30 and mean_rel_err <= 0.20:
        return "Acceptable"

    return "Distributional difference"

    

def select_steady_state(df: pd.DataFrame, state: str) -> pd.DataFrame:
    if "severity" not in df.columns:
        raise KeyError(
            "Column 'severity' not found in daily_features.csv. "
            "D6 requires severity to select steady-state periods."
        )

    if state == "baseline":
        return df[df["severity"] <= BASELINE_MAX_SEVERITY].copy()

    if state == "distress":
        return df[df["severity"] >= DISTRESS_MIN_SEVERITY].copy()

    raise ValueError(f"Unknown state: {state}")


def validate_one_scenario(
    scenario_id: str,
    scenario_root: Path,
    profiles: dict,
    feature_validation: dict,
) -> list[dict]:
    daily_features_path = scenario_root / scenario_id / "daily_features.csv"

    if not daily_features_path.exists():
        raise FileNotFoundError(f"Missing file: {daily_features_path}")

    df = pd.read_csv(daily_features_path)

    print(f"\nLoaded {daily_features_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    results = []

    for state in ["baseline", "distress"]:
        steady_df = select_steady_state(df, state)

        if state == "baseline":
            steady_rule = f"severity <= {BASELINE_MAX_SEVERITY}"
        else:
            steady_rule = f"severity >= {DISTRESS_MIN_SEVERITY}"

        print(f"{scenario_id} | {state} steady-state samples: {len(steady_df)} ({steady_rule})")

        for feature in FEATURES:
            if feature not in steady_df.columns:
                print(f"[WARNING] Feature '{feature}' not found in {daily_features_path}. Skipping.")
                continue

            if feature not in feature_validation:
                raise KeyError(f"Missing validation strategy for feature: {feature}")

            validation_strategy = feature_validation[feature]

            reference = get_reference_distribution(
                profiles=profiles,
                state=state,
                feature=feature,
            )

            empirical_samples = pd.Series(
                convert_feature_samples(steady_df[feature], feature)
            )

            n_samples = len(empirical_samples)

            if reference["distribution"] == "normal":
                reference_mean = reference["mean"]
                reference_std = reference["std"]
                reference_lambda = np.nan
            elif reference["distribution"] == "poisson":
                reference_mean = reference["lambda"]
                reference_std = np.sqrt(reference["lambda"])
                reference_lambda = reference["lambda"]
            else:
                reference_mean = np.nan
                reference_std = np.nan
                reference_lambda = np.nan

            empirical_mean = empirical_samples.mean() if n_samples > 0 else np.nan
            empirical_std = empirical_samples.std(ddof=1) if n_samples > 1 else np.nan

            relative_error = compute_relative_error(empirical_mean, reference_mean)
            std_relative_error = compute_relative_error(empirical_std, reference_std)
                    
            if validation_strategy == "moment-only":
                ks_statistic = np.nan
                p_value = np.nan
                validation_category = classify_moment_status(
                    mean_rel_err=relative_error,
                    std_rel_err=std_relative_error,
                )

            elif validation_strategy == "ks-gaussian":
                ks_statistic, p_value, n_samples = ks_test_against_reference(
                    samples=steady_df[feature],
                    reference=reference,
                    feature=feature,
                )

                validation_category = classify_ks_alignment(
                    ks_statistic=ks_statistic,
                    mean_rel_err=relative_error,
                )


            else:
                raise ValueError(
                    f"Unsupported validation strategy for {feature}: {validation_strategy}"
                )

            results.append(
                {
                    "scenario": scenario_id,
                    "state": state,
                    "feature": feature,
                    "n_samples": n_samples,
                    "empirical_mean": empirical_mean,
                    "empirical_std": empirical_std,
                    "reference_distribution": reference["distribution"],
                    "reference_mean": reference_mean,
                    "reference_std": reference_std,
                    "reference_lambda": reference_lambda,
                    "relative_error": relative_error,
                    "relative_error": relative_error,
                    "std_relative_error": std_relative_error,
                    "validation_strategy": validation_strategy,
                    "validation_category": validation_category,
                    "ks_statistic": ks_statistic,
                    "p_value": p_value,
                    "steady_state_rule": steady_rule,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="D6 KS validation for synthetic behavioral monitoring scenarios."
    )

    parser.add_argument(
        "--scenario-dir",
        type=str,
        default="data/scenarios",
        help="Directory containing P01-P04 scenario folders.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/profiles.yaml",
        help="Path to profiles.yaml.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/scenarios/ks_validation_results.csv",
        help="Output CSV path for KS validation results.",
    )

    args = parser.parse_args()

    scenario_root = Path(args.scenario_dir)
    config_path = Path(args.config)
    output_path = Path(args.output)

    profiles = load_profiles(config_path)

    if "feature_validation" not in profiles:
        raise KeyError("Missing 'feature_validation' section in profiles.yaml")

    feature_validation = profiles["feature_validation"]
    all_results = []

    for scenario_id in SCENARIOS:
        print(f"\nValidating scenario {scenario_id}...")
        scenario_results = validate_one_scenario(
            scenario_id=scenario_id,
            scenario_root=scenario_root,
            profiles=profiles,
            feature_validation=feature_validation,
        )
        all_results.extend(scenario_results)

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        print("\nNo KS validation results were generated.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\nD6 KS validation completed.")
    print(f"Saved results to: {output_path}")

    summary_cols = [
    "scenario",
    "state",
    "feature",
    "n_samples",
    "validation_strategy",
    "validation_category",
    "empirical_mean",
    "empirical_std",
    "reference_mean",
    "reference_std",
    "relative_error",
    "std_relative_error",
    "ks_statistic",
    "p_value",
    ]

    print("\nSummary:")
    print(results_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()