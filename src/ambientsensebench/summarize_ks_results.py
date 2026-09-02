from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/scenarios/ks_validation_results.csv")

OUTPUT_COMPACT = Path("data/scenarios/ks_validation_compact_summary.csv")
OUTPUT_THESIS = Path("data/scenarios/ks_validation_thesis_table.csv")


def build_compact_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one compact summary row per scenario and state.

    This script no longer performs validation classification itself.
    Classification is produced in validate_scenarios_ks.py according to:
      - ks-gaussian
      - moment-only

    The summary only aggregates the resulting validation_category values.
    """
    rows = []

    grouped = df.groupby(["scenario", "state"], sort=True)

    for (scenario, state), group in grouped:
        ks_group = group[group["validation_strategy"] == "ks-gaussian"]
        moment_group = group[group["validation_strategy"] == "moment-only"]

        rows.append(
            {
                "scenario": scenario,
                "state": state,
                "n_samples_min": group["n_samples"].min(),
                "n_samples_max": group["n_samples"].max(),
                "n_features": len(group),

                "ks_strong": (ks_group["validation_category"] == "Strong").sum(),
                "ks_acceptable": (ks_group["validation_category"] == "Acceptable").sum(),
                "ks_distributional_difference": (
                    ks_group["validation_category"] == "Distributional difference"
                ).sum(),

                "moment_strong": (moment_group["validation_category"] == "Strong").sum(),
                "moment_acceptable": (
                    moment_group["validation_category"] == "Acceptable"
                ).sum(),
                "moment_mean_mismatch": (
                    moment_group["validation_category"] == "Mean mismatch"
                ).sum(),

                "mean_ks_statistic": ks_group["ks_statistic"].mean(),
                "median_ks_statistic": ks_group["ks_statistic"].median(),

                "mean_relative_error": group["relative_error"].mean(),
                "mean_std_relative_error": group["std_relative_error"].mean(),
            }
        )

    compact_df = pd.DataFrame(rows)
    compact_df["overall_comment"] = compact_df.apply(make_overall_comment, axis=1)

    return compact_df


def make_overall_comment(row: pd.Series) -> str:
    if (
        row["ks_distributional_difference"] == 0
        and row["moment_mean_mismatch"] == 0
    ):
        return "Overall aligned"

    if row["ks_distributional_difference"] <= 1 and row["moment_mean_mismatch"] == 0:
        return "Mostly aligned; minor KS-level deviation"

    return "Feature-level deviations require review"


def build_thesis_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a thesis-oriented feature-level validation table.

    Moment-only features keep NaN for ks_statistic and p_value.
    KS-gaussian features keep their KS statistic and p-value for reporting,
    but the category itself is based on KS statistic and mean relative error,
    not on p-value.
    """
    cols = [
        "scenario",
        "state",
        "feature",
        "validation_strategy",
        "validation_category",
        "n_samples",
        "empirical_mean",
        "empirical_std",
        "reference_distribution",
        "reference_mean",
        "reference_std",
        "relative_error",
        "std_relative_error",
        "ks_statistic",
        "p_value",
    ]

    thesis_df = df[cols].copy()
    thesis_df = thesis_df.sort_values(["scenario", "state", "feature"])

    return thesis_df


def check_required_columns(df: pd.DataFrame) -> None:
    required_columns = {
        "scenario",
        "state",
        "feature",
        "n_samples",
        "validation_strategy",
        "validation_category",
        "empirical_mean",
        "empirical_std",
        "reference_distribution",
        "reference_mean",
        "reference_std",
        "relative_error",
        "std_relative_error",
        "ks_statistic",
        "p_value",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise KeyError(
            f"Missing required columns in {INPUT_PATH}: {sorted(missing)}"
        )


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    check_required_columns(df)

    compact_df = build_compact_summary(df)
    thesis_df = build_thesis_table(df)

    OUTPUT_COMPACT.parent.mkdir(parents=True, exist_ok=True)

    compact_df.to_csv(OUTPUT_COMPACT, index=False)
    thesis_df.to_csv(OUTPUT_THESIS, index=False)

    print("\nD6 summary files generated:")
    print(f"- Validation results: {INPUT_PATH}")
    print(f"- Compact summary: {OUTPUT_COMPACT}")
    print(f"- Thesis table: {OUTPUT_THESIS}")

    print("\nCompact summary:")
    print(compact_df.to_string(index=False))

    print("\nThesis-oriented table:")
    print(thesis_df.to_string(index=False))


if __name__ == "__main__":
    main()