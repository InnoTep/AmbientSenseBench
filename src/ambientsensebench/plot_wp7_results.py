"""
WP7 Plotting Script

This script generates figures for the WP7 epsilon=1.0 differential privacy
utility evaluation.

Input:
- data/results/wp7_dp_utility_comparison.csv

Outputs:
- data/results/figures/wp7_auprc_comparison.png
- data/results/figures/wp7_f1_comparison.png
- data/results/figures/wp7_utility_retention.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path("data/results/wp7_dp_utility_comparison.csv")
FIGURE_DIR = Path("data/results/figures")


def plot_grouped_bar(
    df: pd.DataFrame,
    metric_wp6: str,
    metric_wp7: str,
    ylabel: str,
    title: str,
    output_file: Path,
) -> None:
    """
    Plot grouped WP6 vs WP7 bars for each scenario and model.
    """
    labels = [
        f"{row.scenario}\n{row.model.replace(' ', '\n')}"
        for row in df.itertuples(index=False)
    ]

    x = range(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        [i - width / 2 for i in x],
        df[metric_wp6],
        width,
        label="WP6 original",
    )

    ax.bar(
        [i + width / 2 for i in x],
        df[metric_wp7],
        width,
        label="WP7 DP ε=1.0",
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

    print(f"[OK] Saved {output_file}")


def plot_retention(df: pd.DataFrame, output_file: Path) -> None:
    """
    Plot AUPRC and fixed-threshold F1 retention after DP perturbation.
    """
    labels = [
        f"{row.scenario}\n{row.model.replace(' ', '\n')}"
        for row in df.itertuples(index=False)
    ]

    x = range(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        [i - width / 2 for i in x],
        df["auprc_retention"],
        width,
        label="AUPRC retention",
    )

    ax.bar(
        [i + width / 2 for i in x],
        df["f1_fixed_retention"],
        width,
        label="Fixed-threshold F1 retention",
    )

    ax.axhline(1.0, linestyle="--", linewidth=1)

    ax.set_ylabel("Retention ratio")
    ax.set_title("WP7 utility retention under DP ε=1.0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1.15)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

    print(f"[OK] Saved {output_file}")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    expected_rows = 8
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(df)}")

    plot_grouped_bar(
        df=df,
        metric_wp6="wp6_auprc",
        metric_wp7="wp7_auprc",
        ylabel="AUPRC",
        title="WP6 vs WP7 AUPRC under DP ε=1.0",
        output_file=FIGURE_DIR / "wp7_auprc_comparison.png",
    )

    plot_grouped_bar(
        df=df,
        metric_wp6="wp6_f1_fixed",
        metric_wp7="wp7_f1_fixed",
        ylabel="Fixed-threshold F1",
        title="WP6 vs WP7 fixed-threshold F1 under DP ε=1.0",
        output_file=FIGURE_DIR / "wp7_f1_comparison.png",
    )

    plot_retention(
        df=df,
        output_file=FIGURE_DIR / "wp7_utility_retention.png",
    )

    print("[DONE] WP7 figures generated.")


if __name__ == "__main__":
    main()