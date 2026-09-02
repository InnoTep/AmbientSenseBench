"""Convert per-day raw event files into the benchmark's daily feature table."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .feature_extraction import extract_all_features, load_events


DATASET_COLUMNS = (
    "date",
    "label",
    "wake_time",
    "sleep_hours",
    "meal_count",
    "social_proxy",
    "night_activity",
    "mobility_score",
    "kitchen_activity_score",
    "room_transition_entropy",
    "evening_routine_consistency",
)


def build_feature_row(events: list[dict[str, Any]], label: str) -> list[Any]:
    """Extract one daily feature record in the stable CSV column order."""
    features = extract_all_features(events)
    wake_time = features["wake_time"]

    return [
        wake_time.strftime("%Y-%m-%d") if wake_time else "",
        label,
        wake_time.strftime("%H:%M:%S") if wake_time else "",
        round(features["sleep_hours"], 2)
        if features["sleep_hours"] is not None
        else "",
        features["meal_count"],
        features["social_proxy"],
        features["night_activity"],
        round(features["mobility_score"], 2)
        if features["mobility_score"] is not None
        else "",
        round(features["kitchen_score"], 2)
        if features["kitchen_score"] is not None
        else "",
        round(features["room_transition_entropy"], 2)
        if features["room_transition_entropy"] is not None
        else "",
        round(features["evening_routine_consistency"], 2)
        if features["evening_routine_consistency"] is not None
        else "",
    ]


def build_daily_dataset(
    input_folder: str | Path,
    output_file: str | Path,
    label: str,
) -> None:
    """Append features from all daily CSVs in ``input_folder`` to ``output_file``.

    The append behaviour is retained for compatibility with existing analysis
    workflows. Scenario generation starts with a fresh scenario directory, so
    its produced table always has one row per generated day.
    """
    input_path = Path(input_folder)
    output_path = Path(output_file)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Raw-day directory does not exist: {input_path}")

    rows = [
        build_feature_row(load_events(path), label)
        for path in sorted(input_path.glob("*.csv"))
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(DATASET_COLUMNS)
        writer.writerows(rows)


if __name__ == "__main__":
    build_daily_dataset("../data/raw_days", "../data/daily_features.csv", "baseline")
