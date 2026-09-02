from datetime import datetime

import pandas as pd

from ambientsensebench.build_dataset import DATASET_COLUMNS, build_daily_dataset
from ambientsensebench.generate_raw_data import generate_one_day, save_day


def test_daily_dataset_builder_writes_the_expected_schema(tmp_path):
    raw_days = tmp_path / "raw_days"
    raw_days.mkdir()
    events = generate_one_day(datetime(2026, 1, 1), seed_override=7)
    save_day(events, raw_days / "day_001.csv")

    output_file = tmp_path / "nested" / "daily_features.csv"
    build_daily_dataset(raw_days, output_file, label="baseline")

    dataset = pd.read_csv(output_file)
    assert tuple(dataset.columns) == DATASET_COLUMNS
    assert len(dataset) == 1
    assert dataset.loc[0, "label"] == "baseline"
    assert dataset.loc[0, "wake_time"]
