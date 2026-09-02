import pandas as pd

from ambientsensebench.generate_scenarios import generate_scenario


def test_scenario_generation_uses_requested_output_directory(tmp_path):
    generate_scenario(
        "P01",
        seed_override=7,
        n_days=3,
        output_root=tmp_path,
    )

    scenario_directory = tmp_path / "P01"
    features = pd.read_csv(scenario_directory / "daily_features.csv")
    labels = pd.read_csv(scenario_directory / "scenario_labels.csv")

    assert len(features) == 3
    assert len(labels) == 3
    assert list(features["label"]) == list(labels["label"])
    assert len(list((scenario_directory / "raw_days").glob("*.csv"))) == 3
