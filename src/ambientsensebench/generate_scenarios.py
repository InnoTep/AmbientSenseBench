import os
import math
import shutil
import tempfile
from datetime import datetime, timedelta

from .generate_raw_data import (
    load_profiles,
    generate_one_day,
    save_day,
)
from .build_dataset import build_daily_dataset


N_DAYS = 365
START_DATE = datetime(2026, 1, 1)
PACKAGE_DIR = os.path.dirname(__file__)
DEFAULT_CONFIG_PATH = os.path.join(PACKAGE_DIR, "config", "profiles.yaml")
SCENARIOS = ("P01", "P02", "P03", "P04")


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_rise(day_index, onset_day, tau=5.0):
    """
    Smooth transition from 0 to 1 around onset_day.
    day_index is 1-based.
    """
    return sigmoid((day_index - onset_day) / tau)


def sigmoid_fall(day_index, offset_day, tau=5.0):
    """
    Smooth transition from 1 to 0 around offset_day.
    day_index is 1-based.
    """
    return 1.0 - sigmoid((day_index - offset_day) / tau)


def episode_severity(day_index, onset_day, offset_day, tau=5.0, max_severity=1.0):
    """
    Produces a smooth episode:
    baseline -> distress -> baseline.

    severity = 0 means baseline.
    severity = 1 means full distress.
    """
    rise = sigmoid_rise(day_index, onset_day, tau)
    fall = sigmoid_fall(day_index, offset_day, tau)
    return max_severity * min(rise, fall)

def monotone_sigmoid(day_index, onset_day, tau=6.0, max_severity=1.0):
    """
    Smooth monotone increase from 0 to max_severity.
    Unlike episode_severity(), this function does not include recovery.
    """
    return max_severity * sigmoid((day_index - onset_day) / tau)

def get_scenario_severity(scenario_id, day_index):
    """
    Returns severity in [0, 1] for a given day.
    Labels remain binary:
      severity < 0.5 -> baseline
      severity >= 0.5 -> distress
    """

    if scenario_id == "P01":
        # Single distress episode.
        # Days 60–90.
        severity = episode_severity(
            day_index,
            onset_day=60,
            offset_day=90,
            tau=2.5,
        )

    elif scenario_id == "P02":
        # Sustained distress period.
        # Days 45–180, no full recovery until late.
        severity = episode_severity(
            day_index,
            onset_day=45,
            offset_day=180,
            tau=7.0,
        )

    elif scenario_id == "P03":
        # Recurrent distress episodes.
        # Days 50–80 and 200–230.
        ep1 = episode_severity(
            day_index,
            onset_day=50,
            offset_day=80,
            tau=2.5,
        )
        ep2 = episode_severity(
            day_index,
            onset_day=200,
            offset_day=230,
            tau=2.5,
        )
        severity = max(ep1, ep2)

    elif scenario_id == "P04":
        # Monotone escalating distress severity.
        # The first sigmoid introduces moderate behavioural deterioration.
        # The second sigmoid adds a later escalation without recovery.
        moderate = monotone_sigmoid(
            day_index,
            onset_day=60,
            tau=6.0,
            max_severity=0.45,
        )

        severe = monotone_sigmoid(
            day_index,
            onset_day=120,
            tau=6.0,
            max_severity=0.55,
        )

        severity = moderate + severe

    else:
        raise ValueError("scenario_id must be one of: P01, P02, P03, P04")

    # Remove negligible sigmoid tails.
    # Very small values are mathematically caused by the sigmoid curve,
    # but clinically they should still be treated as clean baseline.
    if severity < 0.01:
        severity = 0.0

    # Safety clamp: keep severity within [0, 1].
    severity = min(max(severity, 0.0), 1.0)

    return severity

def interpolate_numeric_value(baseline_value, distress_value, severity):
    return baseline_value + severity * (distress_value - baseline_value)


def interpolate_transition_matrix(baseline_matrix, distress_matrix, severity):
    """
    Interpolates transition probabilities row by row.
    Keeps the same YAML structure:
      bedroom: [...]
      livingroom: [...]
      kitchen: [...]
      hallway: [...]
    """
    result = {}

    for room in baseline_matrix:
        base_row = baseline_matrix[room]
        dist_row = distress_matrix[room]

        row = [
            interpolate_numeric_value(b, d, severity)
            for b, d in zip(base_row, dist_row)
        ]

        # Normalize defensively to ensure row sum = 1.
        row_sum = sum(row)
        if row_sum == 0:
            raise ValueError(f"Invalid transition row for room {room}")

        result[room] = [v / row_sum for v in row]

    return result


def build_interpolated_profile(baseline_profile, distress_profile, severity):
    """
    Creates a temporary profile for a specific day using severity.
    severity = 0 -> baseline profile
    severity = 1 -> distress profile
    """

    profile = {}

    for key, baseline_value in baseline_profile.items():
        if key == "transition_matrix":
            profile[key] = interpolate_transition_matrix(
                baseline_profile["transition_matrix"],
                distress_profile["transition_matrix"],
                severity,
            )
            continue

        distress_value = distress_profile[key]

        if isinstance(baseline_value, (int, float)) and isinstance(distress_value, (int, float)):
            profile[key] = interpolate_numeric_value(
                baseline_value,
                distress_value,
                severity,
            )
        else:
            profile[key] = baseline_value

    return profile


def write_temp_config_for_day(profile, temp_config_path, config_path):
    """
    generate_one_day() currently expects a config file and a state.
    To avoid redesigning generate_raw_data.py, we write a temporary config
    with one generated 'scenario_day' profile.
    """
    import yaml

    temp_config = {
        "scenario_day": profile
    }
    
    main_cfg = load_profiles(config_path)
    if "feature_normalisation" in main_cfg:
        temp_config["feature_normalisation"] = main_cfg["feature_normalisation"]

    with open(temp_config_path, "w") as f:
        yaml.safe_dump(temp_config, f, sort_keys=False)
    

def generate_scenario(
    scenario_id,
    seed_override=0,
    n_days=N_DAYS,
    output_root="outputs",
    config_path=None,
    start_date=START_DATE,
):
    config_path = str(config_path or DEFAULT_CONFIG_PATH)
    output_root = os.fspath(output_root)
    profiles = load_profiles(config_path)

    scenario_seed_offsets = profiles.get("scenario_seed_offsets", {})
    if scenario_id not in scenario_seed_offsets:
        raise KeyError(f"Missing scenario seed offset for {scenario_id}")

    scenario_seed_offset = int(scenario_seed_offsets[scenario_id])

    baseline_profile = profiles["baseline"]
    distress_profile = profiles["distress"]
    scenario_folder = os.path.join(output_root, scenario_id)
    raw_folder = os.path.join(scenario_folder, "raw_days")
    daily_features_file = os.path.join(scenario_folder, "daily_features.csv")
    metadata_file = os.path.join(scenario_folder, "scenario_labels.csv")


    if os.path.exists(scenario_folder):
        shutil.rmtree(scenario_folder)

    os.makedirs(raw_folder, exist_ok=True)

    label_rows = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(n_days):
            day_index = i + 1
            current_date = start_date + timedelta(days=i)

            severity = get_scenario_severity(scenario_id, day_index)
            label = "distress" if severity >= 0.5 else "baseline"

            profile = build_interpolated_profile(
                baseline_profile,
                distress_profile,
                severity,
            )

            temp_config_path = os.path.join(temp_dir, f"{scenario_id}_day_{day_index}.yaml")
            write_temp_config_for_day(profile, temp_config_path, config_path)

            events = generate_one_day(
                current_date,
                state="scenario_day",
                day_index=i,
                config_path=temp_config_path,
                seed_override=seed_override,
                scenario_id=scenario_id,
                scenario_seed_offset=scenario_seed_offset,
            )

            file_name = f"{scenario_id}_day_{day_index:03d}.csv"
            file_path = os.path.join(raw_folder, file_name)
            save_day(events, file_path)

            label_rows.append((current_date.strftime("%Y-%m-%d"), label, round(severity, 4)))

    # Build daily feature matrix using the existing pipeline.
    # First build with one label, then overwrite labels using scenario_labels.csv later.
    build_daily_dataset(raw_folder, daily_features_file, label="temporary")

    # Overwrite label column according to scenario schedule.
    import pandas as pd

    df = pd.read_csv(daily_features_file)
    labels_df = pd.DataFrame(label_rows, columns=["date", "label", "severity"])

    df["date"] = labels_df["date"]
    df["label"] = labels_df["label"]
    df["severity"] = labels_df["severity"]

    df.to_csv(daily_features_file, index=False)
    labels_df.to_csv(metadata_file, index=False)

    print(f"Generated scenario {scenario_id}")
    print(f"Raw days: {raw_folder}")
    print(f"Daily features: {daily_features_file}")
    print(f"Labels: {metadata_file}")


def generate_all_scenarios(output_root="outputs", seed_override=0, n_days=N_DAYS):
    for scenario_id in SCENARIOS:
        generate_scenario(
            scenario_id,
            output_root=output_root,
            seed_override=seed_override,
            n_days=n_days,
        )


if __name__ == "__main__":
    generate_all_scenarios()
