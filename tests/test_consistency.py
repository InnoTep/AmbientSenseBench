import random
from statistics import mean, stdev
from datetime import datetime, timedelta

import pytest

from ambientsensebench.feature_extraction import extract_all_features, load_events
from ambientsensebench.generate_raw_data import (
    generate_one_day,
    get_state_profile,
    save_day,
)


N_DAYS = 1000
SEED = 42

FEATURE_TARGET_MAP = {
    "wake_hour": "wake_mean",
    "sleep_hours": "sleep_mean",
    "meal_count": "meal_mean",
    "social_proxy": "social_mean",
    "night_activity": "night_lambda",
    "mobility_score": "mobility_mean",
    "kitchen_score": "kitchen_mean",
    "room_transition_entropy": "transition_entropy_mean",
    "evening_routine_consistency": "evening_consistency_mean",
}


def time_to_decimal_hour(dt):
    if dt is None:
        return None
    return dt.hour + dt.minute / 60 + dt.second / 3600


def extract_test_features(events):
    features = extract_all_features(events)

    return {
        "wake_hour": time_to_decimal_hour(features["wake_time"]),
        "sleep_hours": features["sleep_hours"],
        "meal_count": features["meal_count"],
        "social_proxy": features["social_proxy"],
        "night_activity": features["night_activity"],
        "mobility_score": features["mobility_score"],
        "kitchen_score": features["kitchen_score"],
        "room_transition_entropy": features["room_transition_entropy"],
        "evening_routine_consistency": features["evening_routine_consistency"],
    }


@pytest.mark.parametrize("state", ["baseline", "distress"])
def test_generator_extractor_consistency(tmp_path, state):
    random.seed(SEED)

    profile = get_state_profile(state)
    observed = {feature: [] for feature in FEATURE_TARGET_MAP}

    start_date = datetime(2026, 1, 1)

    for i in range(N_DAYS):
        current_date = start_date + timedelta(days=i)

        events = generate_one_day(
            current_date,
            state=state,
            day_index=i,
            seed_override=42
         )

        file_path = tmp_path / f"{state}_day_{i + 1}.csv"
        save_day(events, file_path)

        loaded_events = load_events(file_path)
        extracted = extract_test_features(loaded_events)

        for feature, value in extracted.items():
            assert value is not None, f"{state}: {feature} returned None"
            observed[feature].append(value)

    diagnostics = {}

    for feature, target_key in FEATURE_TARGET_MAP.items():
        observed_mean = mean(observed[feature])
        observed_std = stdev(observed[feature])
        target_mean = profile[target_key]
        allowed_delta = 0.10 * abs(target_mean)
        delta = observed_mean - target_mean

        diagnostics[feature] = {
            "observed_mean": round(observed_mean, 4),
            "observed_std": round(observed_std, 4),
            "target_mean": round(target_mean, 4),
            "delta": round(delta, 4),
            "allowed_delta": round(allowed_delta, 4),
        }

        assert abs(delta) < allowed_delta, (
            f"\nState: {state}\n"
            f"Feature: {feature}\n"
            f"Observed mean: {observed_mean:.4f}\n"
            f"Observed std: {observed_std:.4f}\n"
            f"Target mean: {target_mean:.4f}\n"
            f"Delta: {delta:.4f}\n"
            f"Allowed delta: {allowed_delta:.4f}\n"
            f"All diagnostics: {diagnostics}"
        )
