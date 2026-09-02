import csv
import os
import shutil
import random
import math
from datetime import datetime, timedelta
import argparse
import yaml

PACKAGE_DIR = os.path.dirname(__file__)
DEFAULT_CONFIG_PATH = os.path.join(PACKAGE_DIR, "config", "profiles.yaml")

# Room order is fixed and shared with profiles.yaml transition_matrix.
ROOM_ORDER = ["bedroom", "livingroom", "kitchen", "hallway"]
PIR_SENSOR_BY_ROOM = {
    "bedroom": "bedroom_pir",
    "livingroom": "livingroom_pir",
    "kitchen": "kitchen_pir",
    "hallway": "hallway_pir",
}


def add_event(events, ts, sensor_id, state):
    events.append([ts, sensor_id, state])


def sample_poisson(lam):
    if lam <= 0:
        return 0

    l = math.exp(-lam)
    k = 0
    p = 1.0

    while p > l:
        k += 1
        p *= random.random()

    return k - 1


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def load_profiles(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_feature_normalisation(config):
    normalisation = config.get("feature_normalisation")

    if normalisation is None:
        raise ValueError("profiles.yaml is missing 'feature_normalisation' section")

    mobility_reference_events = normalisation["mobility_score"]["reference_events"]
    kitchen_reference_events = normalisation["kitchen_activity_score"]["reference_events"]

    return mobility_reference_events, kitchen_reference_events

def deterministic_seed(
    state,
    day_index,
    seed_override=0,
    scenario_id=None,
    scenario_seed_offset=0,
):
    base_seed = 0 if seed_override is None else int(seed_override)

    state_offsets = {
        "baseline": 1000,
        "distress": 2000,
        "scenario_day": 3000,
    }

    if state not in state_offsets:
        raise ValueError("state must be 'baseline', 'distress', or 'scenario_day'")

    if state == "scenario_day" and scenario_id is None:
        raise ValueError("scenario_id must be provided when state='scenario_day'")

    return (
        base_seed
        + state_offsets[state]
        + int(scenario_seed_offset)
        + day_index
    )

def get_state_profile(state, config_path=None):
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    profiles = load_profiles(config_path)

    if state not in profiles:
        raise ValueError("state must be 'baseline', 'distress', or 'scenario_day'")

    return profiles[state]


def get_transition_matrix(profile):
    """Return the Markov transition matrix as a list of rows indexed by ROOM_ORDER."""
    raw = profile.get("transition_matrix")
    if raw is None:
        raise ValueError("profile is missing 'transition_matrix'")

    matrix = []
    for room in ROOM_ORDER:
        row = raw[room]
        if len(row) != len(ROOM_ORDER):
            raise ValueError(f"transition_matrix row for '{room}' has wrong length")
        matrix.append(list(row))
    return matrix


def sample_next_room(current_room, transition_matrix):
    """Sample the next room from the row of the transition matrix."""
    row_index = ROOM_ORDER.index(current_room)
    weights = transition_matrix[row_index]
    return random.choices(ROOM_ORDER, weights=weights, k=1)[0]


def generate_wake_and_sleep(events, date, profile):
    wake_hour_float = random.gauss(profile["wake_mean"], profile["wake_std"])
    wake_hour_float = clamp(wake_hour_float, 5.0, 10.0)

    wake_hour = int(wake_hour_float)
    wake_minute = int((wake_hour_float - wake_hour) * 60)

    wake_time = datetime(date.year, date.month, date.day, wake_hour, wake_minute)

    sleep_hours = random.gauss(profile["sleep_mean"], profile["sleep_std"])
    sleep_hours = clamp(sleep_hours, 3.0, 12.0)

    sleep_onset = wake_time - timedelta(hours=sleep_hours)

    add_event(events, sleep_onset, "bedside_lamp", "INACTIVE")

    add_event(events, wake_time, "bedroom_pir", "ON")
    add_event(events, wake_time + timedelta(seconds=10), "bedroom_pir", "OFF")

    return wake_time, sleep_onset


def generate_meals(events, date, meal_count):
    meal_slots = [
        datetime(date.year, date.month, date.day, 8, random.randint(0, 40)),
        datetime(date.year, date.month, date.day, 13, random.randint(0, 40)),
        datetime(date.year, date.month, date.day, 19, random.randint(0, 40)),
    ]

    selected_slots = meal_slots[:meal_count]

    for t in selected_slots:
        add_event(events, t, "kitchen_pir", "ON")
        add_event(events, t + timedelta(minutes=2), "kitchen_pir", "OFF")

        add_event(events, t + timedelta(minutes=2), "fridge_door", "OPEN")
        add_event(events, t + timedelta(minutes=3), "fridge_door", "CLOSE")


def generate_extra_kitchen_activity(events, date, extra_count):
    for _ in range(extra_count):
        hour = random.choice([9, 10, 11, 15, 16, 17, 20])
        minute = random.randint(0, 59)
        t = datetime(date.year, date.month, date.day, hour, minute)

        add_event(events, t, "kitchen_pir", "ON")
        add_event(events, t + timedelta(minutes=1), "kitchen_pir", "OFF")


def generate_social(events, date, social_count):
    possible_hours = [10, 15, 18, 20]

    social_count = min(social_count, len(possible_hours))

    for i in range(social_count):
        hour = possible_hours[i]
        minute = random.randint(0, 40)
        t = datetime(date.year, date.month, date.day, hour, minute)

        add_event(events, t, "front_door", "OPEN")
        add_event(events, t + timedelta(seconds=5), "front_door", "CLOSE")


def generate_night_activity(events, date, night_count):
    pir_sensors = list(PIR_SENSOR_BY_ROOM.values())

    for _ in range(night_count):
        hour = random.randint(0, 4)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        t = datetime(date.year, date.month, date.day, hour, minute, second)
        sensor = random.choice(pir_sensors)

        add_event(events, t, sensor, "ON")
        add_event(events, t + timedelta(seconds=10), sensor, "OFF")


def generate_day_mobility(events, date, wake_time, mobility_blocks, transition_matrix):
    """Generate daytime mobility events that are consistent with the
    mobility extractor.

    kitchen_pir is not emitted here because kitchen-specific activity is
    represented separately by kitchen_activity_score. The Markov chain may
    still sample the kitchen as an intermediate room, but kitchen PIR events
    are not added as mobility events. This prevents kitchen activity from
    leaking into mobility_score.
    """

    current_time = wake_time + timedelta(minutes=20)
    current_room = "bedroom"

    emitted_mobility_events = 0
    attempts = 0
    max_attempts = mobility_blocks * 4 + 10

    while emitted_mobility_events < mobility_blocks and attempts < max_attempts:
        attempts += 1

        next_room = sample_next_room(current_room, transition_matrix)

        if next_room != "kitchen":
            sensor = PIR_SENSOR_BY_ROOM[next_room]

            add_event(events, current_time, sensor, "ON")
            add_event(events, current_time + timedelta(seconds=15), sensor, "OFF")

            emitted_mobility_events += 1

        gap_minutes = random.randint(30, 90)
        current_time += timedelta(minutes=gap_minutes)
        current_room = next_room

        if current_time.hour >= 22:
            break


def generate_evening_routine(events, date, profile):
    target_consistency = random.gauss(
        profile["evening_consistency_mean"],
        profile["evening_consistency_std"]
    )
    target_consistency = clamp(target_consistency, 0.0, 1.0)

    max_deviation_hours = 4.0
    deviation = (1.0 - target_consistency) * max_deviation_hours

    direction = random.choice([-1, 1])

    tv_off = datetime(date.year, date.month, date.day, 22, 0) + timedelta(
        hours=direction * deviation
    )

    lamp_off = datetime(date.year, date.month, date.day, 23, 0) + timedelta(
        hours=direction * deviation
    )

    add_event(events, tv_off - timedelta(hours=2), "tv_plug", "ON")
    add_event(events, tv_off, "tv_plug", "OFF")

    add_event(events, lamp_off, "evening_lamp", "INACTIVE")


def generate_one_day(
    date,
    state="baseline",
    day_index=0,
    config_path=None,
    seed_override=None,
    scenario_id=None,
    scenario_seed_offset=0,
):
    seed = deterministic_seed(
        state=state,
        day_index=day_index,
        seed_override=seed_override,
        scenario_id=scenario_id,
        scenario_seed_offset=scenario_seed_offset,
    )
    random.seed(seed)
    events = []
    
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config = load_profiles(config_path)
    profile = config[state]
    transition_matrix = get_transition_matrix(profile)

    mobility_reference_events, kitchen_reference_events = get_feature_normalisation(config)

    wake_time, _ = generate_wake_and_sleep(events, date, profile)
    generate_evening_routine(events, date, profile)

    meal_count = round(random.gauss(profile["meal_mean"], profile["meal_std"]))
    meal_count = clamp(meal_count, 0, 3)

    social_count = round(random.gauss(profile["social_mean"], profile["social_std"]))
    social_count = clamp(social_count, 0, 4)

    night_count = sample_poisson(profile["night_lambda"])
    night_count = clamp(night_count, 0, 12)

    target_mobility_score = random.gauss(
        profile["mobility_mean"], profile["mobility_std"]
    )
    target_mobility_score = clamp(target_mobility_score, 0.1, 2.0)
    mobility_blocks = round(target_mobility_score * mobility_reference_events)
    
    mobility_blocks = clamp(mobility_blocks, 0, 25)

    target_kitchen_score = random.gauss(
        profile["kitchen_mean"], profile["kitchen_std"]
    )
    target_kitchen_score = clamp(target_kitchen_score, 0.0, 2.0)
    target_kitchen_raw = round(target_kitchen_score * kitchen_reference_events)
    target_kitchen_raw = clamp(target_kitchen_raw, 0, 18)

    generate_meals(events, date, meal_count)

    current_kitchen_raw_from_meals = meal_count * 2
    extra_kitchen_needed = max(0, target_kitchen_raw - current_kitchen_raw_from_meals)
    generate_extra_kitchen_activity(events, date, extra_kitchen_needed)

    generate_social(events, date, social_count)
    generate_night_activity(events, date, night_count)
    generate_day_mobility(events, date, wake_time, mobility_blocks, transition_matrix)

    events.sort(key=lambda x: x[0])
    return events


def save_day(events, file_path):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "sensor_id", "state"])

        for event in events:
            writer.writerow([
                event[0].strftime("%Y-%m-%d %H:%M:%S"),
                event[1],
                event[2]
            ])


def generate_multiple_days(
    start_date,
    num_days,
    output_folder,
    state="baseline",
    config_path=None,
    seed_override=None
):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder)

    for i in range(num_days):
        current_date = start_date + timedelta(days=i)
        events = generate_one_day(
            current_date,
            state=state,
            day_index=i,
            config_path=config_path,
            seed_override=seed_override
        )

        file_name = f"{state}_day_{i + 1}.csv"
        file_path = os.path.join(output_folder, file_name)

        save_day(events, file_path)

    print(f"{num_days} days generated for state = {state}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--start-date", default="2026-01-01")

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    generate_multiple_days(
        start_date,
        args.days,
        "../data/raw_days_baseline",
        state="baseline",
        config_path=args.config,
        seed_override=args.seed
    )

    generate_multiple_days(
        start_date,
        args.days,
        "../data/raw_days_distress",
        state="distress",
        config_path=args.config,
        seed_override=args.seed
    )
