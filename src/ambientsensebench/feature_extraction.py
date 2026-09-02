# meal_count = fridge_door OPEN events
# social_proxy = front_door OPEN events
# night_activity = PIR ON events during the night-time window
# mobility_score = normalised daytime non-kitchen PIR activity
# kitchen_activity_score = normalised daytime kitchen PIR and fridge-door activity
import csv
from datetime import datetime
import os
import math
from collections import Counter
import yaml

PACKAGE_DIR = os.path.dirname(__file__)


def load_events(file_path):
    events = []

    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["timestamp"] = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            events.append(row)

    return events

def load_feature_normalisation():
    config_path = os.path.join(PACKAGE_DIR, "config", "profiles.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config["feature_normalisation"]

# wake time
def extract_wake_time(events):
    for e in events:
        if (
            e["sensor_id"] == "bedroom_pir"
            and e["state"] == "ON"
            and e["timestamp"].hour >= 5
        ):
            return e["timestamp"]
    return None


def extract_sleep_onset(events):
    wake_time = extract_wake_time(events)

    if wake_time is None:
        return None

    candidates = []

    for e in events:
        if (
            e["sensor_id"] == "bedside_lamp"
            and e["state"] == "INACTIVE"
            and e["timestamp"] < wake_time
        ):
            candidates.append(e["timestamp"])

    if not candidates:
        return None

    return max(candidates)

# sleep_duration
def extract_sleep_duration(events):
    sleep_onset = extract_sleep_onset(events)
    wake_time = extract_wake_time(events)

    if sleep_onset is None or wake_time is None:
        return None

    return wake_time - sleep_onset


def duration_to_hours(duration):
    if duration is None:
        return None
    return duration.total_seconds() / 3600

#meal_count
def extract_meal_count(events):
    count = 0

    for e in events:
        if e["sensor_id"] == "fridge_door" and e["state"] == "OPEN":
            count += 1

    return count
#social_proxy
def extract_social_proxy(events):
    count = 0

    for e in events:
        if e["sensor_id"] == "front_door" and e["state"] == "OPEN":
            count += 1

    return count
#night_activity
def extract_night_activity(events):
    count = 0

    for e in events:
        if (
            "pir" in e["sensor_id"]   
            and e["state"] == "ON"
            and 0 <= e["timestamp"].hour < 5
        ):
            count += 1

    return count

#mobility_score
def extract_mobility_raw(events):
    wake_time = extract_wake_time(events)

    if wake_time is None:
        return 0

    pir_sensors = {
        "bedroom_pir",
        "livingroom_pir",
        "hallway_pir"
    }

    

    count = 0
    skipped_wake_event = False

    for e in events:
        ts = e["timestamp"]

        if e["sensor_id"] not in pir_sensors:
            continue

        if e["state"] != "ON":
            continue

        if not (wake_time <= ts and ts.hour < 22):
            continue

        if e["sensor_id"] == "bedroom_pir" and ts == wake_time and not skipped_wake_event:
            skipped_wake_event = True
            continue

        count += 1

    return count


def normalize_mobility_score(mobility_raw, normalisation_config=None):
    if normalisation_config is None:
        normalisation_config = load_feature_normalisation()

    reference = normalisation_config["mobility_score"]["reference_events"]
    return mobility_raw / reference

#kitchen_activity
def extract_kitchen_activity_raw(events):
    wake_time = extract_wake_time(events)

    if wake_time is None:
        return 0

    count = 0

    for e in events:
        ts = e["timestamp"]

        if not (wake_time <= ts and ts.hour < 22):
            continue

        if e["sensor_id"] == "kitchen_pir" and e["state"] == "ON":
            count += 1

        if e["sensor_id"] == "fridge_door" and e["state"] == "OPEN":
            count += 1

    return count

def normalize_kitchen_activity(raw_value, normalisation_config=None):
    if normalisation_config is None:
        normalisation_config = load_feature_normalisation()

    reference = normalisation_config["kitchen_activity_score"]["reference_events"]
    return raw_value / reference

def sensor_to_room(sensor_id):
    if sensor_id.startswith("bedroom"):
        return "bedroom"
    if sensor_id.startswith("livingroom"):
        return "livingroom"
    if sensor_id.startswith("kitchen"):
        return "kitchen"
    if sensor_id.startswith("hallway"):
        return "hallway"
    return None


def extract_room_transition_entropy(events):
    wake_time = extract_wake_time(events)

    if wake_time is None:
        return 0.0

    rooms = []

    for e in sorted(events, key=lambda x: x["timestamp"]):
        ts = e["timestamp"]

        if not (wake_time <= ts and ts.hour < 22):
            continue

        if e["state"] != "ON":
            continue

        room = sensor_to_room(e["sensor_id"])

        if room is not None:
            rooms.append(room)

    transitions = []

    for i in range(1, len(rooms)):
        if rooms[i] != rooms[i - 1]:
            transitions.append((rooms[i - 1], rooms[i]))

    if len(transitions) == 0:
        return 0.0

    counts = Counter(transitions)
    total = sum(counts.values())

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    normalized_entropy = min(entropy / 2.65, 1.0)

    return normalized_entropy

def extract_evening_routine_consistency(events, reference_tv_hour=22.0, reference_lamp_hour=23.0):
    tv_off = None
    lamp_off = None

    for e in sorted(events, key=lambda x: x["timestamp"]):
        if e["sensor_id"] == "tv_plug" and e["state"] == "OFF":
            tv_off = e["timestamp"]

        if e["sensor_id"] == "evening_lamp" and e["state"] == "INACTIVE":
            lamp_off = e["timestamp"]

    if tv_off is None or lamp_off is None:
        return None

    tv_hour = tv_off.hour + tv_off.minute / 60 + tv_off.second / 3600
    lamp_hour = lamp_off.hour + lamp_off.minute / 60 + lamp_off.second / 3600

    tv_deviation = abs(tv_hour - reference_tv_hour)
    lamp_deviation = abs(lamp_hour - reference_lamp_hour)

    if tv_deviation > 12:
        tv_deviation = 24 - tv_deviation

    if lamp_deviation > 12:
        lamp_deviation = 24 - lamp_deviation

    average_deviation = (tv_deviation + lamp_deviation) / 2

    consistency = max(0.0, 1.0 - average_deviation / 4.0)

    return consistency

def extract_all_features(events):
    wake_time = extract_wake_time(events)
    sleep_onset = extract_sleep_onset(events)
    sleep_duration = extract_sleep_duration(events)
    sleep_hours = duration_to_hours(sleep_duration)

    meal_count = extract_meal_count(events)
    social_proxy = extract_social_proxy(events)
    night_activity = extract_night_activity(events)

    normalisation_config = load_feature_normalisation()

    mobility_raw = extract_mobility_raw(events)
    mobility_score = normalize_mobility_score(mobility_raw, normalisation_config)

    kitchen_raw = extract_kitchen_activity_raw(events)
    kitchen_score = normalize_kitchen_activity(kitchen_raw, normalisation_config)

    room_transition_entropy = extract_room_transition_entropy(events)
    evening_routine_consistency = extract_evening_routine_consistency(events)

    return {
        "wake_time": wake_time,
        "sleep_onset": sleep_onset,
        "sleep_duration": sleep_duration,
        "sleep_hours": sleep_hours,
        "meal_count": meal_count,
        "social_proxy": social_proxy,
        "night_activity": night_activity,
        "mobility_raw": mobility_raw,
        "mobility_score": mobility_score,
        "kitchen_raw": kitchen_raw,
        "kitchen_score": kitchen_score,
        "room_transition_entropy": room_transition_entropy,
        "evening_routine_consistency": evening_routine_consistency,
    }


if __name__ == "__main__":
    input_file = os.path.join(PACKAGE_DIR, "data", "raw_events.csv")
    events = load_events(input_file)
    features = extract_all_features(events)

    print("Wake time:", features["wake_time"])
    print("Sleep onset:", features["sleep_onset"])
    print("Sleep duration:", features["sleep_duration"])
    print("Sleep duration (hours):", features["sleep_hours"])
    print("Meal count:", features["meal_count"])
    print("Social proxy:", features["social_proxy"])
    print("Night activity:", features["night_activity"])
    print("Mobility raw:", features["mobility_raw"])
    print("Mobility score:", features["mobility_score"])
    print("Kitchen activity raw:", features["kitchen_raw"])
    print("Kitchen activity score:", features["kitchen_score"])
