"""Parametric ("custom") scenario generation for the AmbientSenseBench app.

This module generalises :func:`generate_scenarios.generate_scenario` so that a
user can define an arbitrary routine-change trajectory from a list of episodes
instead of choosing one of the four fixed reference scenarios. It reuses the
same raw-event generator, the same daily-feature extractor, and the same
baseline/distress behavioural profiles, so the generated data is directly
comparable with the reference benchmark.

An episode is a dictionary with the keys::

    {
        "mode": "episode" | "monotone",   # transient episode or monotone drift
        "onset": int,                       # 1-based day the change begins
        "offset": int,                      # 1-based day the change ends (episode only)
        "tau": float,                       # transition smoothness (days)
        "max_severity": float,              # peak severity in [0, 1]
    }

The severity on a given day is the maximum contribution across all episodes,
matching the semantics of the built-in recurrent scenario. Severity in
[0, 1] linearly interpolates the euthymic-baseline and distress behavioural
profiles; a binary label (``distress`` when severity >= 0.5) is retained only
for retrospective evaluation.
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .apply_wp7_dp_noise import apply_laplace_noise
from .build_dataset import build_daily_dataset
from .generate_raw_data import generate_one_day, load_profiles, save_day
from .generate_scenarios import (
    DEFAULT_CONFIG_PATH,
    START_DATE,
    build_interpolated_profile,
    episode_severity,
    monotone_sigmoid,
    write_temp_config_for_day,
)
from .wp6_evaluation import (
    FEATURE_COLUMNS,
    compute_fixed_threshold_from_training_scores,
    convert_wake_time_to_hour,
)

# The baseline-only training window, matching the reference benchmark protocol.
TRAIN_DAYS = 40

# A dedicated seed offset so custom runs do not collide with the reference
# scenario seed streams (which use offsets 101-404).
CUSTOM_SEED_OFFSET = 500
CUSTOM_SCENARIO_ID = "CUSTOM"

# Sensor vocabulary exposed to the interface. These are the sensors the raw
# generator can emit; the schema is vendor-neutral.
SENSOR_VOCABULARY = [
    {"id": "bedroom_pir", "label": "Bedroom motion (PIR)", "kind": "motion"},
    {"id": "livingroom_pir", "label": "Living-room motion (PIR)", "kind": "motion"},
    {"id": "kitchen_pir", "label": "Kitchen motion (PIR)", "kind": "motion"},
    {"id": "hallway_pir", "label": "Hallway motion (PIR)", "kind": "motion"},
    {"id": "front_door", "label": "Front door (contact)", "kind": "contact"},
    {"id": "fridge_door", "label": "Fridge door (contact)", "kind": "contact"},
    {"id": "tv_plug", "label": "TV appliance (plug)", "kind": "appliance"},
    {"id": "evening_lamp", "label": "Evening lamp", "kind": "appliance"},
    {"id": "bedside_lamp", "label": "Bedside lamp", "kind": "appliance"},
]

# Preset trajectories, expressed as episode lists, that reproduce the shapes of
# the four reference scenarios so users can start from a known baseline.
PRESETS = {
    "P01": {
        "label": "Short disruption followed by recovery",
        "episodes": [
            {"mode": "episode", "onset": 60, "offset": 90, "tau": 2.5, "max_severity": 1.0},
        ],
    },
    "P02": {
        "label": "Sustained disruption with late recovery",
        "episodes": [
            {"mode": "episode", "onset": 45, "offset": 180, "tau": 7.0, "max_severity": 1.0},
        ],
    },
    "P03": {
        "label": "Two recurrent disruptions separated by recovery",
        "episodes": [
            {"mode": "episode", "onset": 50, "offset": 80, "tau": 2.5, "max_severity": 1.0},
            {"mode": "episode", "onset": 200, "offset": 230, "tau": 2.5, "max_severity": 1.0},
        ],
    },
    "P04": {
        "label": "Progressive, monotone change in routine",
        "episodes": [
            {"mode": "monotone", "onset": 100, "offset": 0, "tau": 6.0, "max_severity": 1.0},
        ],
    },
}


def _coerce_episode(raw: dict, n_days: int) -> dict:
    """Validate and normalise a single episode definition."""
    mode = str(raw.get("mode", "episode"))
    if mode not in ("episode", "monotone"):
        raise ValueError("episode mode must be 'episode' or 'monotone'")

    onset = int(raw.get("onset", 1))
    if not 1 <= onset <= n_days:
        raise ValueError(f"episode onset {onset} is outside 1..{n_days}")

    tau = float(raw.get("tau", 5.0))
    if not 0.1 <= tau <= 60.0:
        raise ValueError("episode tau must be within 0.1..60")

    max_severity = float(raw.get("max_severity", 1.0))
    if not 0.0 < max_severity <= 1.0:
        raise ValueError("episode max_severity must be within (0, 1]")

    offset = int(raw.get("offset", onset))
    if mode == "episode":
        if not 1 <= offset <= n_days:
            raise ValueError(f"episode offset {offset} is outside 1..{n_days}")
        if offset <= onset:
            raise ValueError("episode offset must be greater than onset")

    return {
        "mode": mode,
        "onset": onset,
        "offset": offset,
        "tau": tau,
        "max_severity": max_severity,
    }


def normalise_episodes(episodes: Iterable[dict], n_days: int) -> list[dict]:
    episodes = [_coerce_episode(raw, n_days) for raw in episodes]
    if not episodes:
        raise ValueError("at least one episode is required")
    if len(episodes) > 6:
        raise ValueError("at most six episodes are supported")
    return episodes


def severity_for_day(day_index: int, episodes: list[dict]) -> float:
    """Severity in [0, 1] for a 1-based day, as the max across episodes."""
    contributions = []
    for episode in episodes:
        if episode["mode"] == "monotone":
            contributions.append(
                monotone_sigmoid(
                    day_index,
                    onset_day=episode["onset"],
                    tau=episode["tau"],
                    max_severity=episode["max_severity"],
                )
            )
        else:
            contributions.append(
                episode_severity(
                    day_index,
                    onset_day=episode["onset"],
                    offset_day=episode["offset"],
                    tau=episode["tau"],
                    max_severity=episode["max_severity"],
                )
            )
    severity = max(contributions) if contributions else 0.0
    if severity < 0.01:
        severity = 0.0
    return min(max(severity, 0.0), 1.0)


def score_isolation_forest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Baseline-trained Isolation Forest scoring for an arbitrary-length run.

    Generalises the explorer scorer, which is fixed to 365-day runs, to any
    trajectory with at least ``TRAIN_DAYS + 1`` days. The first ``TRAIN_DAYS``
    days form the training window and must all be baseline.
    """
    df = df.copy()
    df["wake_time"] = df["wake_time"].apply(convert_wake_time_to_hour)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")

    if len(df) <= TRAIN_DAYS:
        raise ValueError(f"a run needs more than {TRAIN_DAYS} days")

    train_df = df.iloc[:TRAIN_DAYS]
    test_df = df.iloc[TRAIN_DAYS:]

    if not (train_df["label"] == "baseline").all():
        raise ValueError(
            f"episodes must not raise severity within the first {TRAIN_DAYS} "
            "(baseline training) days"
        )

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_df[FEATURE_COLUMNS].to_numpy())
    test_data = scaler.transform(test_df[FEATURE_COLUMNS].to_numpy())

    model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    model.fit(train_data)

    train_scores = -model.decision_function(train_data)
    test_scores = -model.decision_function(test_data)
    threshold = compute_fixed_threshold_from_training_scores(train_scores)
    test_alerts = test_scores >= threshold

    scored = df.copy()
    scored["anomaly_score"] = np.nan
    scored["alarm"] = False
    scored.loc[test_df.index, "anomaly_score"] = test_scores
    scored.loc[test_df.index, "alarm"] = test_alerts

    labels = test_df["label"].to_numpy()
    alert_labels = labels[test_alerts]
    metrics = {
        "threshold": round(float(threshold), 5),
        "alerts": int(test_alerts.sum()),
        "alerts_on_change_days": int((alert_labels == "distress").sum()),
        "alerts_on_baseline_days": int((alert_labels == "baseline").sum()),
        "training_days": TRAIN_DAYS,
    }
    return scored, metrics


def _concatenate_raw_events(
    raw_folder: Path,
    events_file: Path,
    active_sensors: list[str] | None = None,
) -> int:
    """Combine per-day raw event CSVs into one events.csv. Returns row count.

    When ``active_sensors`` is given, only events from those sensors are
    exported. This affects the raw event stream only; the daily-feature
    extraction and detector preview always use the full instrumentation.
    """
    allowed = set(active_sensors) if active_sensors is not None else None
    total = 0
    with open(events_file, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["timestamp", "sensor_id", "state"])
        for day_file in sorted(raw_folder.glob("*.csv")):
            with open(day_file, newline="") as handle:
                reader = csv.reader(handle)
                next(reader, None)  # skip header
                for row in reader:
                    if allowed is not None and row[1] not in allowed:
                        continue
                    writer.writerow(row)
                    total += 1
    return total


def generate_custom_scenario(
    job_dir: Path,
    n_days: int,
    seed: int,
    episodes: list[dict],
    *,
    active_sensors: list[str] | None = None,
    dp_epsilon: float | None = None,
    config_path: str | None = None,
    start_date: datetime = START_DATE,
) -> dict:
    """Generate a custom scenario and return an explorer-style preview payload.

    Writes ``raw_days/``, ``events.csv``, ``daily_features.csv`` and
    ``scenario_labels.csv`` inside ``job_dir``. When ``dp_epsilon`` is given, a
    differentially private feature file ``daily_features_dp.csv`` is also
    written and used for the preview scoring.
    """
    config_path = str(config_path or DEFAULT_CONFIG_PATH)
    job_dir = Path(job_dir)
    if job_dir.exists():
        shutil.rmtree(job_dir)
    raw_folder = job_dir / "raw_days"
    raw_folder.mkdir(parents=True, exist_ok=True)

    profiles = load_profiles(config_path)
    baseline_profile = profiles["baseline"]
    distress_profile = profiles["distress"]

    daily_features_file = job_dir / "daily_features.csv"
    labels_file = job_dir / "scenario_labels.csv"
    events_file = job_dir / "events.csv"

    started = time.perf_counter()
    label_rows = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(n_days):
            day_index = i + 1
            current_date = start_date + timedelta(days=i)

            severity = severity_for_day(day_index, episodes)
            label = "distress" if severity >= 0.5 else "baseline"

            profile = build_interpolated_profile(
                baseline_profile, distress_profile, severity
            )
            temp_config_path = os.path.join(temp_dir, f"custom_day_{day_index}.yaml")
            write_temp_config_for_day(profile, temp_config_path, config_path)

            events = generate_one_day(
                current_date,
                state="scenario_day",
                day_index=i,
                config_path=temp_config_path,
                seed_override=seed,
                scenario_id=CUSTOM_SCENARIO_ID,
                scenario_seed_offset=CUSTOM_SEED_OFFSET,
            )

            save_day(events, str(raw_folder / f"custom_day_{day_index:03d}.csv"))
            label_rows.append(
                (current_date.strftime("%Y-%m-%d"), label, round(severity, 4))
            )

    build_daily_dataset(str(raw_folder), str(daily_features_file), label="temporary")

    features = pd.read_csv(daily_features_file)
    labels_df = pd.DataFrame(label_rows, columns=["date", "label", "severity"])
    features["date"] = labels_df["date"]
    features["label"] = labels_df["label"]
    features["severity"] = labels_df["severity"]
    features.to_csv(daily_features_file, index=False)
    labels_df.to_csv(labels_file, index=False)

    event_count = _concatenate_raw_events(raw_folder, events_file, active_sensors)
    generation_seconds = round(time.perf_counter() - started, 3)

    scoring_frame = features
    dp_applied = False
    if dp_epsilon is not None and dp_epsilon > 0:
        rng = np.random.default_rng(int(seed) + 9000)
        noisy = apply_laplace_noise(features, epsilon=float(dp_epsilon), rng=rng)
        noisy["date"] = labels_df["date"]
        noisy["label"] = labels_df["label"]
        noisy["severity"] = labels_df["severity"]
        noisy.to_csv(job_dir / "daily_features_dp.csv", index=False)
        scoring_frame = noisy
        dp_applied = True

    scored, metrics = score_isolation_forest(scoring_frame)
    selected = ["date", "label", "severity", *FEATURE_COLUMNS, "anomaly_score", "alarm"]
    records = [
        _normalise_record(record)
        for record in scored[selected].to_dict(orient="records")
    ]

    distress_days = int((labels_df["label"] == "distress").sum())
    downloads = {
        "features": "daily_features.csv",
        "events": "events.csv",
        "labels": "scenario_labels.csv",
    }
    if dp_applied:
        downloads["features_dp"] = "daily_features_dp.csv"

    return {
        "n_days": n_days,
        "seed": int(seed),
        "episodes": episodes,
        "features": FEATURE_COLUMNS,
        "records": records,
        "metrics": metrics,
        "distress_days": distress_days,
        "event_count": event_count,
        "generation_seconds": generation_seconds,
        "dp_epsilon": float(dp_epsilon) if dp_applied else None,
        "active_sensors": active_sensors,
        "downloads": downloads,
    }


def _normalise_record(record: dict) -> dict:
    normalised: dict = {}
    for key, value in record.items():
        if isinstance(value, (bool, np.bool_)):
            normalised[key] = bool(value)
        elif pd.isna(value):
            normalised[key] = None
        elif isinstance(value, (np.integer, int)):
            normalised[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            normalised[key] = round(float(value), 5)
        else:
            normalised[key] = value
    return normalised
