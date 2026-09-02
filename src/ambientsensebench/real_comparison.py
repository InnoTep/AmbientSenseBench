"""External plausibility check: compare the synthetic baseline against a real
single-resident smart home (CASAS Aruba).

The Aruba event log uses non-visual motion and door sensors. We map its locations
to the benchmark room/door vocabulary and recompute features with the same
extractor used elsewhere, so definitions are identical. Only sensor-density-
invariant features (notably morning wake time) are meaningfully comparable across
homes; count-based features are confounded by differing sensor density, home
layout, and the generator-specific feature normalisation.

Usage:
    python -m ambientsensebench.real_comparison \
        --aruba datasets/CASAS_Aruba/data/aruba.csv \
        --scenarios outputs/benchmark/scenarios --output outputs/paper
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy import stats

from .feature_extraction import (
    extract_night_activity,
    extract_room_transition_entropy,
    extract_social_proxy,
    extract_wake_time,
)
from .wp6_evaluation import convert_wake_time_to_hour

# Aruba location -> benchmark sensor id (non-visual motion + outside door only).
LOCATION_MAP = {
    "Bedroom": "bedroom_pir",
    "LivingRoom": "livingroom_pir",
    "LoungeChair": "livingroom_pir",
    "Kitchen": "kitchen_pir",
    "DiningRoom": "kitchen_pir",
    "Bathroom": "hallway_pir",
    "WorkArea": "hallway_pir",
    "GuestRoom": "hallway_pir",
    "OtherRoom": "hallway_pir",
}


def aruba_daily_features(aruba_path: str) -> pd.DataFrame:
    by_day: dict[str, list] = defaultdict(list)
    with open(aruba_path) as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            date, tm, loc, state = row[0], row[1], row[2], row[3]
            try:
                ts = datetime.strptime(date + " " + tm.split(".")[0], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if loc == "OutsideDoor":
                sensor_id = "front_door"
            else:
                sensor_id = LOCATION_MAP.get(loc)
                if sensor_id is None:
                    continue
            by_day[date].append({"timestamp": ts, "sensor_id": sensor_id, "state": state})

    rows = []
    for date, events in sorted(by_day.items()):
        wake = extract_wake_time(events)
        if wake is None:
            continue
        rows.append(
            {
                "date": date,
                "wake_hour": wake.hour + wake.minute / 60,
                "room_transition_entropy": extract_room_transition_entropy(events),
                "social_proxy": extract_social_proxy(events),
                "night_activity": extract_night_activity(events),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.iloc[1:-1].reset_index(drop=True)  # drop partial first/last day


def synthetic_baseline(scenario_root: str) -> pd.DataFrame:
    frames = [
        pd.read_csv(p)
        for p in glob.glob(f"{scenario_root}/seed-*/P0*/daily_features.csv")
    ]
    syn = pd.concat(frames, ignore_index=True)
    syn = syn[syn.label == "baseline"].copy()
    syn["wake_hour"] = syn["wake_time"].apply(convert_wake_time_to_hour)
    return syn


def compare(aruba: pd.DataFrame, syn: pd.DataFrame) -> pd.DataFrame:
    ks = stats.ks_2samp(syn.wake_hour, aruba.wake_hour)
    rows = [
        {
            "feature": "wake_hour",
            "comparable": "yes (density-invariant)",
            "synthetic_mean": round(syn.wake_hour.mean(), 3),
            "synthetic_std": round(syn.wake_hour.std(), 3),
            "aruba_mean": round(aruba.wake_hour.mean(), 3),
            "aruba_std": round(aruba.wake_hour.std(), 3),
            "ks_D": round(ks.statistic, 3),
            "ks_p": ks.pvalue,
        },
        {
            "feature": "room_transition_entropy",
            "comparable": "partial (aruba saturates cap)",
            "synthetic_mean": round(syn.room_transition_entropy.mean(), 3),
            "synthetic_std": round(syn.room_transition_entropy.std(), 3),
            "aruba_mean": round(aruba.room_transition_entropy.mean(), 3),
            "aruba_std": round(aruba.room_transition_entropy.std(), 3),
            "ks_D": None,
            "ks_p": None,
        },
        {
            "feature": "social_proxy",
            "comparable": "no (density/layout)",
            "synthetic_mean": round(syn.social_proxy.mean(), 3),
            "synthetic_std": round(syn.social_proxy.std(), 3),
            "aruba_mean": round(aruba.social_proxy.mean(), 3),
            "aruba_std": round(aruba.social_proxy.std(), 3),
            "ks_D": None,
            "ks_p": None,
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic-vs-real plausibility check.")
    parser.add_argument("--aruba", default="datasets/CASAS_Aruba/data/aruba.csv")
    parser.add_argument("--scenarios", default="outputs/benchmark/scenarios")
    parser.add_argument("--output", type=Path, default=Path("outputs/paper"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    aruba = aruba_daily_features(args.aruba)
    aruba.to_csv(args.output / "aruba_daily.csv", index=False)
    syn = synthetic_baseline(args.scenarios)
    summary = compare(aruba, syn)
    summary.to_csv(args.output / "real_comparison_summary.csv", index=False)
    print(f"aruba usable days: {len(aruba)} | synthetic baseline days: {len(syn)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
