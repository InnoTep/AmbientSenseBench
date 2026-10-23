"""Seeds, scenarios, and days must draw from independent random streams.

Two independent seeds can share a generated day only by chance, and a day carries
dozens of events at minute or second resolution, so chance allows no shared day at all.
A generator that derives day seeds additively (seed + day) fails here: seed 1 then
replays seed 0 shifted by one day.
"""

from datetime import datetime, timedelta

import numpy as np

from ambientsensebench.generate_raw_data import (
    deterministic_seed,
    generate_one_day,
    load_profiles,
)
from ambientsensebench.generate_scenarios import DEFAULT_CONFIG_PATH
from ambientsensebench.seeding import detector_random_state, release_rng

START = datetime(2026, 1, 1)
N_DAYS = 365


def _day_signatures(seed):
    """Every generated day of one seed, as its events relative to the day's midnight."""
    signatures = []
    for i in range(N_DAYS):
        day = START + timedelta(days=i)
        events = generate_one_day(day, state="baseline", day_index=i, seed_override=seed)
        signatures.append(tuple((ts - day, sensor, state) for ts, sensor, state in events))
    return signatures


def test_seed_0_and_seed_1_share_no_day():
    seed_0, seed_1 = _day_signatures(0), _day_signatures(1)
    assert len(set(seed_0)) == N_DAYS
    shared = set(seed_0) & set(seed_1)
    assert not shared, f"seeds 0 and 1 share {len(shared)} of {N_DAYS} days"


def test_day_streams_are_distinct_across_seeds_scenarios_and_days():
    offsets = load_profiles(DEFAULT_CONFIG_PATH)["scenario_seed_offsets"]
    day_seeds = {
        deterministic_seed("scenario_day", day, seed, scenario, offset)
        for seed in range(25)
        for scenario, offset in offsets.items()
        for day in range(N_DAYS)
    }
    assert len(day_seeds) == 25 * len(offsets) * N_DAYS


def test_release_noise_and_detector_streams_are_distinct():
    units = [(seed, scenario) for seed in (0, 1) for scenario in ("P01", "P02")]
    draws = [release_rng(seed, scenario).laplace(size=N_DAYS * 9) for seed, scenario in units]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert np.intersect1d(draws[i], draws[j]).size == 0, (units[i], units[j])
    states = {detector_random_state(seed, f"P0{k}") for seed in range(25) for k in range(1, 5)}
    assert len(states) == 100
