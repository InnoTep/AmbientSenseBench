"""Independent, reproducible random streams for every part of the benchmark.

Each stream is derived with :class:`numpy.random.SeedSequence` from the run seed and
words that identify what is drawn: the purpose of the stream, the scenario, and for the
generator the behavioural state and the day. Streams therefore never share draws across
seeds, scenarios, days, or purposes, and each one is reproducible from its words alone.
A generated day's stream does not depend on the day's severity, so a scenario regenerated
with another severity trajectory or profile magnitude reuses the same draws day by day.
"""

from __future__ import annotations

import hashlib

import numpy as np

# First entropy word of every stream: which part of the benchmark draws from it.
GENERATOR_STREAM = 1  # one stream per generated day
RELEASE_STREAM = 2  # Laplace noise of the feature release, per seed and scenario
DETECTOR_STREAM = 3  # random state of randomised detectors (Isolation Forest)


def _word(value) -> int:
    """Map a seed, code, or identifier to a non-negative entropy word."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"seeds and identifiers must be non-negative, got {value}")
        return int(value)
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def seed_sequence(stream: int, *words) -> np.random.SeedSequence:
    return np.random.SeedSequence([stream, *(_word(word) for word in words)])


def day_seed(seed, scenario, state, day_index) -> int:
    """128-bit seed for ``random.seed`` of one generated day."""
    state_words = seed_sequence(GENERATOR_STREAM, seed, scenario, state, day_index)
    return int.from_bytes(state_words.generate_state(4).astype("<u4").tobytes(), "little")


def release_rng(seed, scenario) -> np.random.Generator:
    """Laplace-noise stream of one seed and scenario; every epsilon starts it afresh."""
    return np.random.default_rng(seed_sequence(RELEASE_STREAM, seed, scenario))


def detector_random_state(seed, scenario) -> int:
    """``random_state`` of Isolation Forest for one seed and scenario."""
    return int(seed_sequence(DETECTOR_STREAM, seed, scenario).generate_state(1)[0])
