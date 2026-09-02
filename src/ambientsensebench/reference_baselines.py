"""Trivial statistical reference baselines for the detector benchmark.

Three textbook detectors that any benchmark comparison should include:

* ``zscore_max``  -- memoryless: the day's largest absolute standardised
  feature value.  The simplest possible per-day scorer.
* ``ewma``        -- sequential: per-feature exponentially weighted moving
  average control chart on the standardised features (lambda = 0.2), scored
  as the largest absolute EWMA statistic normalised by its asymptotic
  standard deviation.  Runs strictly online over the chronological
  train-plus-test sequence, like the other sequential detectors.
* ``hotelling_t2`` -- memoryless multivariate: Hotelling's T-squared with a
  Ledoit--Wolf shrinkage covariance estimated on the training window (40
  days x 9 features is too short for a stable sample covariance).

All three consume the same standardised arrays as the core detectors and
feed the same 95th-percentile training threshold rule.  They are evaluated
as reference points and are not part of the benchmark's core detector set.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

BASELINE_LABELS = {
    "zscore_max": "Max |z|",
    "ewma": "EWMA chart",
    "hotelling_t2": "Hotelling T2",
}
# EWMA carries state across days; the other two score days independently.
SEQUENTIAL_BASELINES = ("ewma",)

EWMA_LAMBDA = 0.2


def zscore_max_scores(train_data: np.ndarray, test_data: np.ndarray):
    train_scores = np.abs(np.asarray(train_data, dtype=float)).max(axis=1)
    test_scores = np.abs(np.asarray(test_data, dtype=float)).max(axis=1)
    return train_scores, test_scores


def ewma_scores(train_data: np.ndarray, test_data: np.ndarray):
    """Two-sided EWMA chart statistic, maximised over features.

    The recursion runs once over the chronologically ordered train-plus-test
    sequence.  Scores are |EWMA| / sigma_infinity with
    sigma_infinity = sqrt(lambda / (2 - lambda)) for unit-variance inputs,
    so the score is on an approximate standard-deviation scale.
    """
    lam = EWMA_LAMBDA
    full = np.vstack([np.asarray(train_data, float), np.asarray(test_data, float)])
    sigma_inf = np.sqrt(lam / (2.0 - lam))
    ewma = np.zeros(full.shape[1])
    scores = np.empty(len(full))
    for t, row in enumerate(full):
        ewma = (1.0 - lam) * ewma + lam * row
        scores[t] = float(np.max(np.abs(ewma)) / sigma_inf)
    n_train = len(train_data)
    return scores[:n_train], scores[n_train:]


def hotelling_t2_scores(train_data: np.ndarray, test_data: np.ndarray):
    train = np.asarray(train_data, dtype=float)
    test = np.asarray(test_data, dtype=float)
    lw = LedoitWolf().fit(train)
    mean = train.mean(axis=0)
    precision = lw.precision_

    def t2(x):
        d = x - mean
        return np.einsum("ij,jk,ik->i", d, precision, d)

    return t2(train), t2(test)


def score_baseline(model_id: str, train_data: np.ndarray, test_data: np.ndarray):
    if model_id == "zscore_max":
        return zscore_max_scores(train_data, test_data)
    if model_id == "ewma":
        return ewma_scores(train_data, test_data)
    if model_id == "hotelling_t2":
        return hotelling_t2_scores(train_data, test_data)
    raise ValueError(f"Unknown reference baseline: {model_id}")
