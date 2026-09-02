"""Sequential change-aware detectors for the multi-seed benchmark.

The reference detectors (Isolation Forest, One-Class SVM, Local Outlier
Factor) are memoryless: each day is scored independently of every other
day, so the temporal structure of a routine change never enters the
score. This module adds two lightweight sequential detectors that carry
state across days and therefore score a day in the context of the days
before it:

- ``BocpdDetector`` implements per-feature Bayesian Online Change-Point
  Detection (Adams and MacKay, 2007) with a Normal-Inverse-Gamma
  conjugate model and a constant hazard. The per-day anomaly score is
  the posterior expected absolute deviation of the current segment mean
  from the baseline mean in standardised units, maximised over
  features. This "regime divergence" score stays elevated for as long
  as the inferred current regime differs from baseline, which makes it
  comparable to the day-level state scores of the memoryless detectors
  while remaining strictly online.

- ``CusumDetector`` implements the classic two-sided standardised CUSUM
  chart (Page, 1954) with reference value ``k`` in baseline
  standard-deviation units. The per-day statistic is the larger of the
  upper and lower cumulative sums, maximised over features.

Both detectors expose ``fit_score(train_data, test_data)`` and return
``(train_scores, test_scores)`` with the benchmark convention that a
larger score means a more anomalous day. The recursion runs once over
the concatenated train-plus-test sequence in chronological order;
training scores are used only for the fixed training-percentile
threshold, so no test information reaches the threshold rule.

Inputs are expected to be standardised with a scaler fitted on the
baseline training window only (the shared protocol already does this),
so the baseline mean is 0 and the baseline standard deviation is 1 for
every feature.
"""

from __future__ import annotations

import math
import pickle
import time

import numpy as np

__all__ = ["BocpdDetector", "CusumDetector", "profile_sequential_detector"]


def _student_t_logpdf(x: np.ndarray, df: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Vectorised log-density of the Student-t distribution."""
    z = (x - loc) / scale
    return (
        _gammaln((df + 1.0) / 2.0)
        - _gammaln(df / 2.0)
        - 0.5 * np.log(df * math.pi)
        - np.log(scale)
        - ((df + 1.0) / 2.0) * np.log1p(z * z / df)
    )


def _gammaln(x: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    return gammaln(x)


class _BocpdFeatureState:
    """Run-length posterior and NIG sufficient statistics for one feature."""

    def __init__(self, mu0: float, kappa0: float, alpha0: float, beta0: float):
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.log_r = np.array([0.0])  # log P(r_0 = 0) = 1
        self.mu = np.array([mu0])
        self.kappa = np.array([kappa0])
        self.alpha = np.array([alpha0])
        self.beta = np.array([beta0])

    def step(self, x: float, log_hazard: float, log_one_minus_hazard: float) -> float:
        """Advance one day and return the regime-divergence score."""
        df = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        log_pred = _student_t_logpdf(np.asarray(x, dtype=float), df, self.mu, scale)

        log_growth = self.log_r + log_pred + log_one_minus_hazard
        log_cp = _logsumexp(self.log_r + log_pred + log_hazard)

        log_r_new = np.concatenate(([log_cp], log_growth))
        log_r_new -= _logsumexp(log_r_new)

        mu_new = np.concatenate(
            ([self.mu0], (self.kappa * self.mu + x) / (self.kappa + 1.0))
        )
        beta_new = np.concatenate(
            (
                [self.beta0],
                self.beta + self.kappa * (x - self.mu) ** 2 / (2.0 * (self.kappa + 1.0)),
            )
        )
        kappa_new = np.concatenate(([self.kappa0], self.kappa + 1.0))
        alpha_new = np.concatenate(([self.alpha0], self.alpha + 0.5))

        self.log_r = log_r_new
        self.mu = mu_new
        self.kappa = kappa_new
        self.alpha = alpha_new
        self.beta = beta_new

        # Posterior expected absolute deviation of the current segment
        # mean from the baseline mean (0 in standardised units).
        return float(np.sum(np.exp(self.log_r) * np.abs(self.mu)))


def _logsumexp(values: np.ndarray) -> float:
    peak = np.max(values)
    if np.isneginf(peak):
        return float("-inf")
    return float(peak + np.log(np.sum(np.exp(values - peak))))


class BocpdDetector:
    """Per-feature BOCPD with a regime-divergence anomaly score.

    Parameters
    ----------
    hazard_lambda:
        Expected number of days between change points; the constant
        hazard is ``1 / hazard_lambda``. The default of 50 matches the
        temporal scale of the reference scenarios, where episodes are
        separated by several weeks.
    mu0, kappa0, alpha0, beta0:
        Normal-Inverse-Gamma prior. The defaults encode a unit-scale
        prior centred on the standardised baseline (mean 0, variance of
        order 1), which is appropriate because inputs are standardised
        with the baseline-window scaler.
    """

    model_kind = "sequential"

    def __init__(
        self,
        hazard_lambda: float = 50.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ):
        if hazard_lambda <= 1.0:
            raise ValueError("hazard_lambda must be greater than 1")
        self.hazard_lambda = float(hazard_lambda)
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self._states: list[_BocpdFeatureState] | None = None

    def _reset(self, n_features: int) -> None:
        self._states = [
            _BocpdFeatureState(self.mu0, self.kappa0, self.alpha0, self.beta0)
            for _ in range(n_features)
        ]

    def _step_day(self, row: np.ndarray) -> float:
        hazard = 1.0 / self.hazard_lambda
        log_hazard = math.log(hazard)
        log_one_minus_hazard = math.log1p(-hazard)
        assert self._states is not None
        return max(
            state.step(float(value), log_hazard, log_one_minus_hazard)
            for state, value in zip(self._states, row)
        )

    def fit_score(self, train_data: np.ndarray, test_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the recursion over train then test days in order."""
        train_data = np.asarray(train_data, dtype=float)
        test_data = np.asarray(test_data, dtype=float)
        self._reset(train_data.shape[1])
        train_scores = np.array([self._step_day(row) for row in train_data])
        test_scores = np.array([self._step_day(row) for row in test_data])
        return train_scores, test_scores

    def state_artifact(self) -> dict:
        """Serialisable per-feature state, used for artefact-size profiling."""
        assert self._states is not None
        return {
            "hazard_lambda": self.hazard_lambda,
            "states": [
                {
                    "log_r": s.log_r,
                    "mu": s.mu,
                    "kappa": s.kappa,
                    "alpha": s.alpha,
                    "beta": s.beta,
                }
                for s in self._states
            ],
        }


class CusumDetector:
    """Two-sided standardised CUSUM chart with reset on signal.

    This follows the classical tabular CUSUM scheme (Page, 1954): for
    each feature the chart maintains
    ``S_plus_t = max(0, S_plus_{t-1} + x_t - k)`` and
    ``S_minus_t = max(0, S_minus_{t-1} - x_t - k)``. When either sum
    reaches the decision interval ``h``, the chart signals and that
    feature's sums restart from zero, as in the standard repeated
    sequential-test formulation. Without the restart, the statistic of
    a long-recovered episode would take months of baseline days to
    drain back below threshold, which both delays recovery detection
    and corrupts day-level score ranking.

    The per-day anomaly score is ``min(h, max(S_plus_t, S_minus_t))``
    maximised over features, evaluated before the restart so signal
    days keep their full score. With standardised inputs, ``k = 0.5``
    and ``h = 5`` are the classical values tuned to detect
    one-standard-deviation mean shifts.
    """

    model_kind = "sequential"

    def __init__(self, k: float = 0.5, h: float = 5.0):
        if k < 0:
            raise ValueError("k must be non-negative")
        if h <= 0:
            raise ValueError("h must be positive")
        self.k = float(k)
        self.h = float(h)
        self._s_plus: np.ndarray | None = None
        self._s_minus: np.ndarray | None = None

    def _reset(self, n_features: int) -> None:
        self._s_plus = np.zeros(n_features)
        self._s_minus = np.zeros(n_features)

    def _step_day(self, row: np.ndarray) -> float:
        assert self._s_plus is not None and self._s_minus is not None
        self._s_plus = np.maximum(0.0, self._s_plus + row - self.k)
        self._s_minus = np.maximum(0.0, self._s_minus - row - self.k)
        per_feature = np.maximum(self._s_plus, self._s_minus)
        score = float(np.min([self.h, np.max(per_feature)]))

        # Restart the charts of every feature that signalled, as in the
        # repeated sequential-test formulation of the CUSUM scheme.
        signalled = per_feature >= self.h
        if np.any(signalled):
            self._s_plus[signalled] = 0.0
            self._s_minus[signalled] = 0.0
        return score

    def fit_score(self, train_data: np.ndarray, test_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        train_data = np.asarray(train_data, dtype=float)
        test_data = np.asarray(test_data, dtype=float)
        self._reset(train_data.shape[1])
        train_scores = np.array([self._step_day(row) for row in train_data])
        test_scores = np.array([self._step_day(row) for row in test_data])
        return train_scores, test_scores

    def state_artifact(self) -> dict:
        return {"k": self.k, "h": self.h, "s_plus": self._s_plus, "s_minus": self._s_minus}


def profile_sequential_detector(detector, train_data: np.ndarray, test_data: np.ndarray) -> dict:
    """Per-day inference profiling consistent with the shared protocol.

    Re-runs the sequential recursion with one timed step per test day
    and reports the median and 95th-percentile step latency together
    with the serialised size of the detector state.
    """
    train_data = np.asarray(train_data, dtype=float)
    test_data = np.asarray(test_data, dtype=float)
    detector._reset(train_data.shape[1])
    for row in train_data:
        detector._step_day(row)

    step_times_ms = []
    for row in test_data:
        start = time.perf_counter()
        detector._step_day(row)
        end = time.perf_counter()
        step_times_ms.append((end - start) * 1000.0)

    artefact_bytes = len(pickle.dumps(detector.state_artifact()))
    return {
        "median_inference_ms": float(np.median(step_times_ms)),
        "p95_inference_ms": float(np.percentile(step_times_ms, 95)),
        "artifact_kb": artefact_bytes / 1024.0,
    }
