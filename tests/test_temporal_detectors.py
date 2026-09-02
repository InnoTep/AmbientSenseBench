import numpy as np

from ambientsensebench.temporal_detectors import (
    BocpdDetector,
    CusumDetector,
    profile_sequential_detector,
)


def _step_change_series(n_features=3, n_train=40, n_stable=60, n_shift=40, shift=2.5, seed=0):
    """Standardised series: stable around 0, then one feature shifts."""
    rng = np.random.default_rng(seed)
    train = rng.normal(0.0, 1.0, size=(n_train, n_features))
    stable = rng.normal(0.0, 1.0, size=(n_stable, n_features))
    shifted = rng.normal(0.0, 1.0, size=(n_shift, n_features))
    shifted[:, 0] += shift
    test = np.vstack([stable, shifted])
    return train, test, n_stable


def test_bocpd_scores_rise_after_mean_shift():
    train, test, n_stable = _step_change_series()
    detector = BocpdDetector()
    train_scores, test_scores = detector.fit_score(train, test)

    assert train_scores.shape == (len(train),)
    assert test_scores.shape == (len(test),)

    pre_shift = np.mean(test_scores[:n_stable])
    post_shift = np.mean(test_scores[n_stable + 3 :])
    assert post_shift > pre_shift + 1.0

    threshold = np.percentile(train_scores, 95)
    post_alarm_rate = np.mean(test_scores[n_stable + 3 :] >= threshold)
    assert post_alarm_rate > 0.9


def test_bocpd_stays_quiet_without_change():
    rng = np.random.default_rng(1)
    train = rng.normal(0.0, 1.0, size=(40, 3))
    test = rng.normal(0.0, 1.0, size=(100, 3))
    detector = BocpdDetector()
    train_scores, test_scores = detector.fit_score(train, test)
    threshold = np.percentile(train_scores, 95)
    assert np.mean(test_scores >= threshold) < 0.30


def test_cusum_scores_rise_after_mean_shift():
    train, test, n_stable = _step_change_series()
    detector = CusumDetector()
    train_scores, test_scores = detector.fit_score(train, test)

    # The chart repeatedly reaches the decision interval during a
    # sustained shift and restarts after each signal (sawtooth), so the
    # post-shift days must contain repeated full-score signals and a
    # clearly higher mean score than the stable period.
    post_shift = test_scores[n_stable + 2 :]
    assert np.sum(post_shift >= detector.h) >= 5
    assert np.mean(post_shift) > np.mean(test_scores[:n_stable]) + 1.0

    threshold = np.percentile(train_scores, 95)
    assert np.mean(post_shift >= threshold) > 0.3


def test_cusum_detects_negative_shift():
    train, test, n_stable = _step_change_series(shift=-2.5, seed=3)
    detector = CusumDetector()
    _, test_scores = detector.fit_score(train, test)
    assert np.mean(test_scores[n_stable + 5 :]) > np.mean(test_scores[:n_stable])


def test_sequential_detectors_are_deterministic():
    train, test, _ = _step_change_series(seed=7)
    for detector_cls in (BocpdDetector, CusumDetector):
        first = detector_cls().fit_score(train, test)
        second = detector_cls().fit_score(train, test)
        assert np.allclose(first[0], second[0])
        assert np.allclose(first[1], second[1])


def test_profile_sequential_detector_reports_costs():
    train, test, _ = _step_change_series(seed=11)
    report = profile_sequential_detector(BocpdDetector(), train, test)
    assert report["median_inference_ms"] > 0
    assert report["p95_inference_ms"] >= report["median_inference_ms"]
    assert report["artifact_kb"] > 0


def test_benchmark_accepts_sequential_models(tmp_path):
    from ambientsensebench.benchmark import run_benchmark

    results, summary = run_benchmark(
        tmp_path,
        seeds=(7,),
        scenarios=("P01",),
        models=("bocpd", "cusum"),
    )
    assert set(results["model_id"]) == {"bocpd", "cusum"}
    assert ((results["auprc"] >= 0) & (results["auprc"] <= 1)).all()
    assert ((results["f1_fixed_threshold"] >= 0) & (results["f1_fixed_threshold"] <= 1)).all()
