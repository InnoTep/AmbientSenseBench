"""Semi-synthetic routine-change injection on a real home (CASAS Aruba).

The benchmark's features are independent by construction. This check keeps the protocol
but replaces the synthetic baseline with a real resident's 218 days, with whatever coupling
and irregularity that home has. Features come from the raw Aruba log through the benchmark's
own extractor: wake time, social proxy, night activity, mobility and kitchen activity.
Sleep duration, meals and evening consistency need sensors Aruba lacks, and room-transition
entropy sits at its cap of 1 on every Aruba day, so those four are left out.

The first 40 days are the baseline-only training window. Into the 178 test days we inject
the distress profile's mean shift per feature, expressed in the synthetic baseline's
standard deviations and rescaled by the resident's own training-window standard deviation,
weighted by a severity trajectory: a short episode, a sustained one, and a progressive
two-stage ramp (two separated episodes do not fit in 178 days). Severity is zero throughout
the training window. The eight detectors run under the main protocol (training-only
scaling, 95th-percentile threshold, three-day lead-time and persistence rules), clean and
under the per-feature-day Laplace release at epsilon = 1, whose noise is rescaled to the
resident's units the same way (25 noise draws).

Feature-level injection bypasses the raw-to-feature transform: it tests the detectors on
real day-to-day variability, not the event layer.

Usage:  python tools/aruba_injection.py --aruba datasets/CASAS_Aruba/data/aruba.csv
Outputs: outputs/aruba/aruba_features.csv        (real daily features, before injection)
         outputs/aruba/aruba_injection_per_run.csv
         outputs/aruba/aruba_injection_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambientsensebench.apply_wp7_dp_noise import FEATURE_SENSITIVITIES  # noqa: E402
from ambientsensebench.feature_extraction import (  # noqa: E402
    extract_kitchen_activity_raw,
    extract_mobility_raw,
    extract_night_activity,
    extract_social_proxy,
    extract_wake_time,
    load_feature_normalisation,
)
from ambientsensebench.generate_raw_data import load_profiles  # noqa: E402
from ambientsensebench.generate_scenarios import DEFAULT_CONFIG_PATH  # noqa: E402
from ambientsensebench.real_comparison import aruba_events_by_day  # noqa: E402
from ambientsensebench.tier2_experiments import ALL_MODELS, _score_any  # noqa: E402
from ambientsensebench.wp6_evaluation import (  # noqa: E402
    compute_episode_level_lead_time,
    compute_f1_at_fixed_threshold,
    compute_fixed_threshold_from_training_scores,
)

FEATURES = ["wake_time", "social_proxy", "night_activity", "mobility_score",
            "kitchen_activity_score"]
# feature -> (profile mean key, profile standard-deviation key; None means Poisson rate)
PROFILE_KEYS = {
    "wake_time": ("wake_mean", "wake_std"),
    "social_proxy": ("social_mean", "social_std"),
    "night_activity": ("night_lambda", None),
    "mobility_score": ("mobility_mean", "mobility_std"),
    "kitchen_activity_score": ("kitchen_mean", "kitchen_std"),
}
SEQUENTIAL = ("bocpd", "cusum", "ewma")
TRAIN_DAYS = 40
EPSILON = 1.0


def aruba_features(aruba_path: str) -> pd.DataFrame:
    norm = load_feature_normalisation()
    rows = []
    for date, events in sorted(aruba_events_by_day(aruba_path).items()):
        wake = extract_wake_time(events)
        if wake is None:
            continue
        rows.append({
            "date": date,
            "wake_time": wake.hour + wake.minute / 60,
            "social_proxy": extract_social_proxy(events),
            "night_activity": extract_night_activity(events),
            "mobility_score": extract_mobility_raw(events)
            / norm["mobility_score"]["reference_events"],
            "kitchen_activity_score": extract_kitchen_activity_raw(events)
            / norm["kitchen_activity_score"]["reference_events"],
        })
    # drop the partial first and last day, as in real_comparison
    return pd.DataFrame(rows).iloc[1:-1].reset_index(drop=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def severity(kind: str, n_days: int) -> np.ndarray:
    d = np.arange(n_days, dtype=float)
    if kind == "short":
        s = np.minimum(_sigmoid((d - 70) / 2.5), 1 - _sigmoid((d - 100) / 2.5))
    elif kind == "sustained":
        s = np.minimum(_sigmoid((d - 80) / 7.0), 1 - _sigmoid((d - 190) / 7.0))
    elif kind == "progressive":
        s = 0.45 * _sigmoid((d - 90) / 6.0) + 0.55 * _sigmoid((d - 140) / 6.0)
    else:
        raise ValueError(kind)
    s[s < 0.01] = 0.0
    if s[:TRAIN_DAYS].any():
        raise ValueError(f"{kind}: severity leaks into the training window")
    return np.clip(s, 0.0, 1.0)


def profile_shift_and_scale():
    """Distress mean shift and baseline standard deviation of each feature's profile."""
    cfg = load_profiles(DEFAULT_CONFIG_PATH)
    shift, scale = {}, {}
    for feature, (mean_key, std_key) in PROFILE_KEYS.items():
        base = cfg["baseline"][mean_key]
        sd = np.sqrt(base) if std_key is None else cfg["baseline"][std_key]
        shift[feature] = (cfg["distress"][mean_key] - base) / sd
        scale[feature] = sd
    return shift, scale


def evaluate(values: np.ndarray, labels: np.ndarray, model_id: str) -> dict:
    scaler = StandardScaler().fit(values[:TRAIN_DAYS])
    train = scaler.transform(values[:TRAIN_DAYS])
    test = scaler.transform(values[TRAIN_DAYS:])
    train_scores, test_scores = _score_any(model_id, train, test, TRAIN_DAYS)
    threshold = compute_fixed_threshold_from_training_scores(np.asarray(train_scores, float))
    alarms = (np.asarray(test_scores, float) >= threshold).astype(int)
    persist = np.zeros_like(alarms)
    run = 0
    for t, a in enumerate(alarms):
        run = run + 1 if a else 0
        persist[t] = 1 if run >= 3 else 0
    y = (labels[TRAIN_DAYS:] == "distress").astype(int)
    lead = compute_episode_level_lead_time(
        pd.DataFrame({"label": labels[TRAIN_DAYS:]}), alarms)
    return {
        "auprc": float(average_precision_score(y, test_scores)),
        "f1": compute_f1_at_fixed_threshold(y, alarms),
        "f1_persistence3": compute_f1_at_fixed_threshold(y, persist),
        "false_alarms": int(np.sum((y == 0) & (alarms == 1))),
        "lead_time_days": lead["mean_lead_time_days"],
        "no_warning": lead["n_no_warning"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aruba", default="datasets/CASAS_Aruba/data/aruba.csv")
    parser.add_argument("--draws", type=int, default=25, help="Laplace noise draws")
    parser.add_argument("--output", type=Path, default=Path("outputs/aruba"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    real = aruba_features(args.aruba)
    real.to_csv(args.output / "aruba_features.csv", index=False)
    base = real[FEATURES].to_numpy(float)
    n_days = len(base)
    train_sd = base[:TRAIN_DAYS].std(axis=0)
    shift, profile_sd = profile_shift_and_scale()
    shift_vec = np.array([shift[f] for f in FEATURES]) * train_sd
    noise_scale = np.array([FEATURE_SENSITIVITIES[f] / profile_sd[f] for f in FEATURES]) \
        * train_sd / EPSILON
    upper = np.array([24.0 if f == "wake_time" else np.inf for f in FEATURES])
    print(f"{n_days} Aruba days; training SD {dict(zip(FEATURES, train_sd.round(3)))}")
    print(f"shift in synthetic SDs {dict((f, round(shift[f], 2)) for f in FEATURES)}")

    rows = []
    for kind in ("short", "sustained", "progressive"):
        sev = severity(kind, n_days)
        labels = np.where(sev >= 0.5, "distress", "baseline")
        injected = base + sev[:, None] * shift_vec[None, :]
        conditions = [("clean", -1, injected)]
        for draw in range(args.draws):
            rng = np.random.default_rng(1000 + draw)
            noisy = injected + rng.laplace(0.0, noise_scale, size=injected.shape)
            conditions.append(("eps1", draw, np.clip(noisy, 0.0, upper)))
        for condition, draw, values in conditions:
            for model_id, label in ALL_MODELS.items():
                rows.append({"episode": kind, "condition": condition, "draw": draw,
                             "model_id": model_id, "model": label,
                             "family": "sequential" if model_id in SEQUENTIAL else "memoryless",
                             **evaluate(values, labels, model_id)})
        print(f"[{kind}] {int((labels[TRAIN_DAYS:] == 'distress').sum())} positive test days",
              flush=True)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(args.output / "aruba_injection_per_run.csv", index=False)
    metrics = ["auprc", "f1", "f1_persistence3", "false_alarms", "lead_time_days"]
    summary = (per_run.groupby(["episode", "model_id", "model", "family", "condition"])[metrics]
               .mean().unstack("condition"))
    summary.columns = [f"{m}_{c}" for m, c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output / "aruba_injection_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summary.drop(columns=["model_id"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
