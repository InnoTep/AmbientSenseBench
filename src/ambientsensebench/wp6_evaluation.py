import os
import time
import json
import pickle
import tempfile
import psutil
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve


THRESHOLD_PERCENTILE = 95
LEAD_TIME_WINDOW_DAYS = 14
CONSECUTIVE_ALARMS_REQUIRED = 3

TRAIN_DAYS = 40

SCENARIOS = ["P01", "P02", "P03", "P04"]

FEATURE_COLUMNS = [
    "wake_time",
    "sleep_hours",
    "meal_count",
    "social_proxy",
    "night_activity",
    "mobility_score",
    "kitchen_activity_score",
    "room_transition_entropy",
    "evening_routine_consistency",
]


def convert_wake_time_to_hour(value):
    """
    Convert wake_time from 'HH:MM:SS' or numeric format into decimal hours.
    Example: '07:36:00' -> 7.6
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    value = str(value).strip()

    if ":" in value:
        parts = value.split(":")
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) > 2 else 0
        return hour + minute / 60 + second / 3600

    return float(value)


def load_scenario(base_dir, scenario_id):
    """
    Load one scenario and validate required columns.

    The model input excludes date, label, and severity.
    """
    path = os.path.join(base_dir, scenario_id, "daily_features.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing scenario file: {path}")

    df = pd.read_csv(path)

    required_columns = ["date", "label", "severity"] + FEATURE_COLUMNS
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"{scenario_id} missing columns: {missing}")

    if len(df) != 365:
        raise ValueError(f"{scenario_id} should contain 365 rows, found {len(df)}")

    df = df.sort_values("date").reset_index(drop=True)

    df["wake_time"] = df["wake_time"].apply(convert_wake_time_to_hour)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")

    return df


def split_train_test(df, scenario_id):
    """
    First 40 days are used as pure-baseline training data.
    Remaining 325 days are used for testing.
    """
    train_df = df.iloc[:TRAIN_DAYS].copy()
    test_df = df.iloc[TRAIN_DAYS:].copy()

    if len(train_df) != 40:
        raise ValueError(f"{scenario_id}: training window should contain 40 days")

    if len(test_df) != 325:
        raise ValueError(f"{scenario_id}: test window should contain 325 days")

    if not (train_df["label"] == "baseline").all():
        raise ValueError(
            f"{scenario_id}: first 40 training days are not all baseline"
        )

    return train_df, test_df


def prepare_train_test_arrays(train_df, test_df):
    """
    Prepare scaled train/test feature arrays.

    StandardScaler is fitted only on the training window.
    """
    X_train = train_df[FEATURE_COLUMNS].values
    X_test = test_df[FEATURE_COLUMNS].values

    y_test = (test_df["label"] == "distress").astype(int).values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_test, scaler


def compute_fixed_threshold_from_training_scores(train_anomaly_scores):
    """
    Fixed threshold based only on training anomaly scores.

    This avoids test-label leakage.
    """
    return float(np.percentile(train_anomaly_scores, THRESHOLD_PERCENTILE))


def compute_f1_at_fixed_threshold(y_true, y_pred):
    """
    Compute F1 from binary predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)

    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return float(f1)


def select_oracle_f1_threshold(y_true, anomaly_scores):
    """
    Oracle threshold selected on the test precision-recall curve.

    This is an optimistic retrospective upper bound and must not be used
    as the main operational F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)

    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)

    if len(f1_scores) == 0:
        return np.nan, 0.0

    best_idx = int(np.nanargmax(f1_scores))
    oracle_threshold = thresholds[best_idx]
    oracle_f1 = f1_scores[best_idx]

    return float(oracle_threshold), float(oracle_f1)


def find_distress_episodes(labels):
    """
    Find contiguous distress-labelled episodes in the test window.

    Returns a list of (start_idx, end_idx), both inclusive.
    """
    episodes = []
    in_episode = False
    start_idx = None

    for idx, label in enumerate(labels):
        if label == "distress" and not in_episode:
            in_episode = True
            start_idx = idx

        elif label != "distress" and in_episode:
            end_idx = idx - 1
            episodes.append((start_idx, end_idx))
            in_episode = False
            start_idx = None

    if in_episode:
        episodes.append((start_idx, len(labels) - 1))

    return episodes


def has_consecutive_alarms(alarms, start_idx, required):
    """
    Check whether a run of required consecutive alarms starts at start_idx.

    This follows the corrected One-Class SVM lead-time logic:
    the run may start in the bounded pre-onset window, but the consecutive
    alarm condition is evaluated against the full test alarm sequence.
    """
    end_idx = start_idx + required

    if end_idx > len(alarms):
        return False

    return bool(np.all(alarms[start_idx:end_idx] == 1))


def compute_episode_level_lead_time(test_df, y_pred):
    """
    Compute corrected bounded-window, episode-level lead time.

    For each distress episode:
    - Search only for alarm starts within the 14-day pre-onset window.
    - Require 3 consecutive anomaly alarms.
    - If a valid run starts before onset, lead time is positive.
    - If no pre-onset warning exists but a valid run starts on onset day,
      lead time is 0.
    - If no valid warning exists, encode as NaN.

    This is the shared implementation used by both Isolation Forest and
    One-Class SVM.
    """
    labels = test_df["label"].values
    episodes = find_distress_episodes(labels)

    y_pred = np.asarray(y_pred).astype(int)

    episode_lead_times = []
    n_pre_onset_warnings = 0
    n_onset_day_warnings = 0
    n_no_warning = 0

    for onset_idx, _end_idx in episodes:
        window_start = max(0, onset_idx - LEAD_TIME_WINDOW_DAYS)
        window_end = onset_idx - 1

        warning_found = False
        episode_lead_time = np.nan

        # Search pre-onset alarm starts only.
        for idx in range(window_start, window_end + 1):
            if has_consecutive_alarms(
                y_pred,
                idx,
                CONSECUTIVE_ALARMS_REQUIRED,
            ):
                episode_lead_time = float(onset_idx - idx)
                n_pre_onset_warnings += 1
                warning_found = True
                break

        # If no pre-onset warning exists, check onset-day alarm start.
        if not warning_found:
            if has_consecutive_alarms(
                y_pred,
                onset_idx,
                CONSECUTIVE_ALARMS_REQUIRED,
            ):
                episode_lead_time = 0.0
                n_onset_day_warnings += 1
                warning_found = True

        if not warning_found:
            n_no_warning += 1

        episode_lead_times.append(episode_lead_time)

    valid_lead_times = [x for x in episode_lead_times if not np.isnan(x)]

    if len(valid_lead_times) == 0:
        mean_lead_time_days = np.nan
    else:
        mean_lead_time_days = float(np.mean(valid_lead_times))

    return {
        "mean_lead_time_days": mean_lead_time_days,
        "n_episodes": len(episodes),
        "n_pre_onset_warnings": n_pre_onset_warnings,
        "n_onset_day_warnings": n_onset_day_warnings,
        "n_no_warning": n_no_warning,
        "episode_lead_times": episode_lead_times,
    }


def current_rss_mb():
    """
    Return current resident set size memory in MB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def model_size_kb(model, scaler):
    """
    Store model and scaler temporarily, then measure artefact size.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        pickle.dump({"model": model, "scaler": scaler}, tmp)
        tmp_path = tmp.name

    size_kb = os.path.getsize(tmp_path) / 1024
    os.remove(tmp_path)

    return float(size_kb)


def evaluate_model_outputs(
    scenario_id,
    model_name,
    test_df,
    y_test,
    train_anomaly_scores,
    test_anomaly_scores,
    inference_times_ms,
    peak_rss_mb,
    model,
    scaler,
):
    """
    Shared WP6 evaluation for both anomaly-detection models.

    Inputs must already be anomaly scores where larger = more anomalous.
    """
    fixed_threshold = compute_fixed_threshold_from_training_scores(
        train_anomaly_scores
    )

    y_pred_fixed = (test_anomaly_scores >= fixed_threshold).astype(int)

    auprc = average_precision_score(y_test, test_anomaly_scores)

    f1_fixed_threshold = compute_f1_at_fixed_threshold(
        y_test,
        y_pred_fixed,
    )

    oracle_threshold, oracle_f1 = select_oracle_f1_threshold(
        y_test,
        test_anomaly_scores,
    )

    lead_time_summary = compute_episode_level_lead_time(
        test_df,
        y_pred_fixed,
    )

    n_warnings_total = int(np.sum(y_pred_fixed))
    n_false_alarms = int(np.sum((y_test == 0) & (y_pred_fixed == 1)))
    n_true_alarms = int(np.sum((y_test == 1) & (y_pred_fixed == 1)))

    result = {
        "scenario": scenario_id,
        "model": model_name,
        "auprc": round(float(auprc), 4),
        "f1_fixed_threshold": round(float(f1_fixed_threshold), 4),
        "threshold_rule": f"training_p{THRESHOLD_PERCENTILE}",
        "fixed_threshold": round(float(fixed_threshold), 6),
        "oracle_f1": round(float(oracle_f1), 4),
        "oracle_threshold": round(float(oracle_threshold), 6),
        "mean_lead_time_days": (
            round(float(lead_time_summary["mean_lead_time_days"]), 4)
            if not np.isnan(lead_time_summary["mean_lead_time_days"])
            else np.nan
        ),
        "n_episodes": lead_time_summary["n_episodes"],
        "n_pre_onset_warnings": lead_time_summary["n_pre_onset_warnings"],
        "n_onset_day_warnings": lead_time_summary["n_onset_day_warnings"],
        "n_no_warning": lead_time_summary["n_no_warning"],
        "episode_lead_times": json.dumps(lead_time_summary["episode_lead_times"]),
        "n_warnings_total": n_warnings_total,
        "n_true_alarms": n_true_alarms,
        "n_false_alarms": n_false_alarms,
        "n_train": TRAIN_DAYS,
        "n_test": len(test_df),
        "n_distress_test": int(y_test.sum()),
        "n_baseline_test": int(len(y_test) - y_test.sum()),
        "positive_rate_test": round(float(y_test.mean()), 4),
        "median_inference_ms": round(float(np.median(inference_times_ms)), 6),
        "p95_inference_ms": round(float(np.percentile(inference_times_ms, 95)), 6),
        "peak_rss_mb": round(float(peak_rss_mb), 2),
        "model_size_kb": round(float(model_size_kb(model, scaler)), 2),
    }

    return result


def run_model_inference_per_day(model, X_test_scaled):
    """
    Run one-row-at-a-time inference to approximate daily edge inference.

    Returns:
    - test anomaly scores, where larger = more anomalous
    - per-day inference times in ms
    - peak RSS memory in MB
    """
    inference_times_ms = []
    anomaly_scores = []

    peak_rss_mb = current_rss_mb()

    for row in X_test_scaled:
        row_2d = row.reshape(1, -1)

        rss_before = current_rss_mb()

        start = time.perf_counter()
        score = -model.decision_function(row_2d)[0]
        end = time.perf_counter()

        rss_after = current_rss_mb()
        peak_rss_mb = max(peak_rss_mb, rss_before, rss_after)

        inference_times_ms.append((end - start) * 1000)
        anomaly_scores.append(score)

    return (
        np.array(anomaly_scores),
        np.array(inference_times_ms),
        peak_rss_mb,
    )