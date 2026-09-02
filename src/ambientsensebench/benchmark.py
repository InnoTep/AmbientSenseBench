"""Reproducible multi-seed benchmark runner for ambient-sensing scenarios."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .generate_scenarios import SCENARIOS, generate_scenario
from .temporal_detectors import BocpdDetector, CusumDetector
from .wp6_evaluation import (
    compute_episode_level_lead_time,
    compute_f1_at_fixed_threshold,
    compute_fixed_threshold_from_training_scores,
    load_scenario,
    prepare_train_test_arrays,
    split_train_test,
)

MODEL_LABELS = {
    "isolation_forest": "Isolation Forest",
    "one_class_svm": "One-Class SVM",
    "local_outlier_factor": "Local Outlier Factor",
    "bocpd": "BOCPD",
    "cusum": "CUSUM",
}
# Sequential detectors carry state across days; the memoryless
# detectors score each day independently.
SEQUENTIAL_MODELS = ("bocpd", "cusum")
DEFAULT_MODELS = tuple(MODEL_LABELS)
SUMMARY_METRICS = (
    "auprc",
    "f1_fixed_threshold",
    "n_warnings_total",
    "n_true_alarms",
    "n_false_alarms",
    "mean_lead_time_days",
)


def _build_model(model_id: str, training_rows: int):
    if model_id == "isolation_forest":
        return IsolationForest(
            n_estimators=100,
            contamination="auto",
            n_jobs=1,
            random_state=42,
        )
    if model_id == "one_class_svm":
        return OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    if model_id == "local_outlier_factor":
        return LocalOutlierFactor(
            n_neighbors=min(20, training_rows - 1),
            contamination="auto",
            n_jobs=1,
            novelty=True,
        )
    if model_id == "bocpd":
        return BocpdDetector()
    if model_id == "cusum":
        return CusumDetector()
    raise ValueError(f"Unknown model: {model_id}")


def _score_model(model_id: str, model, train_data, test_data) -> tuple[np.ndarray, np.ndarray]:
    if model_id in SEQUENTIAL_MODELS:
        # Sequential detectors process train and test days in
        # chronological order; training scores feed only the fixed
        # training-percentile threshold rule.
        train_scores, test_scores = model.fit_score(train_data, test_data)
        return np.asarray(train_scores), np.asarray(test_scores)

    model.fit(train_data)

    if model_id == "local_outlier_factor":
        train_scores = -model.negative_outlier_factor_
    else:
        train_scores = -model.decision_function(train_data)

    test_scores = -model.decision_function(test_data)
    return np.asarray(train_scores), np.asarray(test_scores)


def evaluate_detector(
    df: pd.DataFrame,
    scenario_id: str,
    model_id: str,
    seed: int,
) -> dict[str, object]:
    """Evaluate one detector using the fixed baseline-only training protocol."""
    train_df, test_df = split_train_test(df, scenario_id)
    train_data, test_data, y_test, _ = prepare_train_test_arrays(train_df, test_df)
    model = _build_model(model_id, len(train_df))
    train_scores, test_scores = _score_model(model_id, model, train_data, test_data)

    threshold = compute_fixed_threshold_from_training_scores(train_scores)
    alarms = (test_scores >= threshold).astype(int)
    lead_time = compute_episode_level_lead_time(test_df, alarms)

    return {
        "seed": seed,
        "scenario": scenario_id,
        "model_id": model_id,
        "model": MODEL_LABELS[model_id],
        "auprc": float(average_precision_score(y_test, test_scores)),
        "f1_fixed_threshold": compute_f1_at_fixed_threshold(y_test, alarms),
        "fixed_threshold": float(threshold),
        "n_warnings_total": int(alarms.sum()),
        "n_true_alarms": int(np.sum((y_test == 1) & (alarms == 1))),
        "n_false_alarms": int(np.sum((y_test == 0) & (alarms == 1))),
        "mean_lead_time_days": lead_time["mean_lead_time_days"],
        "n_episodes": lead_time["n_episodes"],
        "n_pre_onset_warnings": lead_time["n_pre_onset_warnings"],
        "n_onset_day_warnings": lead_time["n_onset_day_warnings"],
        "n_no_warning": lead_time["n_no_warning"],
        "n_train": len(train_df),
        "n_test": len(test_df),
    }


def summarise_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate core performance metrics by scenario and detector."""
    grouped = results.groupby(["scenario", "model_id", "model"], sort=False)
    summary = grouped[list(SUMMARY_METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.reset_index()
    summary["runs"] = grouped.size().to_numpy()
    for metric in SUMMARY_METRICS:
        summary[f"{metric}_std"] = summary[f"{metric}_std"].fillna(0.0)
    return summary


def run_benchmark(
    output_root: Path,
    seeds: tuple[int, ...],
    scenarios: tuple[str, ...] = SCENARIOS,
    models: tuple[str, ...] = DEFAULT_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate scenarios for every seed and write per-run and summary tables."""
    if not seeds:
        raise ValueError("At least one seed is required")
    unknown_scenarios = set(scenarios).difference(SCENARIOS)
    unknown_models = set(models).difference(MODEL_LABELS)
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {sorted(unknown_scenarios)}")
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")

    output_root = Path(output_root)
    scenario_root = output_root / "scenarios"
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for seed in seeds:
        seed_root = scenario_root / f"seed-{seed}"
        for scenario_id in scenarios:
            generate_scenario(
                scenario_id,
                seed_override=seed,
                output_root=seed_root,
            )
            scenario_frame = load_scenario(str(seed_root), scenario_id)
            for model_id in models:
                records.append(
                    evaluate_detector(scenario_frame, scenario_id, model_id, seed)
                )

    results = pd.DataFrame(records)
    summary = summarise_results(results)
    results.to_csv(output_root / "per_run_results.csv", index=False, float_format="%.6f")
    summary.to_csv(output_root / "summary_results.csv", index=False, float_format="%.6f")
    (output_root / "benchmark_config.json").write_text(
        json.dumps(
            {
                "seeds": list(seeds),
                "scenarios": list(scenarios),
                "models": list(models),
                "training_days": 40,
                "threshold_rule": "training_p95",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return results, summary


def parse_csv_argument(value: str, allowed: tuple[str, ...] | None = None) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("At least one comma-separated value is required")
    if allowed and set(values).difference(allowed):
        raise ValueError(f"Allowed values: {', '.join(allowed)}")
    return values
