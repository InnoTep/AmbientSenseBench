"""Reproduce the supplementary paper results: scenario KS validation,
differential-privacy utility, and edge inference cost.

These three result sets extend the core detector benchmark. They reuse the
already-generated multi-seed scenarios (``<scenario-root>/seed-<n>/P0x``) and
the same baseline-only protocol, so every number in the manuscript can be
regenerated with a single command:

    python -m ambientsensebench.paper_extras \
        --scenario-root outputs/benchmark/scenarios \
        --seeds 0,1,2,3,4 --output outputs/paper

Outputs (under ``--output``):
  ks_per_run.csv, ks_summary.csv          -- distributional validation
  dp_edge_per_run.csv, dp_edge_summary.csv-- clean vs DP utility and edge cost
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .apply_wp7_dp_noise import apply_laplace_noise
from .benchmark import (
    MODEL_LABELS,
    SEQUENTIAL_MODELS,
    _build_model,
    _score_model,
)
from .generate_raw_data import load_profiles
from .generate_scenarios import DEFAULT_CONFIG_PATH
from .temporal_detectors import profile_sequential_detector
from .validate_scenarios_ks import SCENARIOS as KS_SCENARIOS
from .validate_scenarios_ks import validate_one_scenario
from .wp6_evaluation import (
    compute_f1_at_fixed_threshold,
    compute_fixed_threshold_from_training_scores,
    current_rss_mb,
    load_scenario,
    model_size_kb,
    prepare_train_test_arrays,
    run_model_inference_per_day,
    split_train_test,
)

MODELS = tuple(MODEL_LABELS)

# KS alignment thresholds (from profiles/thesis): strong vs acceptable.
STRONG_KS, STRONG_ERR = 0.15, 0.10
ACCEPT_KS, ACCEPT_ERR = 0.30, 0.20


def run_ks_validation(scenario_root: Path, seeds, config_path: Path) -> pd.DataFrame:
    profiles = load_profiles(str(config_path))
    feature_validation = profiles["feature_validation"]
    rows = []
    for seed in seeds:
        seed_dir = scenario_root / f"seed-{seed}"
        for scenario_id in KS_SCENARIOS:
            for record in validate_one_scenario(
                scenario_id=scenario_id,
                scenario_root=seed_dir,
                profiles=profiles,
                feature_validation=feature_validation,
            ):
                record = dict(record)
                record["seed"] = seed
                rows.append(record)
    return pd.DataFrame(rows)


def summarise_ks(ks: pd.DataFrame) -> pd.DataFrame:
    def classify(row) -> str:
        err = abs(row["relative_error"])
        ks_stat = row["ks_statistic"]
        if row["validation_strategy"] == "ks-gaussian":
            if pd.notna(ks_stat) and ks_stat <= STRONG_KS and err <= STRONG_ERR:
                return "strong"
            if pd.notna(ks_stat) and ks_stat <= ACCEPT_KS and err <= ACCEPT_ERR:
                return "acceptable"
            return "weak"
        # moment-only features are judged on mean (and std) relative error
        std_err = abs(row.get("std_relative_error", np.nan))
        worst = max(err, std_err) if pd.notna(std_err) else err
        if worst <= STRONG_ERR:
            return "strong"
        if worst <= ACCEPT_ERR:
            return "acceptable"
        return "weak"

    ks = ks.copy()
    ks["alignment"] = ks.apply(classify, axis=1)
    grouped = ks.groupby(["feature", "validation_strategy"], sort=False)
    summary = grouped.agg(
        n=("alignment", "size"),
        strong=("alignment", lambda s: int((s == "strong").sum())),
        acceptable=("alignment", lambda s: int((s == "acceptable").sum())),
        weak=("alignment", lambda s: int((s == "weak").sum())),
        mean_ks=("ks_statistic", "mean"),
        mean_rel_error=("relative_error", lambda s: float(np.nanmean(np.abs(s)))),
    ).reset_index()
    return summary


def _evaluate(frame: pd.DataFrame, scenario_id: str, model_id: str) -> dict:
    train_df, test_df = split_train_test(frame, scenario_id)
    train_data, test_data, y_test, scaler = prepare_train_test_arrays(train_df, test_df)
    model = _build_model(model_id, len(train_df))
    train_scores, test_scores = _score_model(model_id, model, train_data, test_data)
    threshold = compute_fixed_threshold_from_training_scores(train_scores)
    alarms = (test_scores >= threshold).astype(int)
    from sklearn.metrics import average_precision_score

    return {
        "auprc": float(average_precision_score(y_test, test_scores)),
        "f1": compute_f1_at_fixed_threshold(y_test, alarms),
        "model": model,
        "scaler": scaler,
        "train_data": train_data,
        "test_data": test_data,
        "test_scores": np.asarray(test_scores),
        "y_test": np.asarray(y_test),
        "threshold": float(threshold),
        "alarms": alarms,
    }


def run_dp_edge(scenario_root: Path, seeds, dp_epsilon: float) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        seed_dir = scenario_root / f"seed-{seed}"
        for scenario_id in KS_SCENARIOS:
            clean_frame = load_scenario(str(seed_dir), scenario_id)
            rng = np.random.default_rng(int(seed) * 100 + 7)
            dp_frame = apply_laplace_noise(clean_frame.copy(), epsilon=dp_epsilon, rng=rng)
            for column in ("date", "label", "severity"):
                dp_frame[column] = clean_frame[column].to_numpy()

            for model_id in MODELS:
                clean = _evaluate(clean_frame, scenario_id, model_id)
                dp = _evaluate(dp_frame, scenario_id, model_id)

                # Edge cost from the clean model (deployment cost estimate).
                if model_id in SEQUENTIAL_MODELS:
                    profile = profile_sequential_detector(
                        clean["model"], clean["train_data"], clean["test_data"]
                    )
                    median_ms = profile["median_inference_ms"]
                    p95_ms = profile["p95_inference_ms"]
                    peak_rss = current_rss_mb()
                else:
                    _, inference_ms, peak_rss = run_model_inference_per_day(
                        clean["model"], clean["test_data"]
                    )
                    median_ms = float(np.median(inference_ms))
                    p95_ms = float(np.percentile(inference_ms, 95))
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario_id,
                        "model_id": model_id,
                        "model": MODEL_LABELS[model_id],
                        "auprc_clean": clean["auprc"],
                        "f1_clean": clean["f1"],
                        "auprc_dp": dp["auprc"],
                        "f1_dp": dp["f1"],
                        "auprc_delta": dp["auprc"] - clean["auprc"],
                        "f1_delta": dp["f1"] - clean["f1"],
                        "median_inference_ms": median_ms,
                        "p95_inference_ms": p95_ms,
                        "peak_rss_mb": float(peak_rss),
                        "model_size_kb": model_size_kb(clean["model"], clean["scaler"]),
                    }
                )
    return pd.DataFrame(rows)


def run_dp_threshold_ablation(scenario_root: Path, seeds, dp_epsilon: float) -> pd.DataFrame:
    """Fix the operating threshold from the clean baseline (advisor-requested ablation).

    The deployed protocol recomputes scaler, detector, and the 95th-percentile
    threshold on the noisy training window. This ablation applies the clean
    pipeline's threshold to the noisy pipeline's test scores instead, testing
    whether the sequential detectors' F1 advantage under noise is an artefact
    of threshold re-estimation. AUPRC is threshold-free and unaffected.
    """
    rows = []
    for seed in seeds:
        seed_dir = scenario_root / f"seed-{seed}"
        for scenario_id in KS_SCENARIOS:
            clean_frame = load_scenario(str(seed_dir), scenario_id)
            rng = np.random.default_rng(int(seed) * 100 + 7)
            dp_frame = apply_laplace_noise(clean_frame.copy(), epsilon=dp_epsilon, rng=rng)
            for column in ("date", "label", "severity"):
                dp_frame[column] = clean_frame[column].to_numpy()

            for model_id in MODELS:
                clean = _evaluate(clean_frame, scenario_id, model_id)
                dp = _evaluate(dp_frame, scenario_id, model_id)
                alarms_fixed = (dp["test_scores"] >= clean["threshold"]).astype(int)
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario_id,
                        "model_id": model_id,
                        "model": MODEL_LABELS[model_id],
                        "f1_clean": clean["f1"],
                        "f1_dp_recomputed_threshold": dp["f1"],
                        "f1_dp_clean_threshold": compute_f1_at_fixed_threshold(
                            dp["y_test"], alarms_fixed
                        ),
                        "threshold_clean": clean["threshold"],
                        "threshold_dp": dp["threshold"],
                        "false_alarms_dp_recomputed": int(
                            np.sum((dp["y_test"] == 0) & (dp["alarms"] == 1))
                        ),
                        "false_alarms_dp_clean_threshold": int(
                            np.sum((dp["y_test"] == 0) & (alarms_fixed == 1))
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarise_dp_edge(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "auprc_clean", "f1_clean", "auprc_dp", "f1_dp", "auprc_delta", "f1_delta",
        "median_inference_ms", "p95_inference_ms", "peak_rss_mb", "model_size_kb",
    ]
    grouped = runs.groupby(["scenario", "model_id", "model"], sort=False)
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate supplementary paper results.")
    parser.add_argument("--scenario-root", type=Path, default=Path("outputs/benchmark/scenarios"))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG_PATH))
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/paper"))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.output.mkdir(parents=True, exist_ok=True)

    ks = run_ks_validation(args.scenario_root, seeds, args.config)
    ks.to_csv(args.output / "ks_per_run.csv", index=False)
    ks_summary = summarise_ks(ks)
    ks_summary.to_csv(args.output / "ks_summary.csv", index=False)

    dp_edge = run_dp_edge(args.scenario_root, seeds, args.epsilon)
    dp_edge.to_csv(args.output / "dp_edge_per_run.csv", index=False)
    dp_edge_summary = summarise_dp_edge(dp_edge)
    dp_edge_summary.to_csv(args.output / "dp_edge_summary.csv", index=False)

    ablation = run_dp_threshold_ablation(args.scenario_root, seeds, args.epsilon)
    ablation.to_csv(args.output / "dp_threshold_ablation_per_run.csv", index=False)
    ablation_summary = (
        ablation.groupby(["scenario", "model_id", "model"], sort=False)[
            [
                "f1_clean",
                "f1_dp_recomputed_threshold",
                "f1_dp_clean_threshold",
                "false_alarms_dp_recomputed",
                "false_alarms_dp_clean_threshold",
            ]
        ]
        .mean()
        .reset_index()
    )
    ablation_summary.to_csv(args.output / "dp_threshold_ablation_summary.csv", index=False)

    print("KS alignment summary:")
    print(ks_summary.to_string(index=False))
    print("\nDP + edge summary (means):")
    show = [c for c in dp_edge_summary.columns if c.endswith("_mean") or c in ("scenario", "model")]
    print(dp_edge_summary[["scenario", "model", *[c for c in show if c.endswith('_mean')]]].to_string(index=False))


if __name__ == "__main__":
    main()
