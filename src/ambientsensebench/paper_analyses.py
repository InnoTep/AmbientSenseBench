"""Paper analyses derived from the 25-seed reference run (after phases A and B).

* ``severity``       -- Spearman correlation between each stored test-window score and
  the simulated severity (``severity_consistency.csv``), and the mean within-run score
  percentile per severity bin on P04 (``severity_curve_p04.csv``, Figure 3)
* ``dp-baselines``   -- phase B repeated for the three reference baselines
  (``dp_sweep_baselines.csv``)
* ``dp-persistence`` -- at epsilon = 1, the three-consecutive-day persistence rule applied
  to the noisy alarm streams of the core detectors (``dp_persistence_eps1.csv``)
* ``summary``        -- mean and standard deviation per scenario and core detector
  (``summary_core25.csv``), the input of Figures 1 and 2

Noise is drawn exactly as in phase B, so the noisy feature matrices match it.

Usage:
    python -m ambientsensebench.paper_analyses --analysis all --workers 2 \
        --output outputs/tier2 --scenario-root outputs/tier2/scenarios
"""

from __future__ import annotations

import argparse
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .apply_wp7_dp_noise import apply_laplace_noise
from .benchmark import summarise_results
from .generate_scenarios import SCENARIOS
from .reference_baselines import BASELINE_LABELS
from .tier2_experiments import (
    ALL_MODELS,
    CORE_MODELS,
    EPSILONS,
    _evaluate_frame,
    _init_worker,
)
from .wp6_evaluation import (
    TRAIN_DAYS,
    compute_f1_at_fixed_threshold,
    compute_fixed_threshold_from_training_scores,
    load_scenario,
)

SEVERITY_BINS = 10


def _persistence(alarms: np.ndarray, run_length: int = 3) -> np.ndarray:
    """A day alarms only if it ends a run of at least ``run_length`` over-threshold days."""
    out = np.zeros_like(alarms)
    run = 0
    for t, a in enumerate(alarms):
        run = run + 1 if a else 0
        out[t] = 1 if run >= run_length else 0
    return out


def _noisy_frame(scen_root, seed, scenario, epsilon):
    clean = load_scenario(str(Path(scen_root) / f"seed-{seed}"), scenario)
    rng = np.random.default_rng(int(seed) * 100 + 7)  # phase B seeding, per epsilon
    noisy = apply_laplace_noise(clean.copy(), epsilon=epsilon, rng=rng)
    for column in ("date", "label", "severity"):
        noisy[column] = clean[column].to_numpy()
    return noisy


# --------------------------------------------------------------------------
# Score--severity consistency
# --------------------------------------------------------------------------

def severity_analysis(out: Path, scen_root: Path, seeds):
    rows, curves = [], []
    for seed in seeds:
        for scenario in SCENARIOS:
            labels = pd.read_csv(Path(scen_root) / f"seed-{seed}" / scenario
                                 / "scenario_labels.csv")
            severity = labels["severity"].to_numpy(float)[TRAIN_DAYS:]
            for model_id in ALL_MODELS:
                stored = np.load(out / "scores" / f"s{seed}_{scenario}_{model_id}.npz")
                scores = stored["test_scores"]
                rows.append({"seed": seed, "scenario": scenario, "model_id": model_id,
                             "spearman": float(spearmanr(scores, severity).statistic)})
                if scenario == "P04" and model_id in CORE_MODELS:
                    percentile = rankdata(scores) / len(scores)
                    bins = np.minimum((severity * SEVERITY_BINS).astype(int),
                                      SEVERITY_BINS - 1)
                    for b in range(SEVERITY_BINS):
                        if np.any(bins == b):
                            curves.append({"seed": seed, "model_id": model_id, "bin": b,
                                           "percentile": float(percentile[bins == b].mean())})
    pd.DataFrame(rows).to_csv(out / "severity_consistency.csv", index=False)
    curve = (pd.DataFrame(curves).groupby(["bin", "model_id"]).percentile.mean()
             .unstack("model_id")[list(CORE_MODELS)])
    curve.index.name = None
    curve.columns.name = None
    curve.to_csv(out / "severity_curve_p04.csv")
    print("[severity] done", flush=True)


# --------------------------------------------------------------------------
# Reference baselines under the release, and persistence under the release
# --------------------------------------------------------------------------

def _dp_baselines_unit(args):
    seed, scenario, scen_root, score_dir = args
    rows = []
    for eps in EPSILONS:
        frame = _noisy_frame(scen_root, seed, scenario, eps)
        for model_id in BASELINE_LABELS:
            r = _evaluate_frame(frame, scenario, model_id)
            stored = np.load(Path(score_dir) / f"s{seed}_{scenario}_{model_id}.npz")
            thr_clean = compute_fixed_threshold_from_training_scores(stored["train_scores"])
            alarms_clean_thr = (r["test_scores"] >= thr_clean).astype(int)
            rows.append({
                "seed": seed, "scenario": scenario, "model_id": model_id,
                "model": BASELINE_LABELS[model_id], "epsilon": eps,
                "auprc_dp": r["auprc"], "f1_dp_recomputed": r["f1"],
                "f1_dp_cleanthr": compute_f1_at_fixed_threshold(r["y_test"], alarms_clean_thr),
                "fp_dp_recomputed": r["fp"],
            })
    return rows


def _dp_persistence_unit(args):
    seed, scenario, scen_root, _ = args
    frame = _noisy_frame(scen_root, seed, scenario, 1.0)
    rows = []
    for model_id in CORE_MODELS:
        r = _evaluate_frame(frame, scenario, model_id)
        alarms = (r["test_scores"] >= r["threshold"]).astype(int)
        persist = _persistence(alarms)
        y = r["y_test"]
        rows.append({
            "seed": seed, "scenario": scenario, "model_id": model_id,
            "model": ALL_MODELS[model_id], "epsilon": 1.0,
            "f1_dp_recomputed": r["f1"],
            "f1_dp_persistence3": compute_f1_at_fixed_threshold(y, persist),
            "fp_dp_per_day": int(np.sum((y == 0) & (alarms == 1))),
            "fp_dp_persistence3": int(np.sum((y == 0) & (persist == 1))),
        })
    return rows


def _run_units(unit, out, seeds, workers, scen_root, filename, sort_by, label):
    units = [(s, sc, str(scen_root), str(out / "scores")) for s in seeds for sc in SCENARIOS]
    rows = []
    with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for chunk in pool.imap_unordered(unit, units):
            rows.extend(chunk)
    pd.DataFrame(rows).sort_values(sort_by).to_csv(out / filename, index=False)
    print(f"[{label}] done", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Paper analyses from the reference run")
    parser.add_argument("--analysis", required=True,
                        choices=["summary", "severity", "dp-baselines", "dp-persistence", "all"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("outputs/tier2"))
    parser.add_argument("--scenario-root", type=Path, default=Path("outputs/tier2/scenarios"))
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    if args.analysis in ("summary", "all"):
        per_run = pd.read_csv(args.output / "per_run_main.csv")
        summarise_results(per_run[per_run.model_id.isin(CORE_MODELS)]).to_csv(
            args.output / "summary_core25.csv", index=False)
        print("[summary] done", flush=True)
    if args.analysis in ("severity", "all"):
        severity_analysis(args.output, args.scenario_root, seeds)
    if args.analysis in ("dp-baselines", "all"):
        _run_units(_dp_baselines_unit, args.output, seeds, args.workers, args.scenario_root,
                   "dp_sweep_baselines.csv", ["scenario", "model_id", "epsilon", "seed"],
                   "dp-baselines")
    if args.analysis in ("dp-persistence", "all"):
        _run_units(_dp_persistence_unit, args.output, seeds, args.workers, args.scenario_root,
                   "dp_persistence_eps1.csv", ["scenario", "model_id", "seed"],
                   "dp-persistence")


if __name__ == "__main__":
    main()
