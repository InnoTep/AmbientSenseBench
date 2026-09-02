"""Tier-2 experiment runner: 25-seed benchmark with reference baselines,
epsilon sweep with threshold ablation, sensitivity at full seed count, and
hyperparameter/feature/window sweeps.

Designed for a small multi-core machine: work units are independent
(seed, scenario[, config]) evaluations farmed over a process pool, and the
expensive event-level generation is done once and shared by every phase
that can reuse it.  Results are identical to a serial run (generation and
noise are seeded deterministically per unit; execution order is irrelevant).

Phases (run in this order; later phases reuse phase-A artefacts):
  A  main      -- generate 25 seeds x 4 scenarios; evaluate 5 core detectors
                  + 3 reference baselines; store per-day scores (.npz)
  B  dpsweep   -- Laplace release at eps in {0.1,0.5,1,2,5,10}; core
                  detectors; recomputed-threshold and clean-threshold F1
  C  sensitivity -- alpha in {0.8, 1.2} regenerated at 25 seeds (alpha=1.0
                  equals phase A by construction)
  E  sweeps    -- BOCPD hazard sweep, CUSUM h sweep, feature-set ablation,
                  P04 training-window sweep (P01-P03 onsets forbid longer
                  windows)
  F  extras    -- tracemalloc incremental memory, event data volume,
                  baseline feature correlation matrix

Usage:
    python -m ambientsensebench.tier2_experiments --phase A --workers 2 \
        --seeds 25 --output outputs/tier2
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from multiprocessing import Pool, get_context
from pathlib import Path

import numpy as np
import pandas as pd

from .apply_wp7_dp_noise import (
    FEATURE_CLIPPING_BOUNDS,
    FEATURE_SENSITIVITIES,
    MODEL_INPUT_FEATURES,
    apply_laplace_noise,
)
from .benchmark import MODEL_LABELS, _build_model, _score_model
from .generate_scenarios import SCENARIOS, generate_scenario
from .reference_baselines import BASELINE_LABELS, score_baseline
from .sensitivity import write_perturbed_config
from .temporal_detectors import BocpdDetector, CusumDetector
from .wp6_evaluation import (
    FEATURE_COLUMNS,
    compute_episode_level_lead_time,
    compute_f1_at_fixed_threshold,
    compute_fixed_threshold_from_training_scores,
    load_scenario,
    prepare_train_test_arrays,
    split_train_test,
)
from sklearn.metrics import average_precision_score

CORE_MODELS = tuple(MODEL_LABELS)
ALL_MODELS = {**MODEL_LABELS, **BASELINE_LABELS}
EPSILONS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
# BOCPD hazard sweep, expressed as the expected run length 1/h used by the
# constructor (hazard_lambda); 50 is the phase-A default.
HAZARDS = (25, 100, 200)
CUSUM_HS = (3.0, 4.0, 6.0)  # 5 is the phase-A default
FEATURE_SETS = {
    "no_weak3": [f for f in FEATURE_COLUMNS if f not in (
        "meal_count", "room_transition_entropy", "evening_routine_consistency")],
    "gaussian4": ["wake_time", "sleep_hours", "mobility_score", "kitchen_activity_score"],
}
P04_WINDOWS = (60, 90)  # onset day 106; P01-P03 onsets (59, 44, 49) forbid > 40


def _init_worker():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _score_any(model_id, train_data, test_data, n_train):
    if model_id in MODEL_LABELS:
        model = _build_model(model_id, n_train)
        return _score_model(model_id, model, train_data, test_data)
    return score_baseline(model_id, train_data, test_data)


def _evaluate_frame(frame, scenario, model_id, train_days=None):
    if train_days is None:
        train_df, test_df = split_train_test(frame, scenario)
    else:
        train_df = frame.iloc[:train_days].copy()
        test_df = frame.iloc[train_days:].copy()
        if not (train_df["label"] == "baseline").all():
            raise ValueError(f"{scenario}: window {train_days} not baseline-only")
    train_data, test_data, y_test, _ = prepare_train_test_arrays(train_df, test_df)
    train_scores, test_scores = _score_any(model_id, train_data, test_data, len(train_df))
    train_scores = np.asarray(train_scores, float)
    test_scores = np.asarray(test_scores, float)
    threshold = compute_fixed_threshold_from_training_scores(train_scores)
    alarms = (test_scores >= threshold).astype(int)
    lead = compute_episode_level_lead_time(test_df, alarms)
    return {
        "auprc": float(average_precision_score(y_test, test_scores)),
        "f1": compute_f1_at_fixed_threshold(y_test, alarms),
        "threshold": float(threshold),
        "n_alarms": int(alarms.sum()),
        "tp": int(np.sum((y_test == 1) & (alarms == 1))),
        "fp": int(np.sum((y_test == 0) & (alarms == 1))),
        "lead": lead,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "y_test": np.asarray(y_test, int),
    }


# --------------------------------------------------------------------------
# Phase A: main benchmark, 25 seeds, 8 detectors, scores stored
# --------------------------------------------------------------------------

def _phase_a_unit(args):
    seed, scenario, scen_root, score_dir = args
    seed_root = Path(scen_root) / f"seed-{seed}"
    feat = seed_root / scenario / "daily_features.csv"
    if not feat.exists():
        generate_scenario(scenario, seed_override=seed, output_root=str(seed_root))
    # raw event days are only needed for the data-volume analysis (seeds 0-4)
    raw = seed_root / scenario / "raw_days"
    if seed > 4 and raw.exists():
        shutil.rmtree(raw, ignore_errors=True)
    frame = load_scenario(str(seed_root), scenario)
    rows = []
    for model_id in ALL_MODELS:
        r = _evaluate_frame(frame, scenario, model_id)
        np.savez_compressed(
            Path(score_dir) / f"s{seed}_{scenario}_{model_id}.npz",
            train_scores=r["train_scores"], test_scores=r["test_scores"],
            y_test=r["y_test"],
        )
        rows.append({
            "seed": seed, "scenario": scenario, "model_id": model_id,
            "model": ALL_MODELS[model_id], "auprc": r["auprc"],
            "f1_fixed_threshold": r["f1"], "fixed_threshold": r["threshold"],
            "n_warnings_total": r["n_alarms"], "n_true_alarms": r["tp"],
            "n_false_alarms": r["fp"],
            "mean_lead_time_days": r["lead"]["mean_lead_time_days"],
            "n_episodes": r["lead"]["n_episodes"],
            "n_no_warning": r["lead"]["n_no_warning"],
        })
    return rows


def run_phase_a(out, seeds, workers, scen_root):
    score_dir = out / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    units = [(s, sc, str(scen_root), str(score_dir)) for s in seeds for sc in SCENARIOS]
    rows = []
    with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_phase_a_unit, units)):
            rows.extend(chunk)
            if (i + 1) % 10 == 0:
                pd.DataFrame(rows).to_csv(out / "per_run_main.csv", index=False)
                print(f"[A] {i + 1}/{len(units)} units", flush=True)
    df = pd.DataFrame(rows).sort_values(["scenario", "model_id", "seed"])
    df.to_csv(out / "per_run_main.csv", index=False)
    print(f"[A] done: {len(df)} rows", flush=True)


# --------------------------------------------------------------------------
# Phase B: epsilon sweep with recomputed- and clean-threshold F1
# --------------------------------------------------------------------------

def _phase_b_unit(args):
    seed, scenario, scen_root, score_dir = args
    seed_root = Path(scen_root) / f"seed-{seed}"
    clean_frame = load_scenario(str(seed_root), scenario)
    rows = []
    for eps in EPSILONS:
        rng = np.random.default_rng(int(seed) * 100 + 7)
        dp_frame = apply_laplace_noise(clean_frame.copy(), epsilon=eps, rng=rng)
        for column in ("date", "label", "severity"):
            dp_frame[column] = clean_frame[column].to_numpy()
        for model_id in CORE_MODELS:
            r = _evaluate_frame(dp_frame, scenario, model_id)
            stored = np.load(Path(score_dir) / f"s{seed}_{scenario}_{model_id}.npz")
            thr_clean = compute_fixed_threshold_from_training_scores(stored["train_scores"])
            alarms_fixed = (r["test_scores"] >= thr_clean).astype(int)
            rows.append({
                "seed": seed, "scenario": scenario, "model_id": model_id,
                "model": MODEL_LABELS[model_id], "epsilon": eps,
                "auprc_dp": r["auprc"], "f1_dp_recomputed": r["f1"],
                "f1_dp_cleanthr": compute_f1_at_fixed_threshold(r["y_test"], alarms_fixed),
                "fp_dp_recomputed": r["fp"],
                "fp_dp_cleanthr": int(np.sum((r["y_test"] == 0) & (alarms_fixed == 1))),
            })
    return rows


def run_phase_b(out, seeds, workers, scen_root):
    units = [(s, sc, str(scen_root), str(out / "scores")) for s in seeds for sc in SCENARIOS]
    rows = []
    with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_phase_b_unit, units)):
            rows.extend(chunk)
            if (i + 1) % 10 == 0:
                pd.DataFrame(rows).to_csv(out / "dp_sweep.csv", index=False)
                print(f"[B] {i + 1}/{len(units)} units", flush=True)
    pd.DataFrame(rows).sort_values(
        ["scenario", "model_id", "epsilon", "seed"]
    ).to_csv(out / "dp_sweep.csv", index=False)
    print("[B] done", flush=True)


# --------------------------------------------------------------------------
# Phase C: sensitivity (alpha 0.8 and 1.2) at the full seed count
# --------------------------------------------------------------------------

def _phase_c_unit(args):
    alpha, seed, cfg_path = args
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for scenario in SCENARIOS:
            generate_scenario(scenario, seed_override=seed, output_root=tmp,
                              config_path=cfg_path)
            frame = load_scenario(tmp, scenario)
            for model_id in CORE_MODELS:
                r = _evaluate_frame(frame, scenario, model_id)
                rows.append({
                    "alpha": alpha, "seed": seed, "scenario": scenario,
                    "model_id": model_id, "model": MODEL_LABELS[model_id],
                    "auprc": r["auprc"], "f1": r["f1"],
                })
    return rows


def run_phase_c(out, seeds, workers):
    cfg_dir = out / "sens_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    units = []
    for alpha in (0.8, 1.2):
        cfg = cfg_dir / f"cfg_{alpha}.yaml"
        if not cfg.exists():
            write_perturbed_config(alpha, str(cfg))
        units.extend((alpha, s, str(cfg)) for s in seeds)
    rows = []
    with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_phase_c_unit, units)):
            rows.extend(chunk)
            if (i + 1) % 5 == 0:
                pd.DataFrame(rows).to_csv(out / "sensitivity_extra.csv", index=False)
                print(f"[C] {i + 1}/{len(units)} units", flush=True)
    pd.DataFrame(rows).to_csv(out / "sensitivity_extra.csv", index=False)
    print("[C] done", flush=True)


# --------------------------------------------------------------------------
# Phase E: hazard/h sweep, feature-set ablation, P04 window sweep
# --------------------------------------------------------------------------

def _phase_e_unit(args):
    kind, param, seed, scenario, scen_root = args
    seed_root = Path(scen_root) / f"seed-{seed}"
    frame = load_scenario(str(seed_root), scenario)
    rows = []
    if kind == "hazard":
        train_df, test_df = split_train_test(frame, scenario)
        train, test, y, _ = prepare_train_test_arrays(train_df, test_df)
        det = BocpdDetector(hazard_lambda=float(param))
        tr, te = det.fit_score(train, test)
        thr = compute_fixed_threshold_from_training_scores(np.asarray(tr))
        rows.append({"kind": kind, "param": param, "seed": seed, "scenario": scenario,
                     "model_id": "bocpd",
                     "auprc": float(average_precision_score(y, te)),
                     "f1": compute_f1_at_fixed_threshold(y, (np.asarray(te) >= thr).astype(int))})
    elif kind == "cusum_h":
        train_df, test_df = split_train_test(frame, scenario)
        train, test, y, _ = prepare_train_test_arrays(train_df, test_df)
        det = CusumDetector(h=param)
        tr, te = det.fit_score(train, test)
        thr = compute_fixed_threshold_from_training_scores(np.asarray(tr))
        rows.append({"kind": kind, "param": param, "seed": seed, "scenario": scenario,
                     "model_id": "cusum",
                     "auprc": float(average_precision_score(y, te)),
                     "f1": compute_f1_at_fixed_threshold(y, (np.asarray(te) >= thr).astype(int))})
    elif kind == "features":
        columns = FEATURE_SETS[param]
        train_df, test_df = split_train_test(frame, scenario)
        y = (test_df["label"] == "distress").astype(int).values
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(train_df[columns].values)
        train = scaler.transform(train_df[columns].values)
        test = scaler.transform(test_df[columns].values)
        for model_id in CORE_MODELS:
            trs, tes = _score_any(model_id, train, test, len(train_df))
            trs, tes = np.asarray(trs, float), np.asarray(tes, float)
            thr = compute_fixed_threshold_from_training_scores(trs)
            rows.append({"kind": kind, "param": param, "seed": seed,
                         "scenario": scenario, "model_id": model_id,
                         "auprc": float(average_precision_score(y, tes)),
                         "f1": compute_f1_at_fixed_threshold(y, (tes >= thr).astype(int))})
    elif kind == "window":
        for model_id in CORE_MODELS:
            r = _evaluate_frame(frame, scenario, model_id, train_days=param)
            rows.append({"kind": kind, "param": param, "seed": seed,
                         "scenario": scenario, "model_id": model_id,
                         "auprc": r["auprc"], "f1": r["f1"], "fp": r["fp"]})
    return rows


def run_phase_e(out, seeds, workers, scen_root):
    units = []
    for s in seeds:
        for sc in SCENARIOS:
            units.extend(("hazard", h, s, sc, str(scen_root)) for h in HAZARDS)
            units.extend(("cusum_h", h, s, sc, str(scen_root)) for h in CUSUM_HS)
            units.extend(("features", fs, s, sc, str(scen_root)) for fs in FEATURE_SETS)
        units.extend(("window", w, s, "P04", str(scen_root)) for w in P04_WINDOWS)
    rows = []
    with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_phase_e_unit, units)):
            rows.extend(chunk)
            if (i + 1) % 40 == 0:
                pd.DataFrame(rows).to_csv(out / "param_sweeps.csv", index=False)
                print(f"[E] {i + 1}/{len(units)} units", flush=True)
    pd.DataFrame(rows).to_csv(out / "param_sweeps.csv", index=False)
    print("[E] done", flush=True)


# --------------------------------------------------------------------------
# Phase F: tracemalloc incremental memory, data volume, correlations
# --------------------------------------------------------------------------

def run_phase_f(out, scen_root):
    import tracemalloc

    rows = []
    for seed in range(5):
        for scenario in SCENARIOS:
            frame = load_scenario(str(Path(scen_root) / f"seed-{seed}"), scenario)
            train_df, test_df = split_train_test(frame, scenario)
            train, test, y, _ = prepare_train_test_arrays(train_df, test_df)
            for model_id in CORE_MODELS:
                tracemalloc.start()
                base = tracemalloc.get_traced_memory()[0]
                _score_any(model_id, train, test, len(train_df))
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                rows.append({"seed": seed, "scenario": scenario, "model_id": model_id,
                             "incremental_peak_kb": (peak - base) / 1024.0})
    pd.DataFrame(rows).to_csv(out / "tracemalloc_memory.csv", index=False)

    # Event volume from retained raw days (seeds 0-4): events/day and bytes/day
    vol = []
    for seed in range(5):
        for scenario in SCENARIOS:
            raw = Path(scen_root) / f"seed-{seed}" / scenario / "raw_days"
            if not raw.exists():
                continue
            files = list(raw.glob("*.csv"))
            n_events = 0
            n_bytes = 0
            for f in files:
                with open(f, "rb") as handle:
                    content = handle.read()
                n_bytes += len(content)
                n_events += max(0, content.count(b"\n") - 1)
            if files:
                vol.append({"seed": seed, "scenario": scenario,
                            "days": len(files),
                            "events_per_day": n_events / len(files),
                            "bytes_per_day": n_bytes / len(files)})
    pd.DataFrame(vol).to_csv(out / "data_volume.csv", index=False)

    # Baseline-day feature correlation matrix (per advisor 9.6)
    frames = []
    for seed in range(5):
        for scenario in SCENARIOS:
            frame = load_scenario(str(Path(scen_root) / f"seed-{seed}"), scenario)
            frames.append(frame[frame.severity <= 0.1][MODEL_INPUT_FEATURES])
    corr = pd.concat(frames).corr()
    corr.to_csv(out / "feature_correlation_baseline.csv")
    print("[F] done", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Tier-2 experiment runner")
    parser.add_argument("--phase", required=True, choices=list("ABCEF"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("outputs/tier2"))
    parser.add_argument("--scenario-root", type=Path, default=Path("outputs/tier2/scenarios"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    if args.phase == "A":
        run_phase_a(args.output, seeds, args.workers, args.scenario_root)
    elif args.phase == "B":
        run_phase_b(args.output, seeds, args.workers, args.scenario_root)
    elif args.phase == "C":
        run_phase_c(args.output, seeds, args.workers)
    elif args.phase == "E":
        run_phase_e(args.output, seeds, args.workers, args.scenario_root)
    elif args.phase == "F":
        run_phase_f(args.output, args.scenario_root)


if __name__ == "__main__":
    main()
