"""Counterfactual for the prodrome that P02 carries into its training window.

P02's sustained episode rises slowly (time constant 7 days), so its severity first exceeds
0.01 on day 12 and 28 of the 40 baseline-only training days already carry part of the
distress shift, although none reaches the binary label. This script regenerates P02 with
severity forced to zero on the training days and re-runs the eight detectors under the main
protocol. Every day is reseeded independently, so the 325 test days are identical to the
main run and the comparison isolates the effect of the drifted training window.

Usage:  python tools/p02_prodrome_check.py  (from the repository root, after phase A)
Outputs: outputs/tier2/p02_clean_training.csv          (per seed and detector)
         outputs/tier2/p02_clean_training_summary.csv  (means, main run against clean window)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambientsensebench import generate_scenarios  # noqa: E402
from ambientsensebench.tier2_experiments import ALL_MODELS, _evaluate_frame, _init_worker  # noqa: E402
from ambientsensebench.wp6_evaluation import TRAIN_DAYS, load_scenario  # noqa: E402

SCENARIO = "P02"
_original_severity = generate_scenarios.get_scenario_severity


def _clean_training_severity(scenario_id, day_index):
    # day_index is 1-based in the generator; rows 0..TRAIN_DAYS-1 are the training window
    if scenario_id == SCENARIO and day_index <= TRAIN_DAYS:
        return 0.0
    return _original_severity(scenario_id, day_index)


def _unit(seed):
    generate_scenarios.get_scenario_severity = _clean_training_severity
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            generate_scenarios.generate_scenario(SCENARIO, seed_override=seed, output_root=tmp)
        frame = load_scenario(tmp, SCENARIO)
        for model_id, label in ALL_MODELS.items():
            r = _evaluate_frame(frame, SCENARIO, model_id, seed)
            rows.append({
                "seed": seed, "scenario": SCENARIO, "model_id": model_id, "model": label,
                "auprc": r["auprc"], "f1_fixed_threshold": r["f1"],
                "fixed_threshold": r["threshold"], "n_false_alarms": r["fp"],
                "n_true_alarms": r["tp"],
                "mean_lead_time_days": r["lead"]["mean_lead_time_days"],
                "n_no_warning": r["lead"]["n_no_warning"],
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--main", type=Path, default=Path("outputs/tier2/per_run_main.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tier2"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for chunk in pool.imap_unordered(_unit, range(args.seeds)):
            rows.extend(chunk)
    clean = pd.DataFrame(rows).sort_values(["model_id", "seed"])
    clean.to_csv(args.output / "p02_clean_training.csv", index=False)

    metrics = ["auprc", "f1_fixed_threshold", "n_false_alarms", "n_true_alarms",
               "mean_lead_time_days", "n_no_warning"]
    main_run = pd.read_csv(args.main)
    main_run = main_run[(main_run.scenario == SCENARIO) & (main_run.seed < args.seeds)]
    summary = (
        main_run.groupby("model")[metrics].mean().add_suffix("_main")
        .join(clean.groupby("model")[metrics].mean().add_suffix("_clean"))
    )
    summary.to_csv(args.output / "p02_clean_training_summary.csv")
    with pd.option_context("display.width", 200):
        print(summary.round(3).T.to_string())


if __name__ == "__main__":
    main()
