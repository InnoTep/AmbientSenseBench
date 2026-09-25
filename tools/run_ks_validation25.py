"""Regenerate the 25-seed scenario set and re-run the KS distributional validation.

Restores the provenance of the paper's Table `tab:validation` (200 scenario-seed-state
groups = 4 scenarios x 2 steady states x 25 seeds). Generation is deterministic per
(scenario, seed), so this reproduces the original tier-2 scenario set exactly.

Usage:  python tools/run_ks_validation25.py  (from the repository root)
Outputs: outputs/tier2/ks_validation25/ks_seed-<N>.csv   (per-seed group rows)
         outputs/tier2/ks_validation25/ks_per_group.csv  (all 200 groups x 9 features)
         outputs/tier2/ks_validation25/ks_feature_summary.csv (paper-table aggregation)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

ROOT = Path("outputs/tier2/scenarios")
OUT = Path("outputs/tier2/ks_validation25")
CONFIG = Path("src/ambientsensebench/config/profiles.yaml")
SEEDS = range(25)
SCENARIOS = ("P01", "P02", "P03", "P04")


def _gen_unit(args):
    seed, scenario = args
    from ambientsensebench.generate_scenarios import generate_scenario

    seed_root = ROOT / f"seed-{seed}"
    feat = seed_root / scenario / "daily_features.csv"
    if not feat.exists():
        generate_scenario(scenario, seed_override=seed, output_root=str(seed_root))
    # Raw event days are only needed for the data-volume analysis (seeds 0-4),
    # mirroring tier2_experiments phase A.
    raw = seed_root / scenario / "raw_days"
    if seed > 4 and raw.exists():
        shutil.rmtree(raw, ignore_errors=True)
    return (seed, scenario)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    units = [(s, sc) for s in SEEDS for sc in SCENARIOS]
    with get_context("spawn").Pool(2) as pool:
        for i, unit in enumerate(pool.imap_unordered(_gen_unit, units), 1):
            print(f"[gen {i}/{len(units)}] seed={unit[0]} {unit[1]}", flush=True)

    frames = []
    for seed in SEEDS:
        out_csv = OUT / f"ks_seed-{seed}.csv"
        if not out_csv.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ambientsensebench.validate_scenarios_ks",
                    "--scenario-dir",
                    str(ROOT / f"seed-{seed}"),
                    "--config",
                    str(CONFIG),
                    "--output",
                    str(out_csv),
                ],
                check=True,
            )
        df = pd.read_csv(out_csv)
        df["seed"] = seed
        frames.append(df)
        print(f"[ks {seed + 1}/25] done", flush=True)

    allg = pd.concat(frames, ignore_index=True)
    allg.to_csv(OUT / "ks_per_group.csv", index=False)

    def passes(row) -> bool:
        # Paper convention (Sec. 3.5): moment-validated features pass on BOTH moments
        # (mean and std relative error <= 0.20); Gaussian features keep the script's
        # KS-based category.
        if row["validation_strategy"] == "moment-only":
            return (row["relative_error"] <= 0.20) and (row["std_relative_error"] <= 0.20)
        return row["validation_category"] in ("Strong", "Acceptable")

    rows = []
    for feature, g in allg.groupby("feature", sort=False):
        passed = g.apply(passes, axis=1)
        row = {
            "feature": feature,
            "n_groups": len(g),
            "pass_rate_pct": 100.0 * passed.mean(),
            "validation_strategy": g["validation_strategy"].iloc[0],
            "mean_rel_error": g["relative_error"].mean(),
        }
        ks = g["ks_statistic"].dropna()
        row["mean_ks"] = ks.mean() if len(ks) else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "ks_feature_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
