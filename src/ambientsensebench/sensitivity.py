"""Sensitivity of the benchmark conclusions to the distress-profile magnitude.

The distress profile parameters are illustrative rather than calibrated to
clinical measurements. To show that the qualitative conclusions do not depend on
the exact magnitudes, we scale the separation between the baseline and distress
profiles by a factor ``alpha`` (distress' = baseline + alpha * (distress -
baseline)), regenerate every scenario, and re-run the detectors. A robust
benchmark keeps the same detector orderings across a range of ``alpha``.

Run (slow; regenerates scenarios):

    python -m ambientsensebench.sensitivity --alphas 0.7,0.85,1.0,1.15,1.3 \
        --seeds 0,1,2 --output outputs/paper
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from .benchmark import DEFAULT_MODELS, evaluate_detector
from .generate_raw_data import load_profiles
from .generate_scenarios import (
    DEFAULT_CONFIG_PATH,
    SCENARIOS,
    build_interpolated_profile,
    generate_scenario,
)
from .wp6_evaluation import load_scenario


def write_perturbed_config(alpha: float, out_path: str) -> None:
    """Write a profiles config whose distress profile is scaled by ``alpha``."""
    cfg = load_profiles(DEFAULT_CONFIG_PATH)
    baseline = cfg["baseline"]
    distress = cfg["distress"]
    perturbed = build_interpolated_profile(baseline, distress, alpha)
    new_cfg = dict(cfg)
    new_cfg["distress"] = perturbed
    with open(out_path, "w") as handle:
        yaml.safe_dump(new_cfg, handle, sort_keys=False)


def run_sensitivity(alphas, seeds, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for alpha in alphas:
            cfg_path = os.path.join(temp_dir, f"cfg_{alpha}.yaml")
            write_perturbed_config(alpha, cfg_path)
            for seed in seeds:
                root = os.path.join(temp_dir, f"a{alpha}_s{seed}")
                for scenario in SCENARIOS:
                    generate_scenario(
                        scenario,
                        seed_override=seed,
                        output_root=root,
                        config_path=cfg_path,
                    )
                    frame = load_scenario(root, scenario)
                    for model_id in DEFAULT_MODELS:
                        result = evaluate_detector(frame, scenario, model_id, seed)
                        rows.append(
                            {
                                "alpha": alpha,
                                "seed": seed,
                                "scenario": scenario,
                                "model": result["model"],
                                "auprc": result["auprc"],
                                "f1": result["f1_fixed_threshold"],
                            }
                        )
            # checkpoint after each alpha so progress is inspectable
            pd.DataFrame(rows).to_csv(output_dir / "sensitivity_per_run.csv", index=False)
            print(f"[alpha={alpha}] done ({len(rows)} rows)", flush=True)
    return pd.DataFrame(rows)


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    means = per_run.groupby(["alpha", "scenario", "model"])[["auprc", "f1"]].mean().reset_index()
    out = []
    for (alpha, scenario), grp in means.groupby(["alpha", "scenario"]):
        best_auprc = grp.loc[grp.auprc.idxmax(), "model"]
        best_f1 = grp.loc[grp.f1.idxmax(), "model"]
        out.append(
            {
                "alpha": alpha,
                "scenario": scenario,
                "best_auprc_model": best_auprc,
                "best_f1_model": best_f1,
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile-magnitude sensitivity study.")
    parser.add_argument("--alphas", default="0.7,0.85,1.0,1.15,1.3")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output", type=Path, default=Path("outputs/paper"))
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    per_run = run_sensitivity(alphas, seeds, args.output)
    summary = summarise(per_run)
    summary.to_csv(args.output / "sensitivity_summary.csv", index=False)
    print("SENSITIVITY SUMMARY")
    print(summary.to_string(index=False))
    # Invariance verdict
    f1_ok = (summary.best_f1_model == "Isolation Forest").all()
    print(f"\nIsolation Forest best F1 in EVERY (alpha, scenario)?: {bool(f1_ok)}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
