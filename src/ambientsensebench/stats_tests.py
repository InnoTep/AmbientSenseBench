"""Paired statistical tests over benchmark seeds (advisor items 3.2-3.3).

For each scenario and metric, every detector is compared against the
best-mean detector for that cell with a two-sided Wilcoxon signed-rank test
over paired seeds.  P-values are Holm-corrected within each scenario x
metric family.  Effect size is the rank-biserial correlation; uncertainty
is a paired-difference bootstrap 95% confidence interval.

Also derives, from the stored per-day scores of the main run:
  * the F1-versus-threshold-percentile trace (advisor 4.4)
  * F1 under a three-consecutive-day persistence rule, aligning the F1
    alarm policy with the lead-time policy (advisor 4.5)

Usage:
    python -m ambientsensebench.stats_tests --input outputs/tier2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .wp6_evaluation import compute_f1_at_fixed_threshold

PERCENTILES = (80.0, 85.0, 90.0, 92.5, 95.0, 97.5, 99.0, 100.0)


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def rank_biserial(diff: np.ndarray) -> float:
    """Rank-biserial correlation for paired differences (matched-pairs)."""
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = pd.Series(np.abs(diff)).rank().to_numpy()
    favourable = ranks[diff > 0].sum()
    unfavourable = ranks[diff < 0].sum()
    total = ranks.sum()
    return float((favourable - unfavourable) / total)


def bootstrap_ci(diff: np.ndarray, n_boot: int = 20000, seed: int = 12345):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_tests(per_run: pd.DataFrame, metrics=("auprc", "f1_fixed_threshold")) -> pd.DataFrame:
    rows = []
    for scenario, sgroup in per_run.groupby("scenario"):
        for metric in metrics:
            wide = sgroup.pivot(index="seed", columns="model_id", values=metric)
            best = wide.mean().idxmax()
            cell = []
            for model in wide.columns:
                if model == best:
                    continue
                diff = (wide[best] - wide[model]).to_numpy()
                if np.allclose(diff, 0):
                    stat_p = 1.0
                else:
                    stat_p = float(wilcoxon(diff, zero_method="wilcox",
                                            alternative="two-sided").pvalue)
                lo, hi = bootstrap_ci(diff)
                cell.append({
                    "scenario": scenario, "metric": metric, "best_model": best,
                    "model": model, "mean_diff": float(diff.mean()),
                    "ci95_lo": lo, "ci95_hi": hi,
                    "rank_biserial": rank_biserial(diff),
                    "p_raw": stat_p, "n_seeds": int(len(diff)),
                })
            praw = np.array([c["p_raw"] for c in cell])
            for c, adj in zip(cell, holm(praw)):
                c["p_holm"] = float(adj)
                c["distinguishable_5pct"] = bool(adj < 0.05)
            rows.extend(cell)
    return pd.DataFrame(rows)


def score_derived_analyses(score_dir: Path, per_run: pd.DataFrame):
    """Percentile trace and persistence-rule F1 from stored per-day scores."""
    trace_rows, persist_rows = [], []
    for (seed, scenario, model), _ in per_run.groupby(["seed", "scenario", "model_id"]):
        z = np.load(score_dir / f"s{seed}_{scenario}_{model}.npz")
        train, test, y = z["train_scores"], z["test_scores"], z["y_test"]
        for pct in PERCENTILES:
            thr = float(np.percentile(train, pct))
            alarms = (test >= thr).astype(int)
            trace_rows.append({
                "seed": seed, "scenario": scenario, "model_id": model,
                "percentile": pct,
                "f1": compute_f1_at_fixed_threshold(y, alarms),
                "false_alarms": int(np.sum((y == 0) & (alarms == 1))),
            })
        # persistence rule: a day only alarms if it ends a run of >= 3
        # consecutive over-threshold days (same evidence rule as lead time)
        thr = float(np.percentile(train, 95.0))
        raw = (test >= thr).astype(int)
        run = 0
        persist = np.zeros_like(raw)
        for t, a in enumerate(raw):
            run = run + 1 if a else 0
            persist[t] = 1 if run >= 3 else 0
        persist_rows.append({
            "seed": seed, "scenario": scenario, "model_id": model,
            "f1_per_day": compute_f1_at_fixed_threshold(y, raw),
            "f1_persistence3": compute_f1_at_fixed_threshold(y, persist),
            "fp_per_day": int(np.sum((y == 0) & (raw == 1))),
            "fp_persistence3": int(np.sum((y == 0) & (persist == 1))),
        })
    return pd.DataFrame(trace_rows), pd.DataFrame(persist_rows)


def main():
    parser = argparse.ArgumentParser(description="Paired tests and score-derived analyses")
    parser.add_argument("--input", type=Path, default=Path("outputs/tier2"))
    args = parser.parse_args()

    per_run = pd.read_csv(args.input / "per_run_main.csv")
    tests = paired_tests(per_run)
    tests.to_csv(args.input / "stats_paired_tests.csv", index=False)

    trace, persist = score_derived_analyses(args.input / "scores", per_run)
    trace.to_csv(args.input / "threshold_percentile_trace.csv", index=False)
    persist.to_csv(args.input / "persistence_rule_f1.csv", index=False)
    print("paired tests:", len(tests), "| trace rows:", len(trace),
          "| persistence rows:", len(persist))


if __name__ == "__main__":
    main()
