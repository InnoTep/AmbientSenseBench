"""Difficulty curve over the profile-magnitude sweep.

Two rows (mean AUPRC, mean fixed-threshold F1) by one panel per scenario, as a
function of the factor ``alpha`` that scales the baseline-to-distress
separation.  The alpha = 1.0 point is the main run (phase A); the other points
come from phase C, which can be run at several alpha sets and written to
different directories, so any number of sensitivity tables can be combined.
Sequential detectors are drawn solid and memoryless ones dashed, so the
detector families stay distinguishable without colour; the AUPRC row marks
each scenario's positive-class prevalence (the AUPRC of a random ranking).

Run after phase A and phase C:

    python -m ambientsensebench.plot_alpha_curve --main outputs/tier2/per_run_main.csv \
        --sensitivity outputs/tier2/sensitivity_extra.csv \
        outputs/tier2-alpha/sensitivity_extra.csv --output outputs/tier2/figures
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "ambientsensebench-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchmark import MODEL_LABELS, SEQUENTIAL_MODELS
from .plot_benchmark import MODEL_COLORS, PAPER_STYLE, SCENARIO_PREVALENCE, TEXT_WIDTH_IN
from .plot_privacy_curve import MODEL_MARKERS

COLUMNS = ["alpha", "seed", "scenario", "model_id", "auprc", "f1"]


def load_alpha_runs(per_run_main_path: Path, sensitivity_paths) -> pd.DataFrame:
    """Combine the main run (alpha = 1.0) with phase-C sensitivity tables."""
    main = pd.read_csv(per_run_main_path).rename(columns={"f1_fixed_threshold": "f1"})
    main = main[main.model_id.isin(MODEL_LABELS)].assign(alpha=1.0)
    frames = [main[COLUMNS]]
    frames.extend(pd.read_csv(path)[COLUMNS] for path in sensitivity_paths)
    runs = pd.concat(frames, ignore_index=True)
    duplicated = runs.duplicated(["alpha", "seed", "scenario", "model_id"])
    if duplicated.any():
        raise ValueError("the same (alpha, seed, scenario, detector) appears twice")
    return runs


def create_alpha_curve(runs: pd.DataFrame, output_directory: Path,
                       name: str = "alpha_difficulty_curve") -> Path:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    means = runs.groupby(["scenario", "model_id", "alpha"])[["auprc", "f1"]].mean()

    with plt.rc_context(PAPER_STYLE):
        return _draw_alpha_curve(runs, means, output_directory, name)


def _draw_alpha_curve(runs, means, output_directory, name):
    scenarios = sorted(runs.scenario.unique())
    rows = (("auprc", "Mean AUPRC"), ("f1", "Mean fixed-threshold F1"))
    fig, axes = plt.subplots(len(rows), len(scenarios), figsize=(TEXT_WIDTH_IN, 3.8),
                             sharex=True, sharey=True)
    for (metric, ylabel), row_axes in zip(rows, axes):
        for ax, scenario in zip(row_axes, scenarios):
            for model_id in MODEL_LABELS:
                series = means.loc[(scenario, model_id), metric].sort_index()
                ax.plot(series.index, series.values, color=MODEL_COLORS[model_id],
                        marker=MODEL_MARKERS[model_id], markersize=3, linewidth=1.1,
                        linestyle="-" if model_id in SEQUENTIAL_MODELS else "--",
                        label=MODEL_LABELS[model_id])
            if metric == "auprc":
                ax.axhline(SCENARIO_PREVALENCE[scenario], color="#3a3a3a",
                           linewidth=0.9, linestyle=":")
                ax.set_title(scenario)
            ax.grid(alpha=0.25)
            ax.set_axisbelow(True)
            ax.set_ylim(0, 1.05)
        row_axes[0].set_ylabel(ylabel)
    ax.set_xticks(sorted(runs.alpha.unique()))
    ax.set_yticks(np.linspace(0, 1, 6))

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="#3a3a3a", linewidth=0.9, linestyle=":"))
    labels.append("Prevalence (random-ranking AUPRC)")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncols=3, frameon=False, handlelength=2.2, columnspacing=1.2)
    fig.supxlabel(r"Profile scale $\alpha$", fontsize=8, y=0.01)
    fig.subplots_adjust(top=0.82, bottom=0.13, hspace=0.12, wspace=0.16)

    out = output_directory / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Difficulty curve over alpha.")
    parser.add_argument("--main", type=Path, default=Path("outputs/tier2/per_run_main.csv"))
    parser.add_argument("--sensitivity", type=Path, nargs="+",
                        default=[Path("outputs/tier2/sensitivity_extra.csv")])
    parser.add_argument("--output", type=Path, default=Path("outputs/tier2/figures"))
    args = parser.parse_args()
    runs = load_alpha_runs(args.main, args.sensitivity)
    path = create_alpha_curve(runs, args.output)
    runs.to_csv(args.output / "alpha_curve_per_run.csv", index=False)
    print(f"Saved: {path}")
