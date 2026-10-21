"""Privacy-utility curves over the epsilon sweep (advisor 5.2.4; paper Figure 4).

Two rows by one panel per scenario: mean AUPRC (top) and mean fixed-threshold
F1 (bottom) under the Laplace release as a function of epsilon on a log axis,
per core detector, with the clean-data value drawn as a short dash at the
right edge for reference.  Colours reuse the validated categorical palette of
the benchmark figures; per-series markers add a redundant channel for
colour-vision deficiency.  Fonts are embedded as TrueType so that text,
including the epsilon symbol, stays extractable from the PDF.
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

from .benchmark import MODEL_LABELS
from .plot_benchmark import MODEL_COLORS

MODEL_MARKERS = {
    "isolation_forest": "o",
    "one_class_svm": "s",
    "local_outlier_factor": "^",
    "bocpd": "D",
    "cusum": "v",
}


PRIVACY_ROWS = (
    ("auprc_dp", "auprc", "AUPRC under release"),
    ("f1_dp_recomputed", "f1_fixed_threshold", "F1 under release"),
)


def create_privacy_curve(dp_sweep_path: Path, per_run_main_path: Path,
                         output_directory: Path, rows=PRIVACY_ROWS,
                         name: str = "privacy_utility_curve") -> Path:
    """One row per (noised metric, clean metric, axis label) in ``rows``."""
    sweep = pd.read_csv(dp_sweep_path)
    clean = pd.read_csv(per_run_main_path)
    clean = clean[clean.model_id.isin(MODEL_LABELS)]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    scenarios = sorted(sweep.scenario.unique())
    with plt.rc_context({"pdf.fonttype": 42}):
        fig, axes = plt.subplots(len(rows), len(scenarios),
                                 figsize=(12.5, 3.0 * len(rows) + 0.4),
                                 sharex=True, sharey=True, squeeze=False)
        for (metric, clean_metric, ylabel), row_axes in zip(rows, axes):
            for ax, scenario in zip(row_axes, scenarios):
                sub = sweep[sweep.scenario == scenario]
                for model_id in MODEL_LABELS:
                    m = (sub[sub.model_id == model_id]
                         .groupby("epsilon")[metric].mean().sort_index())
                    ax.plot(m.index, m.values, color=MODEL_COLORS[model_id],
                            marker=MODEL_MARKERS[model_id], markersize=4.5,
                            linewidth=1.6, label=MODEL_LABELS[model_id])
                    clean_mean = clean[(clean.scenario == scenario)
                                       & (clean.model_id == model_id)][clean_metric].mean()
                    ax.plot([sub.epsilon.max() * 1.35, sub.epsilon.max() * 2.4],
                            [clean_mean, clean_mean], color=MODEL_COLORS[model_id],
                            linewidth=1.4, linestyle=(0, (2, 1.5)))
                ax.set_xscale("log")
                ax.grid(alpha=0.25)
                ax.set_axisbelow(True)
                ax.set_ylim(0, 1.05)
            row_axes[0].set_ylabel(ylabel)
        for ax, scenario in zip(axes[0], scenarios):
            ax.set_title(scenario)
        for ax in axes[-1]:
            ax.set_xlabel(r"$\epsilon$ (per feature-day)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color="#3a3a3a", linestyle=(0, (2, 1.5))))
        labels.append("Clean data (no release)")
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncols=6, frameon=False, fontsize=9)
        fig.subplots_adjust(top=0.86, bottom=0.12, hspace=0.12, wspace=0.08)

        out = output_directory / f"{name}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/tier2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tier2/figures"))
    args = parser.parse_args()
    path = create_privacy_curve(args.input / "dp_sweep.csv",
                                args.input / "per_run_main.csv", args.output)
    print(f"Saved: {path}")
