"""Publication-oriented figures generated from benchmark summary tables."""

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
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# Positive-class prevalence per scenario. Fixed by each scenario's construction
# and identical across seeds (verified over seeds 0-4); equals the AUPRC of a
# random ranking, so it is drawn as the chance baseline on the AUPRC panel.
SCENARIO_PREVALENCE = {
    "P01": 31 / 325,
    "P02": 136 / 325,
    "P03": 62 / 325,
    "P04": 259 / 325,
}
PREVALENCE_STYLE = {"color": "#3a3a3a", "linestyle": (0, (4, 2.4)), "linewidth": 1.1}

# The figure is downscaled to text width in the manuscript, so the hatch
# marking false-alarm segments must stay visible after ~2x reduction and
# at screen zoom: use a dense pattern with a heavier hatch line.
matplotlib.rcParams["hatch.linewidth"] = 1.2
FALSE_ALARM_HATCH = "///"

from .benchmark import MODEL_LABELS

# Categorical palette in fixed slot order, validated for adjacent-pair
# colour-vision-deficiency separation and normal-vision separability
# (worst adjacent CVD dE 9.1, worst adjacent normal-vision dE 19.6 on the
# light surface). Detector identity is additionally carried by the fixed
# bar order within each scenario group and by the legend, and every
# plotted value also appears in the manuscript tables and result CSVs.
MODEL_COLORS = {
    "isolation_forest": "#2a78d6",
    "one_class_svm": "#eb6834",
    "local_outlier_factor": "#1baf7a",
    "bocpd": "#eda100",
    "cusum": "#e87ba4",
}
ERRORBAR_STYLE = {
    "fmt": "none",
    "ecolor": "#3a3a3a",
    "elinewidth": 0.9,
    "capsize": 2.0,
    "capthick": 0.9,
}


def _draw_error_bars(axis, positions, values, errors) -> None:
    """Draw standard-deviation bars only where the deviation is non-zero.

    A zero-length error bar still renders its caps, which appear as stray
    black ticks on top of deterministic bars (for example lead times at
    the evaluation-window cap). Masking zero and undefined deviations
    removes the artefact without hiding any real variability.
    """
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    mask = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    if mask.any():
        axis.errorbar(
            np.asarray(positions, dtype=float)[mask],
            values[mask],
            yerr=errors[mask],
            **ERRORBAR_STYLE,
        )


def _model_order(summary: pd.DataFrame) -> list[str]:
    return [model_id for model_id in MODEL_LABELS if model_id in set(summary["model_id"])]


def _scenario_order(summary: pd.DataFrame) -> list[str]:
    return sorted(summary["scenario"].unique())


def _values_for_model(
    summary: pd.DataFrame,
    scenario_ids: list[str],
    model_id: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    subset = summary.loc[summary["model_id"] == model_id].set_index("scenario")
    values = subset.reindex(scenario_ids)[f"{metric}_mean"].to_numpy(dtype=float)
    errors = subset.reindex(scenario_ids)[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
    return values, errors


def _grouped_bars(axis, summary: pd.DataFrame, metric: str, label: str) -> None:
    scenario_ids = _scenario_order(summary)
    model_ids = _model_order(summary)
    positions = np.arange(len(scenario_ids))
    width = 0.72 / len(model_ids)

    for index, model_id in enumerate(model_ids):
        values, errors = _values_for_model(summary, scenario_ids, model_id, metric)
        offset = (index - (len(model_ids) - 1) / 2) * width
        axis.bar(
            positions + offset,
            values,
            width=width,
            color=MODEL_COLORS[model_id],
            label=MODEL_LABELS[model_id],
        )
        _draw_error_bars(axis, positions + offset, values, errors)

    axis.set_xticks(positions, scenario_ids)
    axis.set_ylabel(label)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)


def create_benchmark_figures(summary_path: Path, output_directory: Path) -> dict[str, Path]:
    """Create performance and alarm-burden figures from a benchmark summary CSV."""
    summary = pd.read_csv(summary_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    performance_figure, performance_axes = plt.subplots(1, 2, figsize=(10, 4.4))
    performance_figure.subplots_adjust(top=0.74, bottom=0.18, wspace=0.20)
    _grouped_bars(performance_axes[0], summary, "auprc", "AUPRC")
    performance_axes[0].set_ylim(0, 1.05)
    performance_axes[0].set_title("Ranking performance")
    # Chance baseline: a random ranking attains AUPRC equal to the positive-class
    # prevalence, which differs by design across scenarios. Drawn as a short
    # dashed segment spanning each scenario's bar group.
    scenario_ids = _scenario_order(summary)
    group_half_width = 0.72 / 2 + 0.05
    for index, scenario_id in enumerate(scenario_ids):
        prevalence = SCENARIO_PREVALENCE.get(scenario_id)
        if prevalence is not None:
            performance_axes[0].hlines(
                prevalence,
                index - group_half_width,
                index + group_half_width,
                zorder=4,
                **PREVALENCE_STYLE,
            )
    _grouped_bars(performance_axes[1], summary, "f1_fixed_threshold", "Fixed-threshold F1")
    performance_axes[1].set_ylim(0, 1.05)
    performance_axes[1].set_title("Operational alarm performance")
    handles, labels = performance_axes[0].get_legend_handles_labels()
    handles = [h for h, l in zip(handles, labels) if l in MODEL_LABELS.values()]
    labels = [l for l in labels if l in MODEL_LABELS.values()]
    handles.append(Line2D([0], [0], **PREVALENCE_STYLE))
    labels.append("Prevalence (random-ranking AUPRC)")
    performance_axes[0].legend().remove()
    performance_axes[1].legend().remove()
    performance_figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncols=3,
        frameon=False,
    )

    performance_path = output_directory / "benchmark_performance.pdf"
    performance_figure.savefig(performance_path, bbox_inches="tight")
    performance_figure.savefig(performance_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(performance_figure)

    alarm_figure, alarm_axes = plt.subplots(1, 2, figsize=(10, 4.2))
    alarm_figure.subplots_adjust(top=0.88, bottom=0.28, wspace=0.20)
    scenario_ids = _scenario_order(summary)
    model_ids = _model_order(summary)
    positions = np.arange(len(scenario_ids))
    width = 0.72 / len(model_ids)
    for index, model_id in enumerate(model_ids):
        offset = (index - (len(model_ids) - 1) / 2) * width
        true_alarms, _ = _values_for_model(summary, scenario_ids, model_id, "n_true_alarms")
        false_alarms, _ = _values_for_model(summary, scenario_ids, model_id, "n_false_alarms")
        # False-alarm segment: light series tint with a series-colour
        # border for identity, plus a separate overlay that draws the
        # hatch in dark neutral ink. Dark hatch on a light tint stays
        # legible at any zoom and matches the legend key, whereas
        # same-hue hatch on a same-hue tint washes out when the figure
        # is downscaled (hatch colour follows the patch edgecolor, so
        # the overlay carries the dark edgecolor with a zero-width
        # outline).
        alarm_axes[0].bar(
            positions + offset,
            false_alarms,
            width=width,
            bottom=true_alarms,
            facecolor=to_rgba(MODEL_COLORS[model_id], 0.25),
            edgecolor=MODEL_COLORS[model_id],
            linewidth=0.6,
        )
        alarm_axes[0].bar(
            positions + offset,
            false_alarms,
            width=width,
            bottom=true_alarms,
            facecolor="none",
            edgecolor="#3a3a3a",
            linewidth=0.0,
            hatch=FALSE_ALARM_HATCH,
        )
        alarm_axes[0].bar(
            positions + offset,
            true_alarms,
            width=width,
            color=MODEL_COLORS[model_id],
        )
    alarm_axes[0].set_xticks(positions, scenario_ids)
    alarm_axes[0].set_ylabel("Mean alerts")
    alarm_axes[0].set_title("Alarm composition")
    alarm_axes[0].grid(axis="y", alpha=0.25)
    alarm_axes[0].set_axisbelow(True)
    _grouped_bars(alarm_axes[1], summary, "mean_lead_time_days", "Mean lead time (days)")
    alarm_axes[1].set_title("Episode-level lead time")
    # Lead time is searched within a bounded pre-onset window; bars at the
    # bound are censored there, not measured. Mark the bound explicitly.
    alarm_axes[1].axhline(14, linestyle=":", linewidth=0.9, color="#8a8a8a", zorder=1)
    alarm_axes[1].set_ylim(0, 15.4)
    alarm_axes[1].text(
        0.99,
        14.25,
        "14-day search-window cap",
        transform=alarm_axes[1].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#6b6b6b",
    )
    alarm_axes[1].legend().remove()
    alarm_figure.legend(
        [
            *(Patch(facecolor=MODEL_COLORS[model_id], label=MODEL_LABELS[model_id]) for model_id in model_ids),
            Patch(facecolor="white", edgecolor="black", hatch=FALSE_ALARM_HATCH, label="False-alarm segment"),
        ],
        [*(MODEL_LABELS[model_id] for model_id in model_ids), "False-alarm segment"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncols=2,
        frameon=False,
    )

    alarm_path = output_directory / "benchmark_alarm_burden.pdf"
    alarm_figure.savefig(alarm_path, bbox_inches="tight")
    alarm_figure.savefig(alarm_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(alarm_figure)

    return {
        "performance": performance_path,
        "alarm_burden": alarm_path,
    }
