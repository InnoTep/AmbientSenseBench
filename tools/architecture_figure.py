"""Draw the system-architecture figure of the paper (Figure 1).

Shows the pipeline the benchmark reproduces and where each part runs in the intended
deployment: events and daily features stay on the home gateway; a detector either runs
there, or, when features must leave the home, on the receiving side after the
per-feature-day Laplace release. Labels and severity exist only in the benchmark and are
used for retrospective evaluation.

Usage:  python tools/architecture_figure.py  (writes outputs/figures/architecture.pdf and .png)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HOME_FILL, HOME_EDGE = "#eaf1fb", "#8fa9cc"
OUT_FILL, OUT_EDGE = "#fbf1e6", "#d4a373"
BOX_FILL, BOX_EDGE = "#ffffff", "#3a3a3a"
EVAL_FILL, EVAL_EDGE = "#f2f2f2", "#7a7a7a"
TEXT = "#1f1f1f"


def _region(ax, x, y, w, h, fill, edge, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fill, ec=edge, lw=0.9))
    ax.text(x + 1.2, y + h - 1.4, title, ha="left", va="top", fontsize=8.5,
            fontweight="bold", color=edge)


def _box(ax, x, y, w, h, title, body, fill=BOX_FILL, edge=BOX_EDGE, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fill, ec=edge, lw=0.9, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=7.5,
            fontweight="bold", color=TEXT)
    ax.text(x + w / 2, y + h - 4.3, body, ha="center", va="top", fontsize=6.7,
            color=TEXT, linespacing=1.25)


def _arrow(ax, start, end, dashed=False, label=None, label_xy=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, lw=0.9,
                                 color=TEXT, ls="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=6.6, color=TEXT,
                linespacing=1.2)


def draw(out_dir: Path) -> Path:
    with plt.rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(7.6, 4.1))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 55)
        ax.axis("off")

        _region(ax, 0.5, 12, 67, 42.5, HOME_FILL, HOME_EDGE,
                "Inside the home: sensors and edge gateway")
        _region(ax, 69, 12, 30.5, 42.5, OUT_FILL, OUT_EDGE, "Outside the home")

        _box(ax, 2, 17, 20.5, 31, "Event source",
             "in a deployment,\nambient sensors;\nin the benchmark,\nthe generator renders\n"
             "daily routine\nparameters, sampled\nfrom the profiles and\nseverity trajectory,\n"
             "as events")
        _box(ax, 26.5, 27, 18, 13, "Shared extractor", "nine daily\nfeatures")
        _box(ax, 47.5, 36, 18.5, 13, "Laplace release", "noise per feature-day\nat a nominal ε")
        _box(ax, 47.5, 16, 18.5, 13, "Local detector",
             "scores, threshold\n(95th percentile),\nalarms")
        _box(ax, 72.5, 36, 25, 13, "Remote detector",
             "same detectors and\nthreshold rule, fitted on\nthe released features")
        ax.text(84, 24.5, "Only perturbed daily\nfeatures leave the home;\n"
                "with local detection,\nno features do.", ha="center", va="center",
                fontsize=7, style="italic", color=OUT_EDGE, linespacing=1.3)

        _arrow(ax, (22.5, 33.5), (26.5, 33.5))
        _arrow(ax, (44.5, 36.5), (47.5, 42.5))
        _arrow(ax, (44.5, 30.5), (47.5, 22.5))
        _arrow(ax, (66, 42.5), (72.5, 42.5))

        # Benchmark-only evaluation band
        _box(ax, 0.5, 0.5, 99, 9, "Retrospective evaluation (benchmark only)",
             "simulated labels and severity, never released or perturbed: AUPRC, "
             "fixed-threshold F1, alarm burden, lead time", fill=EVAL_FILL, edge=EVAL_EDGE,
             dashed=True)
        _arrow(ax, (12.25, 17), (12.25, 9.5), dashed=True, label="labels and\nseverity",
               label_xy=(17.6, 14.5))
        _arrow(ax, (56.75, 16), (56.75, 9.5), dashed=True, label="scores, alarms",
               label_xy=(49.8, 14.1))
        _arrow(ax, (95, 36), (95, 9.5), dashed=True, label="scores,\nalarms",
               label_xy=(91, 32.2))

        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "architecture.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("outputs/figures"))
    print(f"Saved: {draw(parser.parse_args().output)}")
