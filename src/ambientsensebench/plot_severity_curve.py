"""Score--severity curve on the progressive P04 trajectory (paper Figure 4).

Mean within-run score percentile of each core detector per tenth of the simulated
severity, read from ``severity_curve_p04.csv`` (``paper_analyses --analysis severity``)
and drawn at the bin midpoints. Colours and markers match the other benchmark figures;
fonts are embedded as TrueType.
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
import pandas as pd

from .benchmark import MODEL_LABELS
from .paper_analyses import SEVERITY_BINS
from .plot_benchmark import MODEL_COLORS
from .plot_privacy_curve import MODEL_MARKERS


def create_severity_curve(curve_path: Path, output_directory: Path,
                          name: str = "severity_consistency") -> Path:
    curve = pd.read_csv(curve_path, index_col=0)
    midpoints = (curve.index.to_numpy() + 0.5) / SEVERITY_BINS
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for model_id in MODEL_LABELS:
            ax.plot(midpoints, curve[model_id], color=MODEL_COLORS[model_id],
                    marker=MODEL_MARKERS[model_id], markersize=4.5, linewidth=1.6,
                    label=MODEL_LABELS[model_id])
        ax.set_xlabel("Simulated severity (bin midpoint)")
        ax.set_ylabel("Mean within-run score percentile")
        ax.set_title("P04: detector score against underlying severity")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        fig.tight_layout()

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
    path = create_severity_curve(args.input / "severity_curve_p04.csv", args.output)
    print(f"Saved: {path}")
