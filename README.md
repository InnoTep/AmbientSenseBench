# AmbientSenseBench

AmbientSenseBench generates synthetic smart-home sensor events and evaluates anomaly
detectors on four routine-change scenarios. It is intended for research use.

The data and labels are simulated. They are not clinical records, diagnoses, or evidence of
clinical performance. Do not use the benchmark for:

- any clinical claim (the profiles are literature assumptions, not patient data);
- training a model that is deployed without validation on real data;
- claiming the absolute performance of a detector (only relative comparison under the
  protocol is supported);
- anything at the level of an individual.

## Setup

Python 3.10 or later is required.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Main commands

Generate a scenario:

```bash
python -m ambientsensebench generate P01 --seed 42 --output outputs
```

Generate all scenarios by replacing `P01` with `all`. Each scenario directory contains raw
event files, `daily_features.csv`, and `scenario_labels.csv`.

Run the tests:

```bash
pytest
```

Run a small benchmark during development:

```bash
python -m ambientsensebench benchmark --seeds 7 --scenarios P01 --output outputs/smoke
```

Run the standard benchmark and create figures:

```bash
python -m ambientsensebench benchmark --output outputs/benchmark
python -m ambientsensebench figures --input outputs/benchmark/summary_results.csv
```

This standard run uses five seeds, four scenarios, and five detectors, and writes per-run
and summary tables plus the command configuration. It is a development-scale check. The
results reported in the paper come from the 25-seed reference run of the
`tier2_experiments` and `stats_tests` modules;
[`docs/reproducibility.md`](docs/reproducibility.md) maps every table and figure in the
paper to the command that produces it.

## Reproducing the paper

Run from the repository root. The 25-seed reference run comes first, because the other
commands reuse its scenarios and stored scores (about two and a half hours with two
workers); [`docs/reproducibility.md`](docs/reproducibility.md) gives the output file of each
step and the full mapping.

```bash
for phase in A B C D; do python -m ambientsensebench.tier2_experiments --phase $phase --workers 2 --seeds 25; done
python -m ambientsensebench.tier2_experiments --phase E
python -m ambientsensebench.stats_tests --input outputs/tier2
python -m ambientsensebench.paper_analyses --analysis all --workers 2
```

| Paper item | Command (after the reference run) |
| --- | --- |
| Figure 1, system architecture | `python tools/architecture_figure.py` |
| Figures 2 and 3, performance and alarm burden | `python -m ambientsensebench figures --input outputs/tier2/summary_core25.csv --output outputs/tier2/figures` (the summary table is written by `paper_analyses`) |
| Table 2, reference baselines; Table 4, private release; Table 5, threshold ablation | means of `per_run_main.csv` and `dp_sweep.csv` by scenario and detector |
| Figure 4, score-severity consistency | `paper_analyses --analysis severity` (included above) |
| Table 3, distributional validation | `python tools/run_ks_validation25.py` |
| Figure 5, privacy-utility curves | `python -m ambientsensebench.plot_privacy_curve --input outputs/tier2 --output outputs/tier2/figures` |
| Table 6, computational cost | `python -m ambientsensebench benchmark --output outputs/benchmark`, then `python -m ambientsensebench.paper_extras --scenario-root outputs/benchmark/scenarios --seeds 0,1,2,3,4 --output outputs/paper` |
| Figure 6, difficulty curve over alpha | `python -m ambientsensebench.tier2_experiments --phase C --alphas 0.4,0.6 --seeds 25 --output outputs/tier2-alpha`, then `python -m ambientsensebench.plot_alpha_curve --sensitivity outputs/tier2/sensitivity_extra.csv outputs/tier2-alpha/sensitivity_extra.csv` |
| P02 training-window counterfactual (Section 5.2) | `python tools/p02_prodrome_check.py` |
| Plausibility check against CASAS Aruba | `python -m ambientsensebench.real_comparison --aruba datasets/CASAS_Aruba/data/aruba.csv` |
| Table 8, semi-synthetic injection on Aruba | `python tools/aruba_injection.py --aruba datasets/CASAS_Aruba/data/aruba.csv` |
| Appendix Table A.1 and all paired tests | `stats_tests` (included above) |

The Aruba file is `data/aruba.csv` inside `data.zip` of the CASAS archive
(<https://doi.org/10.5281/zenodo.15708568>, CC BY 4.0); it is not redistributed here.

## Severity-timeline app and explorer

The data-generation app composes a custom severity timeline (episode or monotone change,
with onset, offset, transition smoothness, peak severity, length, and seed) over the
benchmark's home and profiles, optionally applies the Laplace release, shows the trajectory
and the detector's alerts, and downloads the generated artefacts. The explorer browses
generated scenarios.

```bash
python -m ambientsensebench app
python -m ambientsensebench explore
```

Both serve on port 8765 by default; [`docs/app.md`](docs/app.md) documents the API and a
Docker image.

## Documentation

- [`docs/architecture.md`](docs/architecture.md): package layout and protocol rules.
- [`docs/benchmark-protocol.md`](docs/benchmark-protocol.md): benchmark design.
- [`docs/generation-mechanism.md`](docs/generation-mechanism.md): how events are generated.
- [`docs/reproducibility.md`](docs/reproducibility.md): paper tables and figures mapped to commands.
- [`docs/app.md`](docs/app.md): app API and Docker instructions.
- [`docs/data-card.md`](docs/data-card.md): generated-data scope and limitations.

Generated outputs, external datasets, caches, build products, and manuscript material are
not versioned. The code is released under the Apache License 2.0 (see
[`LICENSE`](LICENSE)). Cite it with the metadata in [`CITATION.cff`](CITATION.cff);
[`CONTRIBUTORS.md`](CONTRIBUTORS.md) records the history of the code base.
