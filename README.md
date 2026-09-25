# AmbientSenseBench

AmbientSenseBench generates synthetic smart-home sensor events and evaluates anomaly
detectors on four routine-change scenarios. It is intended for research use.

The data and labels are simulated. They are not clinical records, diagnoses, or evidence of
clinical performance.

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

Start the local explorer or data-generation app:

```bash
python -m ambientsensebench explore
python -m ambientsensebench app
```

Both use port 8765 by default.

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
