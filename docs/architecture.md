# Architecture

The main data path is:

```text
profiles -> raw events -> daily features -> detector scores -> result tables
```

Event generation is feature-first: each day's feature values are sampled from the active
profile and then rendered as events for the shared extractor to read back. See
[`generation-mechanism.md`](generation-mechanism.md) for the per-feature mechanism.

| Files | Purpose |
| --- | --- |
| `config/profiles.yaml`, `generate_raw_data.py`, `generate_scenarios.py` | Profiles, seeded event generation, and scenario severity. |
| `feature_extraction.py`, `build_dataset.py` | Daily feature extraction and CSV construction. |
| `wp6_evaluation.py`, `benchmark.py`, `temporal_detectors.py` | Data split, scaling, detector execution, thresholds, and metrics. |
| `cli.py`, `server.py`, `explorer.py`, `custom_scenario.py`, `web/` | Command-line and local web interfaces. |
| `tier2_experiments.py`, `stats_tests.py`, `sensitivity.py`, `apply_wp7_dp_noise.py`, and related modules | Extended experiment code. |

## Benchmark path

1. `generate_scenario()` produces 365 daily event files from the profile configuration.
2. `build_daily_dataset()` produces the nine-feature daily table.
3. `load_scenario()` validates the table. The first 40 days form the training window.
4. `prepare_train_test_arrays()` fits the scaler on the training window only.
5. `evaluate_detector()` scores the test window. The alarm threshold is the 95th percentile
   of training scores.

## Adding a detector

A third-party detector joins the benchmark as a plugin, without changes to the package. A
plugin is any object with

```python
fit_score(train_data, test_data) -> (train_scores, test_scores)
```

on the standardised arrays (40 training days and the test days, nine features each, in
chronological order), where a larger score means a more anomalous day. Register it at the
top level of an importable module:

```python
# my_detectors.py
import numpy as np

from ambientsensebench.benchmark import register_detector


class MeanAbsDeviation:
    def fit_score(self, train_data, test_data):
        return np.abs(train_data).mean(axis=1), np.abs(test_data).mean(axis=1)


register_detector("mean_abs", "Mean |z|", lambda training_rows: MeanAbsDeviation())
```

The factory receives the number of training days. Pass the module with `--plugin` (it must
be importable, for example from the working directory or `PYTHONPATH`):

```bash
python -m ambientsensebench benchmark --plugin my_detectors --models isolation_forest,mean_abs
python -m ambientsensebench.tier2_experiments --phase A --plugin my_detectors --workers 2 --seeds 25
```

The plugin then runs under exactly the protocol of the built-in detectors: training-only
scaling, the 95th-percentile training threshold, AUPRC, fixed-threshold F1, and episode
lead time. In `tier2_experiments` it joins phases A to E (the main run, the privacy sweep,
the profile sweep, the feature-set and training-window ablations, and the memory profile);
only the hazard and decision-interval sweeps, which belong to BOCPD and CUSUM, leave it out. Ids already used by a built-in detector or a reference
baseline are rejected.

## Rules that affect results

- A seed must reproduce the same generated data.
- The first 40 days must be baseline-only.
- The scaler and threshold must use training data only.
- Labels are simulated routine states; they are not medical labels.

`generate` writes `raw_days/`, `daily_features.csv`, and `scenario_labels.csv` for each
scenario. `benchmark` writes per-run results, a summary table, and its configuration. These
are generated files and are ignored by Git.
