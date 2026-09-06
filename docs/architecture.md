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

## Rules that affect results

- A seed must reproduce the same generated data.
- The first 40 days must be baseline-only.
- The scaler and threshold must use training data only.
- Labels are simulated routine states; they are not medical labels.

`generate` writes `raw_days/`, `daily_features.csv`, and `scenario_labels.csv` for each
scenario. `benchmark` writes per-run results, a summary table, and its configuration. These
are generated files and are ignored by Git.
