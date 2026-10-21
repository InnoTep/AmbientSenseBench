# Reproducibility

This page maps every table and figure in the paper to the data file and the command that
produce it. Commands run from the repository root after

```bash
python -m pip install -e ".[dev]"
```

with Python 3.10 or later. Generation is deterministic per scenario and seed: the same
command on the same package version reproduces the same feature matrices
(`tests/test_feature_matrix_freeze.py` pins the P01 seed-0 matrix). Record the package
version, Git commit, Python and library versions, and machine details with any archived
results. Generated results are not committed to the repository.

## 1. Reference run (25 seeds)

Every result in the paper except Table 6 and the Aruba check comes from the 25-seed
reference run below. Phase A must run first; the later phases reuse its generated
scenarios (`outputs/tier2/scenarios/seed-<n>/P0x/`) and its stored per-day scores
(`outputs/tier2/scores/*.npz`). `--workers` only sets the size of the process pool; the
results do not depend on it. The whole sequence took about two and a half hours with two
workers on a laptop-class machine.

```bash
python -m ambientsensebench.tier2_experiments --phase A --workers 2 --seeds 25 --output outputs/tier2 --scenario-root outputs/tier2/scenarios
python -m ambientsensebench.tier2_experiments --phase B --workers 2 --seeds 25 --output outputs/tier2 --scenario-root outputs/tier2/scenarios
python -m ambientsensebench.tier2_experiments --phase C --workers 2 --seeds 25 --output outputs/tier2
python -m ambientsensebench.tier2_experiments --phase D --workers 2 --seeds 25 --output outputs/tier2 --scenario-root outputs/tier2/scenarios
python -m ambientsensebench.tier2_experiments --phase E --output outputs/tier2 --scenario-root outputs/tier2/scenarios
python -m ambientsensebench.stats_tests --input outputs/tier2
python -m ambientsensebench.paper_analyses --analysis all --workers 2 --output outputs/tier2 --scenario-root outputs/tier2/scenarios
python tools/run_ks_validation25.py
```

The revision analyses run after the reference run:

```bash
python -m ambientsensebench.tier2_experiments --phase C --alphas 0.4,0.6 --workers 2 --seeds 25 --output outputs/tier2-alpha
python -m ambientsensebench.plot_alpha_curve --main outputs/tier2/per_run_main.csv --sensitivity outputs/tier2/sensitivity_extra.csv outputs/tier2-alpha/sensitivity_extra.csv --output outputs/tier2/figures
python tools/p02_prodrome_check.py
python tools/aruba_injection.py --aruba datasets/CASAS_Aruba/data/aruba.csv
```

| Phase | What it does | Output files |
| --- | --- | --- |
| A | Generates 25 seeds x 4 scenarios; evaluates the 5 core detectors and the 3 reference baselines; stores per-day scores | `per_run_main.csv`, `scores/*.npz` |
| B | Laplace release at epsilon 0.1, 0.5, 1, 2, 5, 10 for the core detectors; F1 with the threshold recomputed on the noisy window and with the clean threshold | `dp_sweep.csv` |
| C | Regenerates every scenario with the distress profile scaled by alpha 0.8 and 1.2 (alpha 1.0 is Phase A) | `sensitivity_extra.csv` |
| D | BOCPD hazard sweep, CUSUM h sweep, feature-set ablation, P04 training-window sweep | `param_sweeps.csv` |
| E | `tracemalloc` incremental memory, raw event volume, baseline-day feature correlations (seeds 0 to 4) | `tracemalloc_memory.csv`, `data_volume.csv`, `feature_correlation_baseline.csv` |
| `stats_tests` | Wilcoxon paired tests with Holm correction against the best detector per cell, bootstrap CIs, rank-biserial effect sizes; F1-versus-percentile trace; three-day persistence rule | `stats_paired_tests.csv`, `threshold_percentile_trace.csv`, `persistence_rule_f1.csv` |
| `paper_analyses` | Score-severity consistency and the P04 severity curve; Phase B repeated for the three reference baselines; the persistence rule on the noisy alarm streams at epsilon 1; the core-detector summary table | `severity_consistency.csv`, `severity_curve_p04.csv`, `dp_sweep_baselines.csv`, `dp_persistence_eps1.csv`, `summary_core25.csv` |
| `tools/run_ks_validation25.py` | Round-trip distributional validation over the 200 scenario-seed-state groups | `ks_validation25/ks_feature_summary.csv` |
| Phase C with `--alphas 0.4,0.6` | The two weaker profile magnitudes of the difficulty curve (alpha 0.8 and 1.2 come from the default Phase C, alpha 1.0 from Phase A) | `outputs/tier2-alpha/sensitivity_extra.csv` |
| `plot_alpha_curve` | Difficulty-curve figure over alpha 0.4 to 1.2 | `alpha_difficulty_curve.pdf`, `alpha_curve_per_run.csv` |
| `tools/p02_prodrome_check.py` | P02 regenerated with zero severity on the 40 training days (test days identical) and compared with Phase A | `p02_clean_training.csv`, `p02_clean_training_summary.csv` |
| `tools/aruba_injection.py` | Distress shift injected into the CASAS Aruba baseline; eight detectors clean and under the release (25 noise draws) | `outputs/aruba/aruba_injection_summary.csv` |

**SciPy version.** The Wilcoxon p-values depend on how the installed SciPy treats ties and
zero differences. With SciPy 1.14, `stats_tests` reproduces every mean difference,
bootstrap interval, and effect size of the paper exactly, but some Holm-corrected p-values
move by up to 0.007 and no significance verdict changes. Record the SciPy version with any
archived results.

**Phase letters.** Before release 1.0.0 the sweeps and extras phases were called E and F
and the letter D was unused; the experiment logs from the paper revision use those older
names. The phases were renamed contiguously for the release. Output-file names are
unchanged, so older logs and result files still match. The training-window sweep inside
Phase D covers only P04 because the first distress onsets of P01, P02 and P03 fall on days
59, 44 and 49, so no baseline-only window longer than the protocol's 40 days exists there.

## 2. Paper tables and figures

Every row below is produced by code in this repository. Numbering follows the revised
manuscript (Table 8 is the Aruba injection; the former Table 8, best detector per alpha, is
replaced by Figure 5). A rerun of Phase A on 21 October 2026 reproduced all 800 rows of the
paper's `per_run_main.csv` exactly.

| Paper item | Source file(s) | Command or derivation |
| --- | --- | --- |
| Table 1, feature sampling and rendering | none (documents `generate_raw_data.py`) | See [`generation-mechanism.md`](generation-mechanism.md) |
| Figure 1, mean AUPRC and F1 across 25 seeds | `outputs/tier2/summary_core25.csv` (`paper_analyses --analysis summary`, from Phase A) | `python -m ambientsensebench figures --input outputs/tier2/summary_core25.csv --output outputs/tier2/figures` |
| Figure 2, alarm composition and lead time | same | Same command; both figures are written together |
| Table 2, reference baselines | `per_run_main.csv`, the three baseline rows | Mean `auprc` and `f1_fixed_threshold` by scenario and detector |
| Figure 3, score-severity consistency (P04) | `severity_consistency.csv`, `severity_curve_p04.csv` | `python -m ambientsensebench.paper_analyses --analysis severity`: Spearman correlation between each stored test-window score and the simulated severity; the P04 curve is the mean within-run score percentile per tenth of severity |
| Table 3, distributional validation (200 groups) | `outputs/tier2/ks_validation25/ks_feature_summary.csv` | `python tools/run_ks_validation25.py` |
| Table 4, AUPRC and F1 at epsilon = 1 | `dp_sweep.csv` (Phase B) rows with `epsilon == 1.0`; clean columns from `per_run_main.csv` | Means by scenario and detector |
| Figure 4, privacy-utility curves | `dp_sweep.csv`, `per_run_main.csv` | `python -m ambientsensebench.plot_privacy_curve --input outputs/tier2 --output outputs/tier2/figures` (both rows, AUPRC and F1) |
| Table 5, threshold-provenance ablation | `dp_sweep.csv` at epsilon 1: `f1_dp_recomputed`, `f1_dp_cleanthr`; clean F1 from `per_run_main.csv` | Means by scenario and detector |
| Table 6, computational cost (five seeds) | Latency and artefact size: `outputs/paper/dp_edge_summary.csv`; incremental memory: `outputs/tier2/tracemalloc_memory.csv` (Phase E, seeds 0 to 4) | `python -m ambientsensebench benchmark --output outputs/benchmark` then `python -m ambientsensebench.paper_extras --scenario-root outputs/benchmark/scenarios --seeds 0,1,2,3,4 --output outputs/paper`; memory column from Phase E. Timings are machine-specific and support only comparison between detectors |
| Table 7, analytical state and update cost | none | Derived by hand from the detector definitions in `temporal_detectors.py` and the scikit-learn models |
| Figure 5, difficulty curve over alpha | `outputs/tier2-alpha/sensitivity_extra.csv` (Phase C, alpha 0.4 and 0.6), `sensitivity_extra.csv` (alpha 0.8 and 1.2), `per_run_main.csv` (alpha 1.0) | `python -m ambientsensebench.plot_alpha_curve` (full command in section 1) |
| Section 5.2, P02 with a prodrome-free training window | `p02_clean_training_summary.csv` | `python tools/p02_prodrome_check.py` |
| Table 8, semi-synthetic injection on Aruba | `outputs/aruba/aruba_injection_summary.csv` | `python tools/aruba_injection.py --aruba datasets/CASAS_Aruba/data/aruba.csv` |
| Appendix Table A.1, paired differences with bootstrap intervals | `stats_paired_tests.csv` | `stats_tests` |
| Paired tests, effect sizes, "statistically tied" | `stats_paired_tests.csv` | `python -m ambientsensebench.stats_tests --input outputs/tier2` |
| Full pairwise matrix among the core detectors | `stats_pairwise_core.csv` | `stats_tests` (`pairwise_core_tests`: every pair of core detectors, Holm-corrected within each scenario and metric) |
| Persistence-rule F1 on clean data; F1-versus-percentile trace | `persistence_rule_f1.csv`, `threshold_percentile_trace.csv` | `stats_tests` |
| Persistence rule under the release; reference baselines under the release (EWMA robustness) | `dp_persistence_eps1.csv`, `dp_sweep_baselines.csv` | `python -m ambientsensebench.paper_analyses --analysis dp-persistence` and `--analysis dp-baselines` (same noise draws as Phase B) |
| Hazard, h, feature-set and P04 window sweeps | `param_sweeps.csv` | Phase D |
| Raw data volume; baseline feature correlations | `data_volume.csv`, `feature_correlation_baseline.csv` | Phase E |
| Plausibility check against CASAS Aruba | `outputs/paper/real_comparison_summary.csv`, `aruba_daily.csv` | `python -m ambientsensebench.real_comparison --aruba <aruba.csv> --scenarios outputs/benchmark/scenarios --output outputs/paper` (see below) |

## 3. The Aruba comparison file

`real_comparison.py` does not read the raw CASAS download. It reads a comma-separated
file with at least four columns per row, in this order:

```text
date,time,location,state
2010-11-04,00:03:50.209589,Kitchen,ON
2010-11-04,00:03:57.399391,OutsideDoor,OPEN
```

- `date` is `YYYY-MM-DD`; `time` is `HH:MM:SS` with an optional fractional part, which is
  discarded. Rows that do not parse (including a header row) are skipped.
- `location` must be one of `Bedroom`, `LivingRoom`, `LoungeChair`, `Kitchen`,
  `DiningRoom`, `Bathroom`, `WorkArea`, `GuestRoom`, `OtherRoom` (mapped to the benchmark's
  PIR sensors through `LOCATION_MAP`) or `OutsideDoor` (mapped to `front_door`). Rows with
  any other label, such as temperature sensors, are dropped.
- `state` is passed through unchanged (`ON`/`OFF` for motion, `OPEN`/`CLOSE` for doors).

The CASAS recordings, Aruba among them, are archived on Zenodo under CC BY 4.0
(<https://doi.org/10.5281/zenodo.15708568>). The member `data/aruba.csv` of `data.zip`
already has this format, with locations rather than sensor ids, so it can be used as is:
extract that one file (about 12 MB compressed) to `datasets/CASAS_Aruba/data/aruba.csv`.
`real_comparison` run on it reproduces the paper's `aruba_daily.csv` exactly (218 usable
days, partial first and last days dropped), and `tools/aruba_injection.py` reads the same
file. The Aruba data are not redistributed here.

## 4. Five-seed development run

The `benchmark` command in the README is the five-seed, core-detector run used during
development and for the computational-cost profile (Table 6). It is not the source of the
other tables.

```bash
python -m ambientsensebench benchmark --output outputs/benchmark
python -m ambientsensebench figures --input outputs/benchmark/summary_results.csv
```

Defaults: seeds `0,1,2,3,4`; scenarios `P01` to `P04`; Isolation Forest, One-Class SVM,
Local Outlier Factor, BOCPD, and CUSUM. The command writes its configuration next to the
result tables.

`apply_wp7_dp_noise.py` is a per-feature, per-day Laplace-noise experiment. It is not a
trajectory-level privacy guarantee.
