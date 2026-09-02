# Reproducibility

## Standard benchmark

```bash
python -m ambientsensebench benchmark --output outputs/benchmark
python -m ambientsensebench figures --input outputs/benchmark/summary_results.csv
```

Defaults: seeds `0,1,2,3,4`; scenarios `P01` to `P04`; Isolation Forest, One-Class SVM,
Local Outlier Factor, BOCPD, and CUSUM. The benchmark command writes its configuration with
the result tables.

## Extended experiments

Run Phase A first. The other phases use its generated scenarios and stored scores.

```bash
python -m ambientsensebench.tier2_experiments --phase A --workers 2 --seeds 25 --output outputs/tier2
python -m ambientsensebench.tier2_experiments --phase B --workers 2 --seeds 25 --output outputs/tier2
python -m ambientsensebench.tier2_experiments --phase C --workers 2 --seeds 25 --output outputs/tier2
python -m ambientsensebench.tier2_experiments --phase E --workers 2 --seeds 25 --output outputs/tier2
python -m ambientsensebench.tier2_experiments --phase F --output outputs/tier2
python -m ambientsensebench.stats_tests --input outputs/tier2
```

Record the package versions, command line, machine details, and Git commit with archived
results. Do not commit generated results to the source repository.

`apply_wp7_dp_noise.py` is a per-feature, per-day Laplace-noise experiment. It is not a
trajectory-level privacy guarantee. `real_comparison.py` needs a separately obtained CASAS
Aruba dataset; this repository does not include or download it.
