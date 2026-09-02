# Benchmark protocol

## Aim

This protocol evaluates whether detectors trained only on an initial routine baseline
remain stable when the simulator produces alternative event-level realizations of the
same reference scenario.

## Data generation

For every requested seed, the runner regenerates each selected 365-day scenario from
raw ambient events. Daily features are then extracted through the standard pipeline.
The seed affects event-level variability; scenario definitions and profile parameters
remain fixed.

## Detectors and training rule

The standard benchmark compares Isolation Forest, One-Class SVM, Local Outlier Factor,
Bayesian Online Change-Point Detection (BOCPD), and a two-sided CUSUM chart. The first
three score each day independently; BOCPD and CUSUM process the train and test days in
chronological order.

Every detector is calibrated on the first 40 days, which must all carry the simulated
baseline label. The operational threshold is the 95th percentile of its baseline anomaly
scores. Test labels are never used to set that threshold. Feature scaling is fitted on the
same baseline window only.

## Reported results

`per_run_results.csv` contains one row for every seed, scenario, and detector.
`summary_results.csv` reports mean and standard deviation by scenario and detector
for AUPRC, fixed-threshold F1, total alarms, true alarms, false alarms, and lead time.

The 25-seed analysis adds statistical reference baselines, privacy experiments, robustness
checks, and paired tests. See [`reproducibility.md`](reproducibility.md) for the commands.

The simulated labels and all measurements are research artefacts. They do not support
clinical claims or individual-level decision-making.
