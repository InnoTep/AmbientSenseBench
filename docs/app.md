# AmbientSenseBench generation app

The generation app is a small, dependency-light web interface for producing
synthetic ambient-sensing datasets. It exposes the same raw-event generator,
daily-feature extractor, and baseline-only Isolation Forest scoring used by the
reference benchmark, but lets a user define an arbitrary routine-change
trajectory instead of choosing one of the four fixed scenarios.

The scenarios model changes in daily routine studied as a behavioural proxy in
mental-health research. They are synthetic data generated from explicit
assumptions; they are not clinical data, diagnostic labels, or evidence of
performance with real residents.

## Run locally

```bash
python -m pip install -e '.[dev]'
ambientsensebench app
```

Open <http://127.0.0.1:8765>. Choose a preset or build a custom trajectory,
optionally apply differential-privacy noise to the daily features, generate,
inspect the trajectory and detector alerts, and download the artefacts.

Options:

```bash
ambientsensebench app --host 127.0.0.1 --port 8765 --output outputs/app
```

## What you can configure

- **Days**: run length between 60 and 1095 days. The first 40 days are the
  baseline-only training window and must not contain a routine change.
- **Seed**: deterministic generator seed for reproducibility.
- **Episodes**: one to six routine-change episodes. Each episode is either an
  *episode* (rises to a peak severity and recovers) or a *monotone* drift
  (rises and stays). Onset, offset, smoothness (tau), and peak severity are
  configurable. Severity in [0, 1] interpolates the euthymic-baseline and
  distress behavioural profiles.
- **Differential privacy**: optionally add day-level Laplace noise (epsilon
  0.5, 1.0, or 2.0) to the daily features, reproducing the WP7 privacy release.
- **Sensors in the raw export**: choose which sensors appear in `events.csv`.
  This affects the exported raw stream only; daily features and the detector
  preview always use the full instrumentation.

## Downloads

Each run produces:

- `events.csv` — timestamped raw sensor events (`timestamp, sensor_id, state`).
- `daily_features.csv` — nine daily behavioural features with `date`, `label`,
  and `severity`.
- `daily_features_dp.csv` — differentially private features (only when a
  privacy level is selected).
- `scenario_labels.csv` — per-day label and severity schedule.

## Deploy publicly with Docker

```bash
docker build -t ambientsensebench .
docker run --rm -p 8765:8765 -v "$(pwd)/data:/data" ambientsensebench
```

The container binds to `0.0.0.0:8765` and writes generated datasets to the
mounted `/data` volume. Put it behind a TLS-terminating reverse proxy (for
example Nginx or Caddy) for a public deployment. The bind address and port can
be overridden with the `ASB_HOST` and `ASB_PORT` environment variables.

## API

The interface is backed by a small JSON API, which can also be used directly:

- `GET /api/defaults` — presets, sensor vocabulary, profile summaries, limits.
- `POST /api/generate` — body `{n_days, seed, episodes, dp_epsilon,
  active_sensors}`; returns the preview payload and a `job_id`.
- `GET /api/download?job=<job_id>&file=<name>` — download a generated artefact.
