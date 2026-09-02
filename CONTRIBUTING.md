# Contributing

Use Python 3.10 or later.

```bash
python -m pip install -e ".[dev]"
pytest
```

Changes to profiles, feature extraction, splitting, scaling, thresholds, or metrics can
change the reported results. Describe the change in the pull request, update tests, and
record regenerated results outside the repository.

Do not commit generated event logs, benchmark results, application jobs, environments, or
build files.

Keep reusable code in `src/ambientsensebench`. Use caller-supplied paths instead of local
hard-coded paths. Use the shared evaluation functions for detector comparisons. Preserve
the formatting style of the module being changed.
