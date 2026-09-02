# Development

Install the development dependencies and run the tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

Use `python -m ambientsensebench --help` for the public commands. The project uses a `src/`
layout, so install it before running commands from another directory.

For a quick integration check:

```bash
python -m ambientsensebench benchmark --seeds 7 --scenarios P01 --output outputs/smoke
```

The tests cover scenario generation, daily features, detector behaviour, benchmark tables,
figures, and the two local HTTP handlers.

Dependencies are declared in `pyproject.toml`. Do not add a second dependency list unless
there is a separate supported deployment environment that needs one.
