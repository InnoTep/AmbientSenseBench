"""Command-line interface for reproducible scenario generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import DEFAULT_MODELS, parse_csv_argument, run_benchmark
from .generate_scenarios import SCENARIOS, generate_scenario
from .server import run_explorer, run_generator_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ambientsensebench",
        description="Generate reproducible ambient-sensing benchmark scenarios.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="generate one reference scenario or all scenarios",
    )
    generate_parser.add_argument("scenario", choices=[*SCENARIOS, "all"])
    generate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="directory where scenario folders are created (default: outputs)",
    )
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--days", type=int, default=365)
    generate_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path to a compatible profile configuration",
    )

    explore_parser = subparsers.add_parser(
        "explore",
        help="start the local scenario explorer",
    )
    explore_parser.add_argument("--host", default="127.0.0.1")
    explore_parser.add_argument("--port", type=int, default=8765)
    explore_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/explorer"),
        help="directory for explorer-generated scenarios",
    )

    app_parser = subparsers.add_parser(
        "app",
        help="start the synthetic-data generation web app",
    )
    app_parser.add_argument("--host", default="127.0.0.1")
    app_parser.add_argument("--port", type=int, default=8765)
    app_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/app"),
        help="directory for app-generated datasets",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="run the multi-seed anomaly-detection benchmark",
    )
    benchmark_parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="comma-separated generator seeds",
    )
    benchmark_parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="comma-separated reference scenario identifiers",
    )
    benchmark_parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="comma-separated detector identifiers",
    )
    benchmark_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmark"),
        help="directory for generated scenarios and result tables",
    )

    figures_parser = subparsers.add_parser(
        "figures",
        help="create publication-oriented benchmark figures",
    )
    figures_parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/benchmark/summary_results.csv"),
        help="benchmark summary CSV",
    )
    figures_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmark/figures"),
        help="directory for PDF and PNG figures",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "explore":
        run_explorer(args.host, args.port, args.output)
        return

    if args.command == "app":
        run_generator_app(args.host, args.port, args.output)
        return

    if args.command == "benchmark":
        try:
            seeds = tuple(int(seed) for seed in parse_csv_argument(args.seeds))
            scenarios = parse_csv_argument(args.scenarios, SCENARIOS)
            models = parse_csv_argument(args.models, DEFAULT_MODELS)
        except ValueError as error:
            raise SystemExit(str(error)) from error

        _, summary = run_benchmark(args.output, seeds, scenarios, models)
        print(summary.to_string(index=False))
        print(f"\nSaved benchmark tables to: {args.output}")
        return

    if args.command == "figures":
        if not args.input.exists():
            raise SystemExit(f"Missing benchmark summary: {args.input}")
        from .plot_benchmark import create_benchmark_figures

        figures = create_benchmark_figures(args.input, args.output)
        for name, path in figures.items():
            print(f"Saved {name} figure to: {path}")
        return

    if args.days < 1:
        raise SystemExit("--days must be at least 1")

    scenario_ids = SCENARIOS if args.scenario == "all" else [args.scenario]
    for scenario_id in scenario_ids:
        generate_scenario(
            scenario_id,
            seed_override=args.seed,
            n_days=args.days,
            output_root=args.output,
            config_path=args.config,
        )


if __name__ == "__main__":
    main()
