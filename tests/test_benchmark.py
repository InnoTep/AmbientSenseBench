from ambientsensebench.benchmark import run_benchmark


def test_benchmark_writes_per_run_and_summary_tables(tmp_path):
    results, summary = run_benchmark(
        tmp_path,
        seeds=(7,),
        scenarios=("P01",),
        models=("isolation_forest", "local_outlier_factor"),
    )

    assert len(results) == 2
    assert len(summary) == 2
    assert set(results["model_id"]) == {"isolation_forest", "local_outlier_factor"}
    assert (tmp_path / "per_run_results.csv").exists()
    assert (tmp_path / "summary_results.csv").exists()
    assert (tmp_path / "benchmark_config.json").exists()
