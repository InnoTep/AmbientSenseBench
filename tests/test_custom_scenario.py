import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from ambientsensebench.custom_scenario import (
    generate_custom_scenario,
    normalise_episodes,
    severity_for_day,
)
from ambientsensebench.server import GeneratorHandler
from ambientsensebench.wp6_evaluation import FEATURE_COLUMNS


def test_severity_episode_peaks_and_recovers():
    episodes = normalise_episodes(
        [{"mode": "episode", "onset": 50, "offset": 80, "tau": 2.5, "max_severity": 1.0}],
        n_days=120,
    )
    assert severity_for_day(10, episodes) == 0.0
    assert severity_for_day(65, episodes) > 0.9
    assert severity_for_day(119, episodes) < 0.1


def test_generate_custom_scenario_produces_clean_features(tmp_path):
    payload = generate_custom_scenario(
        tmp_path / "job",
        n_days=90,
        seed=5,
        episodes=normalise_episodes(
            [{"mode": "episode", "onset": 55, "offset": 80, "tau": 2.5, "max_severity": 1.0}],
            n_days=90,
        ),
        dp_epsilon=1.0,
    )
    assert len(payload["records"]) == 90
    assert payload["distress_days"] > 0
    assert payload["event_count"] > 0
    assert payload["dp_epsilon"] == 1.0

    features = pd.read_csv(tmp_path / "job" / "daily_features.csv")
    assert features[FEATURE_COLUMNS].isna().sum().sum() == 0
    assert (tmp_path / "job" / "events.csv").exists()
    assert (tmp_path / "job" / "daily_features_dp.csv").exists()


def test_sensor_filter_only_affects_raw_export(tmp_path):
    episodes = normalise_episodes(
        [{"mode": "episode", "onset": 55, "offset": 80, "tau": 2.5, "max_severity": 1.0}],
        n_days=90,
    )
    payload = generate_custom_scenario(
        tmp_path / "job",
        n_days=90,
        seed=5,
        episodes=episodes,
        active_sensors=["bedroom_pir", "kitchen_pir"],
    )
    # Features remain complete even though sensors were dropped from the export.
    features = pd.read_csv(tmp_path / "job" / "daily_features.csv")
    assert features[FEATURE_COLUMNS].isna().sum().sum() == 0
    # Exported events only contain the selected sensors.
    events = pd.read_csv(tmp_path / "job" / "events.csv")
    assert set(events["sensor_id"].unique()) <= {"bedroom_pir", "kitchen_pir"}
    assert payload["event_count"] == len(events)


def test_generator_server_endpoints(tmp_path):
    GeneratorHandler.service_output_root = tmp_path / "outputs"
    server = ThreadingHTTPServer(("127.0.0.1", 0), GeneratorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"

        with urlopen(f"{base_url}/") as response:
            assert b"AmbientSenseBench" in response.read()

        defaults = json.loads(urlopen(f"{base_url}/api/defaults").read())
        assert set(defaults["presets"]) == {"P01", "P02", "P03", "P04"}
        assert defaults["limits"]["min_days"] == 60

        request = Request(
            f"{base_url}/api/generate",
            data=json.dumps(
                {
                    "n_days": 90,
                    "seed": 3,
                    "episodes": [
                        {"mode": "episode", "onset": 55, "offset": 80, "tau": 2.5, "max_severity": 1.0}
                    ],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read())
        assert len(payload["records"]) == 90
        job_id = payload["job_id"]

        with urlopen(f"{base_url}/api/download?job={job_id}&file=events.csv") as response:
            assert response.read().startswith(b"timestamp,sensor_id,state")

        # Path traversal / unknown files are rejected.
        for bad in ["job=../../etc&file=events.csv", f"job={job_id}&file=secret.txt"]:
            try:
                urlopen(f"{base_url}/api/download?{bad}")
                raise AssertionError("download should have been rejected")
            except HTTPError as error:
                assert error.code == 404

        # Episodes inside the training window are rejected with a clear message.
        bad_request = Request(
            f"{base_url}/api/generate",
            data=json.dumps(
                {
                    "n_days": 90,
                    "seed": 1,
                    "episodes": [
                        {"mode": "episode", "onset": 10, "offset": 30, "tau": 2.0, "max_severity": 1.0}
                    ],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(bad_request)
            raise AssertionError("should reject training-window episode")
        except HTTPError as error:
            assert error.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
