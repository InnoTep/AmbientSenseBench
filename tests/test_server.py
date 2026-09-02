import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from ambientsensebench.server import ExplorerHandler


def test_local_server_serves_the_explorer_and_generates_data(tmp_path):
    ExplorerHandler.service_output_root = tmp_path / "outputs"
    server = ThreadingHTTPServer(("127.0.0.1", 0), ExplorerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base_url}/") as response:
            assert b"Scenario Explorer" in response.read()

        request = Request(
            f"{base_url}/api/generate",
            data=json.dumps({"scenario": "P01", "seed": 3}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read())

        assert payload["scenario"] == "P01"
        assert len(payload["records"]) == 365
        assert payload["metrics"]["training_days"] == 40
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
