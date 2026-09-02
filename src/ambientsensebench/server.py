"""Local web server for the AmbientSenseBench explorer and generator app."""

from __future__ import annotations

import json
import re
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .custom_scenario import (
    PRESETS,
    SENSOR_VOCABULARY,
    generate_custom_scenario,
    normalise_episodes,
)
from .explorer import SCENARIO_DESCRIPTIONS, generate_explorer_payload
from .generate_scenarios import DEFAULT_CONFIG_PATH
from .generate_raw_data import load_profiles


class ExplorerHandler(SimpleHTTPRequestHandler):
    service_output_root: Path
    web_root = files("ambientsensebench").joinpath("web")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/scenarios":
            self._send_json({"scenarios": SCENARIO_DESCRIPTIONS})
            return

        static_files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.css": ("app.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "application/javascript; charset=utf-8"),
        }
        if path not in static_files:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        filename, content_type = static_files[path]
        content = self.web_root.joinpath(filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/generate":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > 2048:
                raise ValueError("Request is too large")

            request = json.loads(self.rfile.read(content_length) or b"{}")
            scenario_id = str(request.get("scenario", ""))
            seed = int(request.get("seed", 42))
            payload = generate_explorer_payload(
                scenario_id,
                seed,
                self.service_output_root,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json(payload)

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        return


def run_explorer(host: str, port: int, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    ExplorerHandler.service_output_root = output_root
    server = ThreadingHTTPServer((host, port), ExplorerHandler)
    print(f"AmbientSenseBench Explorer: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


_JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_DOWNLOADABLE_FILES = {
    "daily_features.csv": "text/csv; charset=utf-8",
    "daily_features_dp.csv": "text/csv; charset=utf-8",
    "scenario_labels.csv": "text/csv; charset=utf-8",
    "events.csv": "text/csv; charset=utf-8",
}


class GeneratorHandler(SimpleHTTPRequestHandler):
    """Serves the synthetic-data generation app and its JSON API."""

    service_output_root: Path
    web_root = files("ambientsensebench").joinpath("web")

    static_files = {
        "/": ("generator.html", "text/html; charset=utf-8"),
        "/generator.js": ("generator.js", "application/javascript; charset=utf-8"),
        "/generator.css": ("generator.css", "text/css; charset=utf-8"),
    }

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/defaults":
            self._send_json(self._defaults_payload())
            return

        if path == "/api/download":
            self._handle_download(parse_qs(parsed.query))
            return

        if path not in self.static_files:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        filename, content_type = self.static_files[path]
        content = self.web_root.joinpath(filename).read_bytes()
        self._send_bytes(content, content_type)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/generate":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > 8192:
                raise ValueError("Request is too large")

            request = json.loads(self.rfile.read(content_length) or b"{}")

            n_days = int(request.get("n_days", 365))
            if not 60 <= n_days <= 1095:
                raise ValueError("n_days must be within 60..1095")

            seed = int(request.get("seed", 42))

            episodes = normalise_episodes(request.get("episodes", []), n_days)

            active_sensors = request.get("active_sensors")
            if active_sensors is not None:
                valid = {sensor["id"] for sensor in SENSOR_VOCABULARY}
                active_sensors = [s for s in active_sensors if s in valid]
                if not active_sensors:
                    raise ValueError("at least one sensor must remain active")

            dp_epsilon = request.get("dp_epsilon")
            if dp_epsilon in (None, "", 0, "0"):
                dp_epsilon = None
            else:
                dp_epsilon = float(dp_epsilon)
                if not 0.01 <= dp_epsilon <= 10.0:
                    raise ValueError("dp_epsilon must be within 0.01..10")

            job_id = uuid.uuid4().hex
            job_dir = self.service_output_root / "jobs" / job_id

            payload = generate_custom_scenario(
                job_dir,
                n_days=n_days,
                seed=seed,
                episodes=episodes,
                active_sensors=active_sensors,
                dp_epsilon=dp_epsilon,
            )
            payload["job_id"] = job_id
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json(payload)

    def _defaults_payload(self) -> dict:
        profiles = load_profiles(DEFAULT_CONFIG_PATH)
        return {
            "presets": PRESETS,
            "sensors": SENSOR_VOCABULARY,
            "profiles": {
                "baseline": _profile_summary(profiles["baseline"]),
                "distress": _profile_summary(profiles["distress"]),
            },
            "limits": {"min_days": 60, "max_days": 1095, "max_episodes": 6},
        }

    def _handle_download(self, query: dict) -> None:
        job_id = (query.get("job", [""])[0]).strip()
        filename = (query.get("file", [""])[0]).strip()

        if not _JOB_ID_PATTERN.match(job_id) or filename not in _DOWNLOADABLE_FILES:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        file_path = self.service_output_root / "jobs" / job_id / filename
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", _DOWNLOADABLE_FILES[filename])
        self.send_header(
            "Content-Disposition", f'attachment; filename="{filename}"'
        )
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_bytes(self, content: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        return


def _profile_summary(profile: dict) -> dict:
    """Return only the scalar behavioural means, for display in the UI."""
    return {
        key: value
        for key, value in profile.items()
        if isinstance(value, (int, float)) and key.endswith("mean")
    }


def run_generator_app(host: str, port: int, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    GeneratorHandler.service_output_root = output_root
    server = ThreadingHTTPServer((host, port), GeneratorHandler)
    print(f"AmbientSenseBench Generator: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
