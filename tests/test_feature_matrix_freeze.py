"""Regression freeze of the generated feature matrix.

Pins the exact bytes of the P01 / seed-0 daily feature matrix so that any later
refactor of the generator or extractor can be checked against the corpus the paper's
results were computed on. Generation is deterministic per (scenario, seed), so this
test regenerates the scenario from scratch and must reproduce the frozen digest.
"""

import hashlib

from ambientsensebench.generate_scenarios import generate_scenario

# SHA-256 of outputs/tier2/scenarios/seed-0/P01/daily_features.csv as used for the
# paper's 25-seed reference run. The digest is taken over LF-normalised bytes so that
# it is platform-independent: the csv writer emits CRLF on Windows and LF elsewhere,
# and only the line terminator differs between them.
FROZEN_SHA256 = "c43981057047bdc272f2d6d2b7a7de6d9be0afd7bf4217aed3fafc664f700e78"


def test_daily_feature_matrix_is_frozen(tmp_path):
    generate_scenario("P01", seed_override=0, output_root=str(tmp_path))
    csv_path = tmp_path / "P01" / "daily_features.csv"
    normalised = csv_path.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(normalised).hexdigest()
    assert digest == FROZEN_SHA256, (
        "daily_features.csv for P01/seed-0 no longer matches the frozen reference "
        "matrix; generator or extractor behaviour has changed"
    )
