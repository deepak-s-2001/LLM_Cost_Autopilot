import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    # Global safety net so no test ever touches the real dev DB, even through a code path that wasn't explicitly mocked.
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
