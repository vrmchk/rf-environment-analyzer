"""Event log storage. SQLite always; PostgreSQL with Alembic migrations when
RFEA_TEST_DATABASE_URL points to an empty test database (e.g. the one in docker compose)."""

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from rfea_backend.db.repo import Repo

EVENT = {
    "session_id": "s1", "track_id": 7, "detected_at_s": 1.5, "decided_at_s": 1.75,
    "detected_at": "2026-11-25T12:00:00+00:00", "decided_at": "2026-11-25T12:00:00.250000+00:00",
    "center_freq_hz": 100.2e6, "bandwidth_hz": 25e3, "status": "identified", "label": "BPSK",
    "confidence": 0.97, "n_frames": 4, "observation_s": 0.25, "snr_db": 12.5,
    "bandwidth_out_of_range": False,
}  # fmt: skip


def roundtrip(repo: Repo) -> None:
    repo.migrate()
    repo.create_session("s1", "demo", "ab" * 32, {"gamma_db": 6.0}, {"commit": "x"},
                        datetime(2026, 11, 25, 12, tzinfo=UTC))  # fmt: skip
    repo.add_event(EVENT)
    repo.add_event(EVENT | {"track_id": 8, "status": "unknown", "label": None, "snr_db": None})
    repo.end_session("s1", "finished", None)
    [s] = repo.list_sessions()
    assert s["n_events"] == 2 and s["end_reason"] == "finished" and s["params"]["gamma_db"] == 6
    events = repo.list_events("s1")
    assert [e["track_id"] for e in events] == [7, 8]
    assert events[0]["label"] == "BPSK" and events[1]["label"] is None
    csv = repo.events_csv("s1")
    data = [x for x in csv.splitlines() if not x.startswith("#")]
    assert len(data) == 3
    assert data[2].split(",")[8] == ""  # NULL label -> empty field
    assert repo.events_csv("missing") is None


def test_repo_on_sqlite(tmp_path) -> None:
    roundtrip(Repo(f"sqlite:///{(tmp_path / 'log.db').as_posix()}"))


@pytest.mark.skipif("RFEA_TEST_DATABASE_URL" not in os.environ, reason="no test PostgreSQL")
def test_repo_on_postgresql_with_migrations() -> None:
    repo = Repo(os.environ["RFEA_TEST_DATABASE_URL"])
    with repo.engine.begin() as c:  # start from an empty schema
        c.execute(text("DROP TABLE IF EXISTS events, sessions, alembic_version"))
    roundtrip(repo)
    with repo.engine.connect() as c:
        assert c.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0001"
