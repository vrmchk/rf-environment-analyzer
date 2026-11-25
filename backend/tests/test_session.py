"""Session behavior checked with test doubles: a spy, a mock and dummies."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from rfea_backend.app import create_app
from rfea_backend.classify.model import load_classifier
from rfea_backend.dsp.detector import DetectorConfig
from rfea_backend.pipeline import Session, SessionParams
from rfea_backend.settings import Settings
from rfea_backend.sources.virtual import VirtualSource


def test_event_is_stored_before_it_is_sent(scenes_dir, model_dir) -> None:
    """Spy: on_event (the log) and emit (the clients) record their calls in one list."""
    calls: list[tuple[str, dict | bytes]] = []
    session = Session(
        VirtualSource(scenes_dir, "tone"), "tone", SessionParams(), realtime=False,
        classifier=load_classifier(model_dir),
        emit=lambda m: calls.append(("emit", m)),
        on_event=lambda e: calls.append(("store", e)),
    )  # fmt: skip
    session.start()
    session.join(timeout=30)

    stored = [i for i, (kind, _) in enumerate(calls) if kind == "store"]
    assert len(stored) == 1  # the tone of the scene
    i = stored[0]
    event = calls[i][1]
    sent = [j for j, (kind, m) in enumerate(calls) if kind == "emit" and m == event]
    assert sent == [i + 1]  # NFR-9: shown only after it is in the log, and only once
    assert calls[-1][1]["type"] == "session_ended"


def test_session_does_not_start_when_the_log_cannot_store_it(scenes_dir, model_dir) -> None:
    """Mock: the event log refuses to store the session."""
    repo = MagicMock()
    repo.create_session.side_effect = OperationalError("INSERT", {}, Exception("disk full"))
    settings = Settings(scenes_dir=scenes_dir, model_dir=model_dir, database_url="unused")
    with patch("rfea_backend.app.Repo", return_value=repo):
        client = TestClient(create_app(settings, realtime=False))
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "scene": "tone"})
        reply = ws.receive_json()
    assert reply == {"type": "error", "message": "cannot store the session: OperationalError"}
    repo.create_session.assert_called_once()
    repo.add_event.assert_not_called()
    repo.end_session.assert_not_called()
    assert client.get("/api/health").json()["session"] is None


def test_stored_params_take_gamma_from_the_session() -> None:
    """Dummies: the source and emit are needed to build a session but never used here."""
    dummy_source, dummy_emit = object(), object()
    session = Session(dummy_source, "any", SessionParams(gamma_db=6, tau=0.9, k_max=10),
                      dummy_emit, detector=DetectorConfig(gamma_db=3, merge_gap=4))  # fmt: skip
    p = session.stored_params()
    assert (p["gamma_db"], p["tau"], p["k_max"]) == (6, 0.9, 10)  # FR-7, FR-8 from the session
    assert p["merge_gap"] == 4 and p["min_rows"] == 3  # the rest of the detector and tracker
