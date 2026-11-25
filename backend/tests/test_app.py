import json

import numpy as np
from fastapi.testclient import TestClient

from rfea_backend.api.protocol import decode
from rfea_backend.app import create_app
from rfea_backend.settings import Settings


def client(scenes_dir, model_dir) -> TestClient:
    settings = Settings(scenes_dir=scenes_dir, model_dir=model_dir)
    return TestClient(create_app(settings, realtime=False))


def test_rest_lists_scenes_and_health(scenes_dir, model_dir) -> None:
    c = client(scenes_dir, model_dir)
    assert c.get("/api/health").json()["status"] == "ok"
    [scene] = c.get("/api/scenes").json()
    assert scene["name"] == "tone"
    assert scene["n_signals"] == 0


def test_session_streams_rows_until_the_scene_ends(scenes_dir, model_dir) -> None:
    with client(scenes_dir, model_dir).websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "scene": "tone", "params": {"gamma_db": 3}})
        started = ws.receive_json()
        assert started["type"] == "session_started"
        assert started["params"]["gamma_db"] == 3
        rows, tracks, events = [], [], []
        while True:
            msg = ws.receive()
            if msg.get("bytes") is not None:
                rows.append(decode(msg["bytes"]))
                continue
            m = json.loads(msg["text"])
            if m["type"] in ("tracks", "event"):
                (tracks if m["type"] == "tracks" else events).append(m)
                continue
            ended = m
            break
        assert ended["type"] == "session_ended"
        assert ended["reason"] == "finished", ended
        assert [m["row"] for m in tracks] == list(range(6))
        # the tone is confirmed as a track from the third row (min_rows = 3)
        assert tracks[1]["tracks"] == []
        [box] = tracks[2]["tracks"]
        assert box["f_lo_hz"] < 100.1e6 < box["f_hi_hz"]
        assert box["f_hi_hz"] - box["f_lo_hz"] < 10e3
        assert box["t0"] == 0 and box["n_frames"] > 0
        # the untrained model either decides or the signal is logged as unknown at the end
        [event] = events
        assert event["track_id"] == box["id"]
        assert event["status"] in ("identified", "unknown")
        assert 0 < event["n_frames"] <= 20
        assert abs(event["center_freq_hz"] - 100.1e6) < 2e3
        assert event["detected_at_s"] == 0 and event["snr_db"] > 10
        assert [r["row"] for r in rows] == list(range(6))
        r = rows[-1]
        assert r["n"] == 1024 and len(r["levels"]) == 1024
        assert r["t"] == 0.3
        peak = int(np.argmax(np.frombuffer(r["levels"], dtype=np.uint8)))
        assert abs(peak - (512 + round(100e3 / 1e6 * 1024))) <= 1


def test_start_errors_are_reported(scenes_dir, model_dir) -> None:
    with client(scenes_dir, model_dir).websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "scene": "nope"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "pause"})
        assert ws.receive_json()["message"] == "no session is running"
