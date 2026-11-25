"""FastAPI app: REST for lists, the event log and CSV; WebSocket /ws for control and streams.

Run: uv run uvicorn rfea_backend.app:create_app --factory --port 8000
Env: RFEA_DATA_DIR (scenes in <dir>/scenes) or RFEA_SCENES_DIR; RFEA_MODEL_DIR;
RFEA_DATABASE_URL.
"""

import asyncio
import contextlib
import hashlib
from dataclasses import asdict, fields
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from sqlalchemy.exc import SQLAlchemyError

from rfea_backend.api import protocol
from rfea_backend.classify.model import Classifier, ModelError, load_classifier
from rfea_backend.db.repo import Repo
from rfea_backend.pipeline import Session, SessionParams
from rfea_backend.settings import Settings
from rfea_backend.sources.virtual import META_SUFFIX, VirtualSource, list_scenes

CLIENT_QUEUE = 400  # messages; ~10 s of rows. A slow client loses the oldest ones.


class Hub:
    """Fans session messages out to the connected clients (runs on the event loop)."""

    def __init__(self) -> None:
        self.clients: set[asyncio.Queue] = set()

    def add(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(CLIENT_QUEUE)
        self.clients.add(q)
        return q

    def remove(self, q: asyncio.Queue) -> None:
        self.clients.discard(q)

    def publish(self, msg: dict | bytes) -> None:
        for q in self.clients:
            if q.full():
                q.get_nowait()
            q.put_nowait(msg)


def create_app(settings: Settings | None = None, realtime: bool = True) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="RF environment analyzer")
    hub = Hub()
    state: dict = {"session": None}
    classifier: Classifier | None = None
    model_error: str | None = None
    try:  # NFR-8: an incompatible model is not loaded; sessions do not start without one
        classifier = load_classifier(settings.model_dir)
    except ModelError as e:
        model_error = str(e)
    repo = Repo(settings.database_url)
    db_error: str | None = None
    try:
        repo.migrate()
    except (SQLAlchemyError, OSError) as e:
        db_error = f"{type(e).__name__}: {e}"

    def current() -> Session | None:
        s = state["session"]
        return s if s is not None and s.ended is None else None

    @app.get("/api/health")
    def health() -> dict:
        s = current()
        ok = classifier is not None and db_error is None
        return {
            "status": "ok" if ok else "error",
            "model": classifier.describe() if classifier else None,
            "model_error": model_error,
            "db_error": db_error,
            "session": s.id if s else None,
        }

    @app.get("/api/scenes")
    def scenes() -> list[dict]:
        return [asdict(s) for s in list_scenes(settings.scenes_dir)]

    @app.get("/api/sessions")
    def sessions() -> list[dict]:
        return repo.list_sessions()

    @app.get("/api/sessions/{sid}/events")
    def events(sid: str) -> list[dict]:
        if repo.get_session(sid) is None:
            raise HTTPException(404, "session not found")
        return repo.list_events(sid)

    @app.get("/api/sessions/{sid}/events.csv")
    def events_csv(sid: str) -> PlainTextResponse:
        text = repo.events_csv(sid)
        if text is None:
            raise HTTPException(404, "session not found")
        return PlainTextResponse(
            text,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="session-{sid}.csv"'},
        )

    def start(msg: dict, loop: asyncio.AbstractEventLoop) -> dict | None:
        if classifier is None:
            return protocol.error(f"model not loaded: {model_error}")
        if db_error is not None:
            return protocol.error(f"database unavailable: {db_error}")
        if current() is not None:
            return protocol.error("a session is already running")
        known = {f.name for f in fields(SessionParams)}
        try:
            params = SessionParams(**{k: v for k, v in (msg.get("params") or {}).items()
                                      if k in known})  # fmt: skip
            source = VirtualSource(settings.scenes_dir, str(msg.get("scene", "")))
        except (FileNotFoundError, ValueError, TypeError) as e:
            return protocol.error(str(e))

        def emit(m: dict | bytes) -> None:
            loop.call_soon_threadsafe(hub.publish, m)

        session = Session(source, source.info.name, params, emit, classifier, realtime=realtime,
                          on_event=repo.add_event)  # fmt: skip
        session.on_end = lambda reason, error: repo.end_session(session.id, reason, error)
        meta = settings.scenes_dir / f"{source.info.name}{META_SUFFIX}"
        try:
            repo.create_session(session.id, source.info.name,
                                hashlib.sha256(meta.read_bytes()).hexdigest(),
                                session.stored_params(), classifier.describe(),
                                datetime.now(UTC))  # fmt: skip
        except SQLAlchemyError as e:
            source.close()
            return protocol.error(f"cannot store the session: {type(e).__name__}")
        state["session"] = session
        hub.publish(session.started_message())
        session.start()
        return None

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        loop = asyncio.get_running_loop()
        queue = hub.add()
        if (s := current()) is not None:
            queue.put_nowait(s.started_message())

        async def sender() -> None:
            while True:
                msg = await queue.get()
                if isinstance(msg, bytes):
                    await websocket.send_bytes(msg)
                else:
                    await websocket.send_json(msg)

        send_task = asyncio.create_task(sender())
        try:
            while True:
                msg = await websocket.receive_json()
                kind, s = msg.get("type"), current()
                reply = None
                if kind == "start":
                    reply = start(msg, loop)
                elif kind in ("pause", "resume", "stop"):
                    if s is None:
                        reply = protocol.error("no session is running")
                    else:
                        getattr(s, kind)()
                        if kind != "stop":
                            hub.publish(s.started_message())  # carries the paused flag
                else:
                    reply = protocol.error(f"unknown message type {kind!r}")
                if reply is not None:
                    queue.put_nowait(reply)
        except WebSocketDisconnect:
            pass
        finally:  # the session keeps running without clients (NFR-8)
            hub.remove(queue)
            send_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await send_task

    return app
