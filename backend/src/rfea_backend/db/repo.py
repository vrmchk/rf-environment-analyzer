"""Event log storage. Every write is its own transaction (NFR-9): an event is stored whole
or not at all, and a session row exists before any of its events."""

import csv
import io
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session as DbSession

from rfea_backend.db.models import EVENT_COLUMNS, Base, EventRow, SessionRow

MIGRATIONS = Path(__file__).resolve().parent / "migrations"  # inside the package


def _dt(v: str | datetime) -> datetime:
    return v if isinstance(v, datetime) else datetime.fromisoformat(v)


class Repo:
    def __init__(self, url: str):
        self.url = url
        self.engine = create_engine(url, pool_pre_ping=True)

    def migrate(self) -> None:
        """Bring the schema up to date: Alembic on PostgreSQL, create_all elsewhere."""
        if self.engine.dialect.name == "postgresql":
            cfg = Config()
            cfg.set_main_option("script_location", str(MIGRATIONS))
            cfg.set_main_option("sqlalchemy.url", self.url.replace("%", "%%"))
            command.upgrade(cfg, "head")
        else:
            Base.metadata.create_all(self.engine)

    def create_session(self, sid: str, scene: str, scene_sha256: str, params: dict,
                       model: dict, started_at: datetime) -> None:  # fmt: skip
        with DbSession(self.engine) as db, db.begin():
            db.add(SessionRow(id=sid, scene=scene, scene_sha256=scene_sha256, params=params,
                              model=model, started_at=started_at))  # fmt: skip

    def end_session(self, sid: str, reason: str, error: str | None) -> None:
        with DbSession(self.engine) as db, db.begin():
            row = db.get(SessionRow, sid)
            if row is not None:
                row.ended_at, row.end_reason, row.error = datetime.now(UTC), reason, error

    def add_event(self, e: dict) -> None:
        values = {k: e[k] for k in EVENT_COLUMNS}
        values["detected_at"] = _dt(values["detected_at"])
        values["decided_at"] = _dt(values["decided_at"])
        with DbSession(self.engine) as db, db.begin():
            db.add(EventRow(session_id=e["session_id"], **values))

    def list_sessions(self) -> list[dict]:
        with DbSession(self.engine) as db:
            n = (select(EventRow.session_id, func.count().label("n"))
                 .group_by(EventRow.session_id).subquery())  # fmt: skip
            rows = db.execute(
                select(SessionRow, n.c.n).outerjoin(n, n.c.session_id == SessionRow.id)
                .order_by(SessionRow.started_at.desc())
            ).all()  # fmt: skip
            return [_session_dict(s) | {"n_events": c or 0} for s, c in rows]

    def get_session(self, sid: str) -> dict | None:
        with DbSession(self.engine) as db:
            row = db.get(SessionRow, sid)
            return _session_dict(row) if row else None

    def list_events(self, sid: str) -> list[dict]:
        with DbSession(self.engine) as db:
            rows = db.scalars(
                select(EventRow).where(EventRow.session_id == sid).order_by(EventRow.id)
            ).all()
            return [_event_dict(r) for r in rows]

    def events_csv(self, sid: str) -> str | None:
        """FR-12: UTF-8, comma, dot decimals, ISO 8601; scene and parameters as # comments."""
        s = self.get_session(sid)
        if s is None:
            return None
        out = io.StringIO()
        out.write(f"# session {s['id']}\n# scene {s['scene']} sha256 {s['scene_sha256']}\n")
        out.write(f"# started {s['started_at']} ended {s['ended_at']} ({s['end_reason']})\n")
        out.write(f"# params {_kv(s['params'])}\n# model {_kv(s['model'])}\n")
        w = csv.writer(out, lineterminator="\n")
        w.writerow(EVENT_COLUMNS)
        for e in self.list_events(sid):
            w.writerow(["" if e[k] is None else e[k] for k in EVENT_COLUMNS])
        return out.getvalue()


def _kv(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items())


def _iso(v: datetime | None) -> str | None:
    return None if v is None else v.isoformat()


def _session_dict(s: SessionRow) -> dict:
    return {"id": s.id, "scene": s.scene, "scene_sha256": s.scene_sha256, "params": s.params,
            "model": s.model, "started_at": _iso(s.started_at), "ended_at": _iso(s.ended_at),
            "end_reason": s.end_reason, "error": s.error}  # fmt: skip


def _event_dict(r: EventRow) -> dict:
    d = {k: getattr(r, k) for k in EVENT_COLUMNS}
    d["detected_at"], d["decided_at"] = _iso(r.detected_at), _iso(r.decided_at)
    return d | {"session_id": r.session_id}
