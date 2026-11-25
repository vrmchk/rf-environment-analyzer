"""NFR-4: measure the waterfall update rate and the delay from an IQ block to the client.

Connects to a running system as the operator's client does (WebSocket /ws), starts a
session on a scene and, for every waterfall row, takes the time from the moment its IQ
block arrived in the backend (`wall` in the row header) to the moment the row is received
here. Both clocks are the host's (the backend in Docker Desktop shares it), so the delay
includes the processing, the queues and the network hop, as in the requirement.

Usage: uv run python -m rfea_backend.measure_latency --scene eval1 --seconds 60
       [--url ws://localhost:8080/ws] [--out latency.json]
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import websockets


async def measure(url: str, scene: str, seconds: float) -> dict:
    delays, arrivals, events = [], [], 0
    async with websockets.connect(url, max_size=2**24) as ws:
        await ws.send(json.dumps({"type": "start", "scene": scene, "params": {}}))
        t_end = time.time() + seconds
        started = False
        while time.time() < t_end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
            except TimeoutError:
                break
            now = time.time()
            if isinstance(msg, bytes):
                row = msgpack.unpackb(msg)
                delays.append(now - row["wall"])
                arrivals.append(now)
                continue
            m = json.loads(msg)
            if m["type"] == "error":
                raise SystemExit(f"server: {m['message']}")
            if m["type"] == "session_started":
                started = True
            elif m["type"] == "event":
                events += 1
            elif m["type"] == "session_ended":
                break
        if started:
            await ws.send(json.dumps({"type": "stop"}))
    d = np.array(delays) * 1000
    gaps = np.diff(arrivals)
    span = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0
    return {
        "scene": scene, "rows": len(d), "events": events,
        "rows_per_s": (len(d) - 1) / span if span else None,
        "delay_ms": {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)),
                     "p99": float(np.percentile(d, 99)), "max": float(d.max())},
        "row_gap_ms_max": float(gaps.max() * 1000) if len(gaps) else None,
        "nfr4": {"delay_max_1s": bool(d.max() <= 1000),
                 "rate_min_10": bool(span and (len(d) - 1) / span >= 10)},
    }  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://localhost:8080/ws")
    p.add_argument("--scene", required=True)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    r = asyncio.run(measure(a.url, a.scene, a.seconds))
    text = json.dumps(r, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
