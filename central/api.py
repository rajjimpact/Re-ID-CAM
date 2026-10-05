"""
central/api.py  —  FastAPI application (Phases 5, 6, 7, 8, 9).

Endpoints
─────────
POST /ingest                   Edge workers submit TrackPayload (internal use).
GET  /shoppers                 List all Global IDs with summary.
GET  /shoppers/export.csv      Download all shopper data as CSV (Phase 6).
GET  /shoppers/{id}            Full visit history for one shopper.
GET  /analytics/store          Store-wide metrics.
GET  /cameras                  Camera / zone configuration list.
GET  /stream/{camera_id}       Live MJPEG feed (annotated).
WS   /ws                       Live push of every identity update.
GET  /health                   Unauthenticated health check.
GET  /                         Serves the dashboard HTML.
POST /registry/reset           Flush all identities + truncate SQLite. Auth required.
GET  /registry/exemplars       Per-identity gallery stats. Auth required.

Auth (Phase 7)
──────────────
All endpoints except /health and / require Authorization: Bearer <REID_API_KEY>.
Set REID_API_KEY env var (or config.py api_key field).
Leave api_key empty to disable auth during local development.
"""
from __future__ import annotations
import asyncio
import csv
import io
import json
import os
import time
from typing import Optional, Set

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import (
    FileResponse, HTMLResponse, StreamingResponse
)
from fastapi.staticfiles import StaticFiles

from config import CONFIG
from central.schemas import TrackPayload
from central.reid_engine import GlobalIdentityRegistry
from central.analytics import compute_store_analytics, compute_shopper_summary
from central.frame_hub import FRAME_HUB
from central.auth import require_key
from central.persistence import IdentityStore

# ─────────────────────────────────────────────────────────────────────────────
# App + singletons
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Cross-Camera Re-ID API", version="2.0.0")

# Phase 8a — SQLite store
os.makedirs(os.path.dirname(CONFIG.db_path) or ".", exist_ok=True)
STORE = IdentityStore(CONFIG.db_path)

REGISTRY = GlobalIdentityRegistry(
    embedding_dim=CONFIG.embedding_dim,
    similarity_threshold=CONFIG.similarity_threshold,
    inactive_ttl_seconds=CONFIG.inactive_identity_ttl_seconds,
    max_exemplars_per_camera=CONFIG.max_exemplars_per_camera,
    store=STORE,
)

_ws_clients: Set[WebSocket] = set()
_ws_lock = asyncio.Lock()

# Captured event loop for thread-safe broadcasts (fix from Phase 3)
_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


@app.on_event("startup")
async def _on_startup() -> None:
    global _EVENT_LOOP
    _EVENT_LOOP = asyncio.get_running_loop()
    n = len(REGISTRY.get_all())
    print(f"[api] FastAPI ready. Restored {n} identities from DB. "
          f"Auth={'enabled' if CONFIG.api_key else 'disabled (dev mode)'}.")


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    STORE.close()


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket broadcast helper
# ─────────────────────────────────────────────────────────────────────────────

async def _broadcast(message: dict) -> None:
    if not _ws_clients:
        return
    data = json.dumps(message)
    dead: Set[WebSocket] = set()
    async with _ws_lock:
        for ws in list(_ws_clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.add(ws)
        _ws_clients.difference_update(dead)


# ─────────────────────────────────────────────────────────────────────────────
# In-process ingest (called from edge worker threads)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_sync(payload_dict: dict) -> None:
    """
    Called from edge worker daemon threads.  Must NOT use await.
    Returns (global_id, is_cross_camera, match_score) via registry.ingest().
    """
    tp = TrackPayload(
        camera_id=payload_dict["camera_id"],
        local_track_id=payload_dict["local_track_id"],
        timestamp=payload_dict["timestamp"],
        bbox=payload_dict["bbox"],
        embedding=payload_dict["embedding"],
        zone=payload_dict["zone"],
        frame_width=payload_dict.get("frame_width", 0),
        frame_height=payload_dict.get("frame_height", 0),
    )
    global_id, is_cross_camera, match_score = REGISTRY.ingest(tp)
    identity = REGISTRY.get_by_id(global_id)

    msg = {
        "type": "identity_update",
        "global_id": global_id,
        "cross_camera_match": is_cross_camera,
        "match_score": round(match_score, 4),
        "camera_id": tp.camera_id,
        "zone": tp.zone,
        "timestamp": tp.timestamp,
        "summary": identity.to_summary_dict() if identity else {},
    }

    if _EVENT_LOOP is not None and _EVENT_LOOP.is_running():
        asyncio.run_coroutine_threadsafe(_broadcast(msg), _EVENT_LOOP)


# ─────────────────────────────────────────────────────────────────────────────
# Health check (no auth needed)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "shoppers": len(REGISTRY.get_all())}


# ─────────────────────────────────────────────────────────────────────────────
# Shopper endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/shoppers")
async def list_shoppers(_: None = Depends(require_key)):
    return compute_shopper_summary(REGISTRY.get_all())


@app.get("/shoppers/export.csv")
async def export_shoppers_csv(_: None = Depends(require_key)):
    """Phase 6: Download full shopper history as CSV."""
    identities = REGISTRY.get_all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "global_id", "first_seen", "last_seen", "active",
        "total_dwell_seconds", "zone_visit_count",
        "zone", "camera_id", "enter_time", "exit_time", "dwell_seconds"
    ])
    for identity in sorted(identities, key=lambda i: i.global_id):
        for visit in identity.zone_visits:
            writer.writerow([
                identity.global_id,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(identity.first_seen)),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(identity.last_seen)),
                identity.active,
                round(identity.total_dwell_seconds, 1),
                len(identity.zone_visits),
                visit.zone,
                visit.camera_id,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(visit.enter_time)),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(visit.exit_time)) if visit.exit_time else "",
                round(visit.dwell_seconds, 1),
            ])
    output.seek(0)
    filename = f"shoppers_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/shoppers/{global_id}")
async def get_shopper(global_id: str, _: None = Depends(require_key)):
    identity = REGISTRY.get_by_id(global_id)
    if not identity:
        raise HTTPException(status_code=404, detail=f"Shopper {global_id!r} not found.")
    return identity.to_detail_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/analytics/store")
async def store_analytics(_: None = Depends(require_key)):
    return compute_store_analytics(REGISTRY.get_all())


# ─────────────────────────────────────────────────────────────────────────────
# Re-ID Register  (Phase 9)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/registry/reset")
async def registry_reset(_: None = Depends(require_key)):
    """
    Flush all in-memory identities and truncate the SQLite tables.
    The identity counter restarts from GSI-0001.
    """
    REGISTRY.reset()
    n = REGISTRY.identity_count()
    return {"status": "ok", "identities_remaining": n}


@app.get("/registry/exemplars")
async def registry_exemplars(_: None = Depends(require_key)):
    """
    Return per-identity gallery breakdown for the Re-ID Register panel.
    Each entry: { global_id, first_seen, last_seen, active,
                  total_exemplars, per_camera: {cam_id: count} }
    """
    return REGISTRY.exemplar_stats()


# ─────────────────────────────────────────────────────────────────────────────
# Camera list
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/cameras")
async def list_cameras(_: None = Depends(require_key)):
    return [
        {"camera_id": c.camera_id, "name": c.name, "zone": c.zone}
        for c in CONFIG.cameras
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MJPEG stream — async generator (Phase 3 fix retained)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/stream/{camera_id}")
async def stream(camera_id: str):
    # No auth on streams — browsers load <img src=...> without custom headers
    async def _gen():
        last_frame: Optional[bytes] = None
        while True:
            # Block in a thread until a *new* frame is ready (event-driven, ~0ms latency).
            # Falls back after 2 s so the generator doesn't hang if the camera drops.
            frame_bytes: Optional[bytes] = await asyncio.to_thread(
                FRAME_HUB.wait_for_frame, camera_id, last_frame, 2.0
            )
            if frame_bytes and frame_bytes is not last_frame:
                last_frame = frame_bytes
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )

    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — live identity updates
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, key: str = ""):
    # WebSocket auth: check ?key= query param manually (HTTP headers unavailable)
    configured_key = CONFIG.api_key.strip()
    if configured_key and key != configured_key:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    async with _ws_lock:
        _ws_clients.add(websocket)
    try:
        identities = REGISTRY.get_all()
        await websocket.send_text(json.dumps({
            "type": "initial_state",
            "shoppers": compute_shopper_summary(identities),
            "analytics": compute_store_analytics(identities),
        }))
        while True:
            await asyncio.sleep(20)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        async with _ws_lock:
            _ws_clients.discard(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard static files
# ─────────────────────────────────────────────────────────────────────────────

_DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")


@app.get("/")
async def dashboard():
    # If auth is enabled, redirect to login page when no session
    index = os.path.join(_DASHBOARD_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return HTMLResponse("<h1>Dashboard not found.</h1>")


@app.get("/login")
async def login_page():
    login = os.path.join(_DASHBOARD_DIR, "login.html")
    if os.path.exists(login):
        return FileResponse(login)
    return HTMLResponse("<h1>Login page not found.</h1>")


if os.path.isdir(_DASHBOARD_DIR):
    app.mount("/static", StaticFiles(directory=_DASHBOARD_DIR), name="static")
