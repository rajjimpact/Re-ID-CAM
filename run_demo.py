"""
run.py  —  Production entry-point for the Cross-Camera Re-ID System.

What this does
──────────────
1. Load .env if present.
2. Verify camera sources are accessible before starting.
3. Start FastAPI/Uvicorn in a background thread.
4. Wait for the server to respond on /health.
5. Launch one EdgeWorker per configured camera (staggered to share GPU).
6. Print the dashboard URL and block until Ctrl-C.

Camera sources
──────────────
Edit config.py (cameras list) OR set environment variables in .env:
  CAM_0_SOURCE=0                              <- laptop/USB webcam index
  CAM_0_SOURCE=rtsp://192.168.1.10/stream1   <- RTSP IP camera
  CAM_0_SOURCE=http://192.168.1.20:8080/video <- phone via IP Webcam app

Identity history is persisted to data/reid_identities.db
and restored automatically on the next startup.
"""
from __future__ import annotations
import os
import sys
import time
import threading
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(SCRIPT_DIR, ".env"))
except ImportError:
    pass


try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def check_source_available(source: str | int, timeout: float = 2.0) -> tuple[bool, str]:
    if isinstance(source, int) or str(source).isdigit():
        return True, "local device"
    s = str(source).strip()
    if s.startswith(("http://", "https://")):
        try:
            req = urllib.request.Request(s, headers={"User-Agent": "reid/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                chunk = resp.read(512)
                if chunk:
                    return True, "online"
                return False, "no data stream"
        except Exception as e:
            return False, str(e)
    return True, "file/stream"


# ── Camera connectivity check ─────────────────────────────────────────────────
def check_cameras():
    from config import CONFIG
    try:
        import cv2
    except ImportError:
        print("[run] WARNING: OpenCV not installed — cannot pre-check cameras.", flush=True)
        return

    print("[run] Checking camera sources...", flush=True)
    for cam in CONFIG.cameras:
        src = cam.source
        print(f"  Checking {cam.camera_id} ({cam.name}): {src} ...", end="", flush=True)

        ok, reason = check_source_available(src, timeout=2.0)
        if not ok:
            print(f" [WARN] Stream unreachable ({reason})", flush=True)
            continue

        try:
            src_int = int(src)
        except (ValueError, TypeError):
            src_int = None

        source = src_int if src_int is not None else src
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f" [OK] ({cam.zone}) {w}x{h}", flush=True)
        else:
            cap.release()
            print(f" [WARN] Could not open!", flush=True)
            print(f"         Check the stream URL or camera index in .env", flush=True)


# ── Server ─────────────────────────────────────────────────────────────────────
def start_server():
    import uvicorn
    from config import CONFIG

    # Use uvloop for lower-latency async I/O when available (Linux/Mac).
    # Falls back to the default asyncio loop on Windows transparently.
    loop_policy = "uvloop" if _uvloop_available() else "auto"

    uvicorn.run(
        "central.api:app",
        host=CONFIG.host,
        port=CONFIG.port,
        log_level="warning",
        access_log=False,
        loop=loop_policy,
        # Increase HTTP backlog for multiple simultaneous MJPEG streams
        backlog=256,
        # Keep-alive timeout — important for MJPEG streaming connections
        timeout_keep_alive=75,
    )


def _uvloop_available() -> bool:
    try:
        import uvloop  # noqa: F401
        return True
    except ImportError:
        return False


def wait_for_server(host: str, port: int, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.15)
    return False


# ── Edge workers ───────────────────────────────────────────────────────────────
def start_edge_workers():
    from config import CONFIG
    from central.api import ingest_sync
    from central.frame_hub import FRAME_HUB
    from edge.edge_worker import EdgeWorker

    workers = []
    for i, cam_cfg in enumerate(CONFIG.cameras):
        ok, reason = check_source_available(cam_cfg.source, timeout=1.5)
        if not ok:
            print(f"[run] Skipping {cam_cfg.camera_id} ({cam_cfg.name}): offline ({reason})", flush=True)
            continue

        if i > 0:
            time.sleep(1.0)    # stagger — one CUDA context at a time
        worker = EdgeWorker(
            camera_cfg=cam_cfg,
            ingest_fn=ingest_sync,
            frame_push_fn=FRAME_HUB.push,
        )
        thread = worker.start()
        workers.append((worker, thread))
        print(f"[run] Edge worker started: {cam_cfg.camera_id}  ({cam_cfg.zone})", flush=True)
    return workers



# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    from config import CONFIG

    print("=" * 62)
    print("  Cross-Camera Person Re-ID System  v2.0  [PRODUCTION]")
    print("  PyTorch + FAISS + FastAPI + YOLOv8 + CUDA")
    print("=" * 62)

    # 1. Create data/ directory for SQLite
    os.makedirs("data", exist_ok=True)

    # 2. Pre-check cameras
    check_cameras()

    # 3. Start Uvicorn (imports central.api which creates the DB + restores state)
    print(f"[run] Starting server on http://{CONFIG.host}:{CONFIG.port} ...")
    server_thread = threading.Thread(target=start_server, daemon=True, name="uvicorn")
    server_thread.start()

    print(f"[run] Waiting for server (up to 40s)...", end="", flush=True)
    if wait_for_server("127.0.0.1", CONFIG.port, timeout=40.0):
        print(" OK")
    else:
        print(" (timeout — continuing anyway)")

    # 4. Edge workers (after server is confirmed up)
    print("[run] Starting edge workers...")
    workers = start_edge_workers()

    auth_note = f"  Auth key: {CONFIG.api_key[:4]}****" if CONFIG.api_key else "  Auth: DISABLED (set REID_API_KEY to enable)"
    print()
    print("-" * 62)
    print(f"  Dashboard  ->  http://localhost:{CONFIG.port}/")
    print(f"  API docs   ->  http://localhost:{CONFIG.port}/docs")
    print(f"  DB         ->  {os.path.abspath(CONFIG.db_path)}")
    print(auth_note)
    print(f"  Press Ctrl-C to stop.")
    print("-" * 62)
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[run] Shutting down edge workers...")
        for worker, _ in workers:
            worker.stop()
        print("[run] Done.")


if __name__ == "__main__":
    main()
