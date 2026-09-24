"""
run_demo.py  —  Single entry-point for the Cross-Camera Re-ID System.
                Runs everything in one process (Phase 5+).

What this does
──────────────
1. Load .env if present (Phase 8: environment config).
2. Verify camera sources are accessible before starting.
3. Start FastAPI/Uvicorn in a background thread.
4. Wait for the server to respond.
5. Launch one EdgeWorker per configured camera (staggered).
6. Print the dashboard URL and block until Ctrl-C.

Camera sources
──────────────
Edit config.py (cameras list) OR set environment variables:
  CAM_0_SOURCE=0            <- laptop webcam
  CAM_0_SOURCE=videos/cam0.avi
  CAM_0_SOURCE=rtsp://192.168.1.10/stream1

Phase 8: Identity history is persisted to data/reid_identities.db
         and restored on the next startup automatically.
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


# ── Camera connectivity check ─────────────────────────────────────────────────
def check_cameras():
    from config import CONFIG
    try:
        import cv2
    except ImportError:
        print("[run] WARNING: OpenCV not installed — cannot pre-check cameras.")
        return

    print("[run] Checking camera sources...")
    for cam in CONFIG.cameras:
        src = cam.source
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
            print(f"  [OK] {cam.camera_id} ({cam.zone})  source={src!r}  {w}x{h}")
        else:
            cap.release()
            print(f"  [WARN] {cam.camera_id} ({cam.zone})  source={src!r}  -- COULD NOT OPEN")
            print(f"         Check the source path or device index in config.py / .env")


# ── Server ─────────────────────────────────────────────────────────────────────
def start_server():
    import uvicorn
    from config import CONFIG
    uvicorn.run(
        "central.api:app",
        host=CONFIG.host,
        port=CONFIG.port,
        log_level="warning",
        access_log=False,
    )


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
        if i > 0:
            time.sleep(1.5)    # stagger — one CUDA context at a time
        worker = EdgeWorker(
            camera_cfg=cam_cfg,
            ingest_fn=ingest_sync,
            frame_push_fn=FRAME_HUB.push,
        )
        thread = worker.start()
        workers.append((worker, thread))
        print(f"[run] Edge worker started: {cam_cfg.camera_id}  ({cam_cfg.zone})")
    return workers


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    from config import CONFIG

    print("=" * 62)
    print("  Cross-Camera Person Re-ID System  v2.0")
    print("  PyTorch + FAISS + FastAPI + YOLOv8 + CUDA")
    print("=" * 62)

    # 1. Create data/ directory for SQLite
    os.makedirs("data", exist_ok=True)

    # 2. Pre-check cameras
    check_cameras()

    # 3. Start Uvicorn (imports central.api which creates the DB + restores state)
    print(f"\n[run] Starting server on http://{CONFIG.host}:{CONFIG.port} ...")
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
