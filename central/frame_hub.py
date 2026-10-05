"""
central/frame_hub.py — Low-latency annotated-frame buffer for MJPEG streaming.

Production design:
  - Buffer depth = 1  → always the most recent frame, never stale.
  - JPEG quality = 85 → crisp enough for surveillance, small enough for LAN.
  - threading.Event per camera → MJPEG streamer wakes *instantly* when a new
    frame arrives instead of polling on a fixed timer.
  - Frame is only re-encoded when the worker pushes; the API just reads bytes.
"""
from __future__ import annotations
import threading
from typing import Dict, Iterator, Optional

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# JPEG quality for the MJPEG stream (0-100). 85 is a good balance of
# sharpness vs bandwidth on a LAN.  Lower to 70 if bandwidth is tight.
_JPEG_QUALITY = 85


class FrameHub:
    """
    Per-camera latest-frame store + event-driven MJPEG streaming.

    Thread safety
    ─────────────
    push() is called from EdgeWorker daemon threads.
    latest_frame() / mjpeg_stream() are called from asyncio (Uvicorn) threads.
    A single RLock guards all mutations; reads under the same lock.
    """

    def __init__(self) -> None:
        # Stores only the *latest* JPEG bytes per camera (depth-1 buffer).
        self._latest: Dict[str, bytes] = {}
        # One Event per camera; set whenever a new frame arrives.
        self._events: Dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_camera(self, camera_id: str) -> None:
        """Create per-camera state if not already present (lock must be held)."""
        if camera_id not in self._events:
            self._latest[camera_id] = b""
            self._events[camera_id] = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, camera_id: str, frame) -> None:
        """
        Encode *frame* (BGR numpy array) to JPEG and store as latest.
        Wakes any MJPEG stream generator waiting on this camera.
        """
        if not _HAS_CV2 or frame is None:
            return
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
        )
        if not ok:
            return
        jpeg_bytes = encoded.tobytes()
        with self._lock:
            self._ensure_camera(camera_id)
            self._latest[camera_id] = jpeg_bytes
            self._events[camera_id].set()   # wake the MJPEG generator immediately

    def latest_frame(self, camera_id: str) -> Optional[bytes]:
        """Return the most recent JPEG bytes, or None if no frame yet."""
        with self._lock:
            self._ensure_camera(camera_id)
            data = self._latest.get(camera_id, b"")
            return data if data else None

    def wait_for_frame(
        self,
        camera_id: str,
        last_seen: Optional[bytes],
        timeout: float = 2.0,
    ) -> Optional[bytes]:
        """
        Block until a frame *different* from *last_seen* is available,
        or until *timeout* seconds have elapsed.
        Returns the new frame bytes, or *last_seen* on timeout.
        """
        with self._lock:
            self._ensure_camera(camera_id)
            event = self._events[camera_id]
            current = self._latest.get(camera_id, b"")

        if current and current is not last_seen:
            return current

        # Clear before waiting so we don't miss the next set()
        event.clear()
        event.wait(timeout=timeout)

        with self._lock:
            data = self._latest.get(camera_id, b"")
            return data if data else last_seen

    def camera_ids(self) -> list:
        with self._lock:
            return list(self._events.keys())


# Module-level singleton shared by api.py and edge workers.
FRAME_HUB = FrameHub()
