"""
central/frame_hub.py — Live annotated-frame buffer for MJPEG streaming.

Each camera worker pushes JPEG-encoded frames here.
The FastAPI /stream/{camera_id} endpoint reads them back as an MJPEG feed
that any <img> tag can consume directly (no WebSocket / JS needed for video).
"""
from __future__ import annotations
import threading
from collections import deque
from typing import Deque, Dict, Iterator, Optional

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class FrameHub:
    """
    Per-camera JPEG frame buffer + blocking iterator for MJPEG streaming.
    """

    def __init__(self, maxlen: int = 2) -> None:
        self._maxlen = maxlen
        self._buffers: Dict[str, Deque[bytes]] = {}
        self._events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def _ensure_camera(self, camera_id: str) -> None:
        if camera_id not in self._buffers:
            self._buffers[camera_id] = deque(maxlen=self._maxlen)
            self._events[camera_id] = threading.Event()

    def push(self, camera_id: str, frame) -> None:
        """
        Push a BGR numpy frame for camera_id.
        Encodes to JPEG and notifies any waiting MJPEG readers.
        """
        if not _HAS_CV2 or frame is None:
            return
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        jpeg_bytes = encoded.tobytes()
        with self._lock:
            self._ensure_camera(camera_id)
            self._buffers[camera_id].append(jpeg_bytes)
            self._events[camera_id].set()

    def latest_frame(self, camera_id: str) -> Optional[bytes]:
        with self._lock:
            self._ensure_camera(camera_id)
            if self._buffers[camera_id]:
                return self._buffers[camera_id][-1]
        return None

    def mjpeg_stream(self, camera_id: str, timeout: float = 5.0) -> Iterator[bytes]:
        """
        Generator that yields multipart MJPEG chunks.
        Blocks until a new frame is available (with a timeout to avoid spin).
        """
        with self._lock:
            self._ensure_camera(camera_id)

        while True:
            event = self._events[camera_id]
            event.wait(timeout=timeout)
            frame_bytes = self.latest_frame(camera_id)
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
            with self._lock:
                self._events[camera_id].clear()

    def camera_ids(self):
        with self._lock:
            return list(self._buffers.keys())


# Module-level singleton, shared by api.py and edge workers
FRAME_HUB = FrameHub()
