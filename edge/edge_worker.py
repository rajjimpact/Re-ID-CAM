"""
edge/edge_worker.py — Orchestrates one camera end-to-end.

For each camera:
  1. OpenCV VideoCapture (file, RTSP, or device index)
  2. Primary tracker (YOLOv8+ByteTrack) → List[Detection]
     If primary finds nothing for _FALLBACK_TRIGGER frames in a row,
     MOG2+centroid fallback is used on the same frame.
  3. crop + reid_embedder → 512-d vector
  4. pack TrackPayload → send to central (in-process or HTTP)
  5. push annotated frame to frame_hub (EVERY frame, regardless of detections)

Throttled to CONFIG.send_interval_seconds per track.

Playback pacing (file sources only)
────────────────────────────────────
For file sources the loop runs with elapsed-time-aware pacing so playback
matches the video's native wall-clock speed:

    target_interval = 1 / effective_fps
    remaining = target_interval - elapsed_this_loop
    if remaining > 0: wait(remaining)   # otherwise proceed immediately

Live sources (RTSP / HTTP / webcam device index) arrive at real-time pace on
their own — adding a wait on top would double-slow them.  Those are NOT paced.

fps_override in CameraConfig overrides CAP_PROP_FPS for files whose codec
reports 0 or an incorrect value (common with older EPFL .avi files).
"""
from __future__ import annotations
import sys
import os
import time
import threading
from typing import Callable, Optional, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG, CameraConfig
from edge.detector_tracker import build_tracker, Mog2CentroidTracker, Detection
from edge.reid_embedder import build_embedder

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# Annotation colours (BGR)
_COL_BOX_YOLO  = (0, 255, 180)     # cyan-green  — real YOLOv8 detections
_COL_BOX_MOG2  = (80, 160, 255)    # amber-orange — MOG2 fallback detections
_COL_TEXT_BG   = (0, 40, 40)
_COL_TEXT      = (200, 255, 220)

# After this many consecutive frames with zero YOLO detections, MOG2 kicks in
_FALLBACK_TRIGGER = 10

# Minimum/maximum effective FPS clamp for file pacing
_FPS_MIN = 1.0
_FPS_MAX = 120.0


def _is_live_source(source) -> bool:
    """
    Return True if source is a live stream (webcam device index or network URL).
    Live streams arrive at real-time pace — they must NOT be paced with a wait.
    File paths are paced to match the video's native speed.
    """
    # Integer device index → live webcam
    if isinstance(source, int):
        return True
    s = str(source)
    # Network streams
    if s.startswith(("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://")):
        return True
    return False


class EdgeWorker:
    """
    Runs the full per-camera pipeline in its own thread.

    Hybrid detection strategy
    ─────────────────────────
    YOLOv8 is tried first on every frame.  If it returns 0 detections for
    _FALLBACK_TRIGGER consecutive frames, the MOG2+centroid tracker is used
    instead.  Both trackers produce the same Detection interface so the rest
    of the pipeline is identical in both cases.
    """

    def __init__(
        self,
        camera_cfg: CameraConfig,
        ingest_fn: Callable[[dict], None],
        frame_push_fn: Optional[Callable] = None,
    ) -> None:
        self.cfg = camera_cfg
        self._ingest = ingest_fn
        self._frame_push = frame_push_fn
        self._stop_event = threading.Event()

        # Per-track throttle
        self._last_sent: dict[int, float] = {}

        # Primary tracker (YOLOv8+ByteTrack or MOG2 if ultralytics missing)
        self.tracker = build_tracker()
        # Secondary fallback — always MOG2, used when primary finds nothing
        self._mog2 = Mog2CentroidTracker()
        self._zero_detection_streak = 0
        self._using_fallback = False

        # Embedder — loads trained checkpoint (§10.5)
        self.embedder = build_embedder(
            embedding_dim=CONFIG.embedding_dim,
            checkpoint_path=CONFIG.reid_checkpoint_path,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self) -> threading.Thread:
        t = threading.Thread(
            target=self._run,
            name=f"edge-{self.cfg.camera_id}",
            daemon=True,
        )
        t.start()
        return t

    def stop(self) -> None:
        self._stop_event.set()

    # ── Detection with hybrid fallback ────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> tuple[List[Detection], bool]:
        """
        Returns (detections, used_fallback).
        Always keeps MOG2 trained regardless of which tracker is active,
        so it's ready immediately when YOLOv8 switches off.
        """
        primary_dets = self.tracker.update(frame)

        if primary_dets:
            # Primary found something — keep MOG2 trained but ignore its output
            self._mog2.update(frame)
            self._zero_detection_streak = 0
            self._using_fallback = False
            return primary_dets, False
        else:
            self._zero_detection_streak += 1
            fallback_dets = self._mog2.update(frame)

            if self._zero_detection_streak >= _FALLBACK_TRIGGER and fallback_dets:
                if not self._using_fallback:
                    print(
                        f"[{self.cfg.camera_id}] YOLOv8 found nothing for "
                        f"{self._zero_detection_streak} frames — "
                        f"switching to MOG2+centroid fallback."
                    )
                self._using_fallback = True
                return fallback_dets, True

            return [], False

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        if not _HAS_CV2:
            print(f"[{self.cfg.camera_id}] OpenCV not available — cannot run.")
            return

        source = self.cfg.source
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[{self.cfg.camera_id}] Could not open source: {self.cfg.source!r}")
            return

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        # ── Determine playback pacing & speed ─────────────────────────────────
        live = _is_live_source(source)
        speed = (
            self.cfg.speed_multiplier
            if getattr(self.cfg, "speed_multiplier", None) is not None
            else getattr(CONFIG, "video_speed", 1.2)
        )
        if speed <= 0:
            speed = 1.0

        if not live:
            # Prefer explicit override, then codec-reported FPS, then default 25
            if self.cfg.fps_override and self.cfg.fps_override > 0:
                native_fps = float(self.cfg.fps_override)
                fps_src = "override"
            else:
                raw_fps = cap.get(cv2.CAP_PROP_FPS)
                if raw_fps and raw_fps > 0:
                    native_fps = float(raw_fps)
                    fps_src = "codec"
                else:
                    native_fps = 25.0
                    fps_src = "default (codec reported 0)"

            native_fps = max(_FPS_MIN, min(_FPS_MAX, native_fps))
            effective_fps = native_fps * speed
            target_interval = 1.0 / effective_fps
            print(
                f"[{self.cfg.camera_id}] Started — zone={self.cfg.zone!r} "
                f"source={self.cfg.source!r} {fw}x{fh} "
                f"native_fps={native_fps:.2f} ({fps_src}) "
                f"speed={speed:.1f}x -> effective_fps={effective_fps:.2f}"
            )
        else:
            target_interval = None   # live — no artificial pacing
            print(
                f"[{self.cfg.camera_id}] Started — zone={self.cfg.zone!r} "
                f"source={self.cfg.source!r} {fw}x{fh} [live stream — no pacing]"
            )

        next_frame_time = time.time()
        frame_idx = 0
        last_detections: List[Detection] = []
        last_fallback: bool = False
        # Live cameras: always run detection on every frame (stride=1) for
        # real-time accuracy. File playback respects CONFIG.detect_stride.
        detect_stride = 1 if live else getattr(CONFIG, "detect_stride", 1)

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if not live:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)    # loop video files
                    next_frame_time = time.time()
                time.sleep(0.02)
                continue

            frame_idx += 1
            now = time.time()

            # ── Detect + track (with stride to maintain 1.2x real-time pace) ───
            if frame_idx % detect_stride == 0 or not last_detections:
                detections, used_fallback = self._detect(frame)
                last_detections = detections
                last_fallback = used_fallback
            else:
                detections = last_detections
                used_fallback = last_fallback

            # ── Annotate frame ────────────────────────────────────────────────
            annotated = frame.copy()
            col_box = _COL_BOX_MOG2 if used_fallback else _COL_BOX_YOLO

            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fw, x2), min(fh, y2)

                # ── Throttle: send embedding at most 2× per second ────────────
                last = self._last_sent.get(det.track_id, 0.0)
                if now - last >= CONFIG.send_interval_seconds:
                    self._last_sent[det.track_id] = now
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        embedding = self.embedder.embed(crop)
                        payload = {
                            "camera_id":      self.cfg.camera_id,
                            "local_track_id": det.track_id,
                            "timestamp":      now,
                            "bbox":           [float(x1), float(y1), float(x2), float(y2)],
                            "embedding":      embedding.tolist(),
                            "zone":           self.cfg.zone,
                            "frame_width":    fw,
                            "frame_height":   fh,
                        }
                        try:
                            self._ingest(payload)
                        except Exception as e:
                            print(f"[{self.cfg.camera_id}] ingest error: {e}")

                # Bounding box + track ID label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), col_box, 2)
                label = f"T{det.track_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                              _COL_TEXT_BG, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COL_TEXT, 1, cv2.LINE_AA)

            # Zone label overlay
            tracker_tag = " [MOG2]" if used_fallback else ""
            cv2.putText(annotated, self.cfg.zone + tracker_tag, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col_box, 2, cv2.LINE_AA)

            # ── Push frame ALWAYS — video streams regardless of detections ────
            if self._frame_push:
                self._frame_push(self.cfg.camera_id, annotated)

            # ── Wall-clock pacing (file sources only) ─────────────────────────
            # Enforces exact 1.2x playback speed. If processing falls behind,
            # skips excess frames to immediately resynchronize with wall-clock time.
            if target_interval is not None:
                next_frame_time += target_interval
                now_check = time.time()
                delay = next_frame_time - now_check
                if delay > 0:
                    self._stop_event.wait(delay)
                elif delay < -0.15:
                    # Dropped behind by > 150ms -> skip frames to catch up
                    skip = int((-delay) / target_interval)
                    for _ in range(min(skip, 10)):
                        cap.grab()
                    next_frame_time = time.time()
            else:
                # Live stream: no artificial sleep — let the camera driver
                # pace the loop naturally via cap.read() blocking.
                pass


        cap.release()
        print(f"[{self.cfg.camera_id}] Stopped.")
