"""
edge/detector_tracker.py — YOLOv8 + ByteTrack, with MOG2+centroid fallback.

Public API (both real and fallback return the same type):

    tracker = build_tracker()
    detections = tracker.update(frame)   # -> List[Detection]

Detection:
    track_id  : int   — stable local ID within this camera
    bbox      : [x1, y1, x2, y2]  — pixel coords
    confidence: float
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# ── Availability flags ────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ─────────────────────────────────────────────────────────────────────────────
# Shared result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    track_id: int
    bbox: List[float]       # [x1, y1, x2, y2]
    confidence: float


# ─────────────────────────────────────────────────────────────────────────────
# Real tracker: YOLOv8 + ByteTrack
# ─────────────────────────────────────────────────────────────────────────────

class YoloByteTracker:
    """
    Thin wrapper around ultralytics YOLOv8 with ByteTrack.
    model.track() returns both detections and persistent track IDs in one call.
    """

    def __init__(self, model_name: str = "yolov8n.pt", device: Optional[str] = None):
        self.model = _YOLO(model_name)
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

    def update(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection + tracking on one BGR frame.
        Returns list of Detection for every tracked person in the frame.
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],       # class 0 = person in COCO
            verbose=False,
            device=self.device,
        )
        detections: List[Detection] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                tid = boxes.id
                if tid is None:
                    continue
                track_id = int(tid[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i].item())
                detections.append(Detection(
                    track_id=track_id,
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                ))
        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Fallback tracker: MOG2 background subtraction + centroid tracking
# ─────────────────────────────────────────────────────────────────────────────

class Mog2CentroidTracker:
    """
    Dependency-light fallback when ultralytics is not installed.

    Pipeline:
      1. MOG2 background subtraction → foreground mask
      2. Morphological cleanup → connected components as candidate blobs
      3. Filter by area and aspect ratio to keep person-like blobs
      4. Centroid-to-centroid IoU/distance matching for ID persistence

    Returns the same Detection interface as YoloByteTracker.
    """

    _MIN_AREA = 1500       # px² — discard tiny noise blobs
    _MAX_AREA = 80_000     # px² — discard room-sized blobs
    _DIST_THRESHOLD = 80   # px  — max centroid distance for same-track match
    _MAX_DISAPPEARED = 15  # frames before a track is dropped

    def __init__(self) -> None:
        if not _HAS_CV2:
            raise RuntimeError("OpenCV (cv2) is required for Mog2CentroidTracker")
        self._bg = _cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )
        self._next_id = 1
        # track_id → {"centroid": (cx, cy), "bbox": [...], "disappeared": int}
        self._tracks: dict = {}

    # ── internal helpers ──────────────────────────────────────────────────────

    def _foreground_blobs(self, frame: np.ndarray):
        """Return list of (bbox, centroid) for person-like blobs."""
        mask = self._bg.apply(frame)
        kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (5, 5))
        mask = _cv2.morphologyEx(mask, _cv2.MORPH_OPEN, kernel, iterations=2)
        mask = _cv2.morphologyEx(mask, _cv2.MORPH_CLOSE, kernel, iterations=3)

        n_labels, _, stats, centroids = _cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        blobs = []
        for i in range(1, n_labels):  # 0 = background
            x, y, w, h = stats[i, :4]
            area = stats[i, _cv2.CC_STAT_AREA]
            if area < self._MIN_AREA or area > self._MAX_AREA:
                continue
            # Prefer tall blobs (standing persons have h > w)
            if h < w * 0.5:
                continue
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            blobs.append(([float(x), float(y), float(x + w), float(y + h)], (cx, cy)))
        return blobs

    @staticmethod
    def _dist(c1, c2) -> float:
        return float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> List[Detection]:
        blobs = self._foreground_blobs(frame)

        # Mark all existing tracks as disappeared for this frame
        for tid in self._tracks:
            self._tracks[tid]["disappeared"] += 1

        if blobs:
            unmatched_blobs = list(range(len(blobs)))
            unmatched_tracks = list(self._tracks.keys())

            # Greedy nearest-centroid matching
            if unmatched_tracks:
                matched_pairs = []
                for blob_idx, (_, centroid) in enumerate(blobs):
                    best_tid, best_d = None, self._DIST_THRESHOLD + 1
                    for tid in unmatched_tracks:
                        d = self._dist(centroid, self._tracks[tid]["centroid"])
                        if d < best_d:
                            best_d = d
                            best_tid = tid
                    if best_tid is not None and best_d <= self._DIST_THRESHOLD:
                        matched_pairs.append((blob_idx, best_tid))

                matched_blob_idxs = {b for b, _ in matched_pairs}
                matched_track_ids = {t for _, t in matched_pairs}

                for blob_idx, tid in matched_pairs:
                    bbox, centroid = blobs[blob_idx]
                    self._tracks[tid].update(
                        {"centroid": centroid, "bbox": bbox, "disappeared": 0}
                    )
                    if blob_idx in unmatched_blobs:
                        unmatched_blobs.remove(blob_idx)
                    if tid in unmatched_tracks:
                        unmatched_tracks.remove(tid)

                unmatched_blobs = [b for b in unmatched_blobs
                                   if b not in matched_blob_idxs]

            # Register new tracks for unmatched blobs
            for blob_idx in unmatched_blobs:
                bbox, centroid = blobs[blob_idx]
                self._tracks[self._next_id] = {
                    "centroid": centroid,
                    "bbox": bbox,
                    "disappeared": 0,
                }
                self._next_id += 1

        # Drop tracks that have been gone too long
        dead = [tid for tid, t in self._tracks.items()
                if t["disappeared"] > self._MAX_DISAPPEARED]
        for tid in dead:
            del self._tracks[tid]

        # Build Detection list from surviving tracks
        detections: List[Detection] = []
        for tid, t in self._tracks.items():
            if t["disappeared"] == 0:
                detections.append(Detection(
                    track_id=tid,
                    bbox=t["bbox"],
                    confidence=0.6,   # fixed synthetic confidence for fallback
                ))
        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_tracker(model_name: str = "yolov8n.pt",
                  device: Optional[str] = None):
    """
    Return the best available tracker.
    Both options produce the same List[Detection] output.
    """
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if _HAS_ULTRALYTICS:
        try:
            print(f"[detector_tracker] Using YOLOv8 + ByteTrack (ultralytics) on {device}")
            return YoloByteTracker(model_name=model_name, device=device)
        except Exception as e:
            print(f"[detector_tracker] YOLOv8 init failed ({e}), falling back to MOG2+centroid")
    print("[detector_tracker] Using MOG2 + centroid fallback tracker")
    return Mog2CentroidTracker()
