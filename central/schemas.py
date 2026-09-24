"""
central/schemas.py — Shared data structures.

TrackPayload  : what each edge worker sends to central.
ZoneVisit     : one contiguous stay in one zone.
GlobalIdentity: the persistent cross-camera identity object.

Phase 9 changes
───────────────
  GlobalIdentity.gallery_json replaces the old embedding_sum running-average
  field.  It stores a list of [camera_id, vector] pairs — one entry per
  gallery exemplar (multiple per camera_id allowed, up to
  max_exemplars_per_camera).

  embedding_sum is kept as a deprecated alias (empty list) so that any code
  still referencing it doesn't crash.  The persistence layer uses gallery_json
  exclusively.

  gallery_size property added — convenience for dashboard display.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class TrackPayload:
    """Payload sent by an edge worker to the central engine."""
    camera_id: str
    local_track_id: int
    timestamp: float          # unix epoch seconds
    bbox: List[float]         # [x1, y1, x2, y2] in pixel coords
    embedding: List[float]    # 512-d L2-normalised appearance vector
    zone: str                 # zone name of the camera (from CameraConfig)
    frame_width: int = 0
    frame_height: int = 0


@dataclass
class ZoneVisit:
    """One contiguous stay of a global identity in one zone."""
    zone: str
    camera_id: str
    enter_time: float
    exit_time: Optional[float] = None  # None while still in zone

    @property
    def dwell_seconds(self) -> float:
        end = self.exit_time if self.exit_time is not None else time.time()
        return max(0.0, end - self.enter_time)

    def close(self, at: Optional[float] = None) -> None:
        self.exit_time = at if at is not None else time.time()

    def to_dict(self) -> dict:
        return {
            "zone":           self.zone,
            "camera_id":      self.camera_id,
            "enter_time":     self.enter_time,
            "exit_time":      self.exit_time,
            "dwell_seconds":  round(self.dwell_seconds, 1),
        }


@dataclass
class GlobalIdentity:
    """Persistent cross-camera identity — one real person."""
    global_id: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float  = field(default_factory=time.time)
    last_camera_id: str = ""
    last_bbox: List[float] = field(default_factory=list)

    # Phase 9: multi-exemplar gallery stored as list of [cam_id, vector] pairs.
    # Each entry: ["cam_0", [0.12, -0.34, ...]]
    gallery_json: List = field(default_factory=list)

    # Total exemplar count (kept in sync by reid_engine)
    embedding_count: int = 0

    # Deprecated alias — kept for backward compat with any code that referenced
    # embedding_sum.  Always returns an empty list; writes are silently ignored.
    @property
    def embedding_sum(self) -> List[float]:  # type: ignore[override]
        return []

    @embedding_sum.setter
    def embedding_sum(self, value: List[float]) -> None:  # type: ignore[override]
        pass  # no-op — data is in gallery_json

    zone_visits: List[ZoneVisit] = field(default_factory=list)
    active: bool = True

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def gallery_size(self) -> int:
        """Total number of exemplar vectors in the gallery."""
        return self.embedding_count

    @property
    def current_zone(self) -> Optional[str]:
        if self.zone_visits and self.zone_visits[-1].exit_time is None:
            return self.zone_visits[-1].zone
        return None

    @property
    def path(self) -> List[str]:
        return [v.zone for v in self.zone_visits]

    @property
    def total_dwell_seconds(self) -> float:
        return sum(v.dwell_seconds for v in self.zone_visits)

    # Minimum time a person must stay in a zone before we record a new one.
    # Prevents rapid toggling when the same person is seen on multiple cameras.
    _MIN_DWELL_SECONDS: float = 3.0

    def open_zone_visit(self, zone: str, camera_id: str, at: float) -> None:
        """Close any open visit and open a new one — only if zone actually changed
        and the current visit has been open long enough (anti-jitter)."""
        if self.zone_visits:
            last = self.zone_visits[-1]
            if last.exit_time is None:
                if last.zone == zone:
                    # Same zone — just update the camera seen on, don't re-open
                    last.camera_id = camera_id
                    return
                # Different zone — only close if current visit is old enough
                if at - last.enter_time < self._MIN_DWELL_SECONDS:
                    return   # too soon — ignore this zone flicker
                last.close(at=at)
        self.zone_visits.append(ZoneVisit(zone=zone, camera_id=camera_id, enter_time=at))

    def close_current_visit(self, at: Optional[float] = None) -> None:
        if self.zone_visits and self.zone_visits[-1].exit_time is None:
            self.zone_visits[-1].close(at=at)

    def to_summary_dict(self) -> dict:
        return {
            "global_id":           self.global_id,
            "first_seen":          self.first_seen,
            "last_seen":           self.last_seen,
            "active":              self.active,
            "current_zone":        self.current_zone,
            "last_camera_id":      self.last_camera_id,
            "path":                self.path,
            "total_dwell_seconds": round(self.total_dwell_seconds, 1),
            "zone_visit_count":    len(self.zone_visits),
            "gallery_size":        self.gallery_size,
        }

    def to_detail_dict(self) -> dict:
        d = self.to_summary_dict()
        d["zone_visits"] = [v.to_dict() for v in self.zone_visits]
        return d
