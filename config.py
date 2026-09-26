"""
config.py  —  Central configuration for the Cross-Camera Re-ID system.

Default setup: 2 live cameras (Laptop Webcam + Phone Camera).

Camera source options:
  - source = 0, 1, 2 …          → laptop/USB webcam (device index)
  - source = "http://x.x.x.x:8080/video"  → Phone via IP Webcam app (Wi-Fi)
  - source = "rtsp://x.x.x.x:8554/live"  → Phone via RTSP stream app
  - source = "videos/cam0.avi"  → local video file (loops)

Environment variables override everything here (.env support):
  REID_API_KEY   – API key for dashboard auth
  REID_HOST      – bind address (default 0.0.0.0)
  REID_PORT      – bind port   (default 8000)
  CAM_0_SOURCE   – Laptop webcam source  (default: 0)
  CAM_0_ZONE     – Laptop webcam zone name
  CAM_0_NAME     – Laptop webcam display name
  CAM_1_SOURCE   – Phone camera source   (default: http://192.168.1.x:8080/video)
  CAM_1_ZONE     – Phone camera zone name
  CAM_1_NAME     – Phone camera display name
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

# ── Load .env if present (Phase 8) ────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass   # python-dotenv optional


# ─────────────────────────────────────────────────────────────────────────────
# Camera configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraConfig:
    camera_id: str
    name: str
    zone: str
    source: str          # file path, RTSP URL, or device index as string
    fps_override: Optional[float] = None  # set if cv2.CAP_PROP_FPS reports 0 or wrong value
    speed_multiplier: Optional[float] = None  # individual camera speed multiplier (default: CONFIG.video_speed)


def _cam_source(n: int, default: str) -> str:
    """Read CAM_n_SOURCE from env or fall back to default."""
    return os.environ.get(f"CAM_{n}_SOURCE", default)

def _cam_zone(n: int, default: str) -> str:
    return os.environ.get(f"CAM_{n}_ZONE", default)

def _cam_name(n: int, default: str) -> str:
    return os.environ.get(f"CAM_{n}_NAME", default)


# ─────────────────────────────────────────────────────────────────────────────
# System configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SystemConfig:
    # ── Server ────────────────────────────────────────────────────────────────
    host: str = field(default_factory=lambda: os.environ.get("REID_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("REID_PORT", "8000")))

    # ── Auth (Phase 7) ────────────────────────────────────────────────────────
    # Set REID_API_KEY env var or change this string.
    # Leave empty ("") to disable authentication (not recommended in production).
    api_key: str = field(
        default_factory=lambda: os.environ.get("REID_API_KEY", "")
    )

    # ── Re-ID model ───────────────────────────────────────────────────────────
    reid_checkpoint_path: str = "models/reid_resnet50_market1501.pth"
    embedding_dim: int = 512

    # ── Matching ──────────────────────────────────────────────────────────────
    # 0.78 → untrained baseline.  After running scripts/calibrate_threshold.py
    # on real footage, update this value.  Expected range: 0.82–0.88 for
    # the trained Market-1501 checkpoint.
    similarity_threshold: float = 0.78

    # Phase 9: multi-exemplar gallery — max vectors stored per (identity × camera).
    # Increasing this improves recall at the cost of memory + FAISS index size.
    # 8 is a good default for short demo sessions; use 16–32 for longer runs.
    max_exemplars_per_camera: int = 8

    # ── Playback Speed & Performance ──────────────────────────────────────────
    # Playback speed multiplier for video streams (1.2 = 1.2x real-time speed)
    video_speed: float = field(
        default_factory=lambda: float(os.environ.get("VIDEO_SPEED", "1.2"))
    )
    # Detection stride: run full YOLOv8 detection every N frames to maintain 1.2x pacing
    detect_stride: int = field(
        default_factory=lambda: int(os.environ.get("DETECT_STRIDE", "2"))
    )

    # ── Timing ────────────────────────────────────────────────────────────────
    send_interval_seconds: float = 0.5   # max embedding-send rate per track
    inactive_identity_ttl_seconds: float = 300.0  # 5 min before identity goes stale

    # ── Persistence (Phase 8) ─────────────────────────────────────────────────
    db_path: str = "data/reid_identities.db"   # SQLite database location

    # ── Cameras  (2 live cameras: Laptop Webcam + Phone Camera) ──────────────
    # Camera 0: Laptop built-in or USB webcam (device index 0).
    # Camera 1: Phone camera streamed over Wi-Fi via the "IP Webcam" app.
    #
    # HOW TO USE:
    #   Option A — Edit .env (no code change needed):
    #     CAM_0_SOURCE=0
    #     CAM_1_SOURCE=http://192.168.1.45:8080/video
    #
    #   Option B — Edit source= values below directly.
    #
    # PHONE SETUP (IP Webcam app, Android):
    #   1. Install "IP Webcam" from Play Store (by Pavel Khlebovich).
    #   2. Connect phone & laptop to the SAME Wi-Fi network.
    #   3. Open app → tap "Start server" at the bottom.
    #   4. Note the IP shown on screen, e.g. http://192.168.1.45:8080
    #   5. Replace the IP below (or in .env) with your phone's actual IP.
    #      Stream URL: http://<phone-ip>:8080/video
    #
    # PHONE SETUP (iOS — iVCam or Camo):
    #   Use iVCam (free) on iOS → installs a virtual webcam on Windows.
    #   Your phone becomes device index 1 or 2 automatically.
    #   Set CAM_1_SOURCE=1 (or 2) in .env.
    cameras: List[CameraConfig] = field(default_factory=lambda: [
        CameraConfig(
            camera_id="cam_0",
            name=_cam_name(0, "Laptop Webcam"),
            zone=_cam_zone(0, "Zone A"),
            source=_cam_source(0, "0"),  # Built-in laptop webcam (device index 0)
        ),
        CameraConfig(
            camera_id="cam_1",
            name=_cam_name(1, "Phone Camera"),
            zone=_cam_zone(1, "Zone B"),
            # Replace 192.168.1.45 with your phone's actual IP shown in IP Webcam app
            source=_cam_source(1, "http://192.168.1.45:8080/video"),
        ),
    ])


CONFIG = SystemConfig()
