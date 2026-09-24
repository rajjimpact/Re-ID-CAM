# Cross-Camera Person Re-ID System

**Multi-camera video analytics with appearance-based person re-identification.**

People are detected in each camera feed (YOLOv8), tracked locally (ByteTrack), embedded into 512-dimensional appearance vectors (ResNet-50 fine-tuned on Market-1501), and matched across cameras using FAISS cosine similarity — resolving isolated, per-camera local IDs into unified, persistent **Global Shopper IDs**.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Comprehensive Feature Guide](#comprehensive-feature-guide)
   - [1. Edge Processing & Hardware Acceleration](#1-edge-processing--hardware-acceleration)
   - [2. Multi-Stage Person Tracking & Fallbacks](#2-multi-stage-person-tracking--fallbacks)
   - [3. Deep Learning Feature Embedder (Market-1501)](#3-deep-learning-feature-embedder-market-1501)
   - [4. Pacing Engine & 1.2x Video Speed Synchronization](#4-pacing-engine--12x-video-speed-synchronization)
   - [5. Central Re-ID Engine & FAISS Vector Matching](#5-central-re-id-engine--faiss-vector-matching)
   - [6. Multi-Exemplar Gallery & Identity Lifecycle](#6-multi-exemplar-gallery--identity-lifecycle)
   - [7. Retail & Facility Analytics Engine](#7-retail--facility-analytics-engine)
   - [8. Real-Time Frame Hub & MJPEG Streaming](#8-real-time-frame-hub--mjpeg-streaming)
   - [9. Operations Dashboard](#9-operations-dashboard)
   - [10. SQLite Persistence & Crash Recovery](#10-sqlite-persistence--crash-recovery)
   - [11. API Key Authentication & Security](#11-api-key-authentication--security)
   - [12. Calibration & Diagnostic Utilities](#12-calibration--diagnostic-utilities)
4. [Folder Structure](#folder-structure)
5. [API Endpoints](#api-endpoints)
6. [Configuration Reference (`config.py` & `.env`)](#configuration-reference-configpy--env)
7. [Running with Real Cameras](#running-with-real-cameras)

---

## Quick Start

```bash
cd reid_system

# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify trained Re-ID checkpoint loading
python -c "from edge.reid_embedder import build_embedder; build_embedder(checkpoint_path='models/reid_resnet50_market1501.pth')"

# 3. Run unit tests
python -m pytest tests/ -v

# 4. Start the system
python run_demo.py
```

Then open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## System Architecture

```
 ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
 │  Camera 0  │   │  Camera 1  │   │  Camera 2  │   │  Camera 3  │
 │  (Zone A)  │   │  (Zone B)  │   │  (Zone C)  │   │  (Zone D)  │
 └─────┬──────┘   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
       │                │                │                │
       ▼                ▼                ▼                ▼
 ┌───────────────────────────────────────────────────────────────┐
 │                      EDGE WORKERS (Threaded)                  │
 │  • OpenCV Ingest (File / RTSP / Webcam)                       │
 │  • GPU Hardware Acceleration (CUDA)                           │
 │  • Pacing Engine (1.2x wall-clock sync + frame skipping)     │
 │  • YOLOv8n Detection + ByteTrack local ID tracking           │
 │  • MOG2 Background Subtraction Fallback                       │
 │  • ResNet-50 Market-1501 512-d Feature Extraction            │
 │  • Annotated MJPEG push to Frame Hub                         │
 └───────────────────────────────┬───────────────────────────────┘
                                 │ Ingest Payloads
                                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │                    CENTRAL RE-ID SERVICE                      │
 │  • FAISS Vector Similarity (IndexFlatIP, Cosine Matching)     │
 │  • Multi-Exemplar Gallery per (Shopper × Camera)              │
 │  • Global Identity Registry & Merge Resolution                │
 │  • Zone Journey Tracker & Dwell Time Analytics                │
 │  • SQLite WAL Persistence (data/reid_identities.db)           │
 │  • FastAPI REST Endpoints + WebSocket Broadcasting            │
 └───────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │                    OPERATIONS DASHBOARD                       │
 │  • 4-Feed Synchronized Live Camera Matrix (MJPEG)             │
 │  • Real-Time Shopper Ledger (Active/Inactive filters)         │
 │  • Customer Journey Modal & Similarity Scores                 │
 │  • Store-Wide KPI Cards & Zone Popularity Charts              │
 │  • One-Click CSV Export                                       │
 └───────────────────────────────────────────────────────────────┘
```

---

## Comprehensive Feature Guide

### 1. Edge Processing & Hardware Acceleration
- **Per-Camera Edge Isolation:** Each camera feed is managed by an independent `EdgeWorker` thread (`edge/edge_worker.py`), simulating real-world distributed edge appliances or microservices.
- **CUDA Acceleration:** Automatically detects and utilizes NVIDIA GPUs via PyTorch for both YOLOv8 object detection and ResNet-50 feature extraction. Drops tracking latency from ~500ms down to ~30ms per frame.
- **Graceful Ingest:** Supports recorded video files (`.avi`, `.mp4`), RTSP/HTTP live IP camera feeds, and local USB/integrated webcams. Video files automatically loop upon reaching EOF.

### 2. Multi-Stage Person Tracking & Fallbacks
- **Primary Tracker (YOLOv8 + ByteTrack):** Runs `ultralytics` YOLOv8n to identify persons (COCO class 0) and assigns persistent local track IDs (`track_id`) across frames using ByteTrack.
- **Secondary Fallback (MOG2 + Centroid):** If YOLOv8 is uninstalled or detects no candidates for 10 consecutive frames (e.g. synthetic test boxes), the system switches seamlessly to OpenCV MOG2 background subtraction with morphological dilation/erosion and centroid tracking.
- **Automatic Visual Cues:** Bounding boxes are color-coded in real time on the live stream:
  - **Cyan/Green:** Real YOLOv8 detection.
  - **Amber/Orange:** MOG2 background-subtraction fallback.

### 3. Deep Learning Feature Embedder (Market-1501)
- **Trained ResNet-50 Trunk:** Pretrained on ImageNet and fine-tuned on the Market-1501 person re-identification benchmark dataset (`models/reid_resnet50_market1501.pth`).
- **512-Dimensional Vector Space:** Person crops are resized to 256×128, transformed via ImageNet normalization, projected through a linear layer down to 512 dimensions, and processed through a BNNeck (`BatchNorm1d`) layer.
- **L2-Normalized Output:** All output embeddings have unit norm ($||v||_2 = 1.0$), allowing cosine similarity to be computed instantly via dot product ($v_1 \cdot v_2$).
- **Color Histogram Fallback:** If PyTorch is absent, `HistogramFallbackEmbedder` extracts a 512-bin 3D HSV color histogram so the pipeline continues operating.

### 4. Pacing Engine & 1.2x Video Speed Synchronization
- **1.2x Playback Speed Multiplier:** Video playback is calibrated to 1.2x speed (`CONFIG.video_speed = 1.2`), allowing faster surveillance review without distortion.
- **Wall-Clock Sync:** Uses high-resolution wall-clock pacing timers. If system load causes a thread delay greater than 150ms, the worker automatically skips decoding frames (`cap.grab()`) to stay synchronized with real time.
- **Detection Striding:** Uses an intelligent detection stride (`detect_stride = 2`). YOLO runs every 2nd frame, while every single frame (30 FPS at 1.2x) is annotated and pushed to the stream, cutting neural network load by 50% with zero visual lag.

### 5. Central Re-ID Engine & FAISS Vector Matching
- **High-Performance Vector Search:** Employs Facebook AI Similarity Search (`faiss-cpu`) using `IndexFlatIP`.
- **Identity Resolution:**
  - When an edge worker submits an embedding, the engine queries FAISS for the nearest neighbor.
  - If similarity $\ge$ `similarity_threshold` (default `0.78`): the track is merged into the existing `GlobalIdentity`.
  - If similarity $<$ `similarity_threshold`: a new `GlobalIdentity` (`shopper_N`) is registered.
- **Cross-Camera Match Detection:** When an existing identity appears on a different camera than previously recorded, the engine flags it as a cross-camera transition, tracks the zone transfer, and emits WebSocket notifications.

### 6. Multi-Exemplar Gallery & Identity Lifecycle
- **Multiple Exemplars per Camera:** As a shopper turns or walks through varying lighting conditions, the engine stores up to `max_exemplars_per_camera = 8` vectors per camera view. This prevents appearance drift and drastically improves cross-angle recall.
- **Active vs. Inactive TTL:** Shoppers not detected for longer than `inactive_identity_ttl_seconds` (default 300 seconds / 5 minutes) transition from **Active** to **Inactive**, keeping active visitor counts accurate.

### 7. Retail & Facility Analytics Engine
- **Dwell Time Tracking:** Computes total store dwell time (from first detection to last detection).
- **Zone Progression Paths:** Records sequential zone visits (e.g., `Zone A → Zone B → Zone C`) with individual entry timestamps, exit timestamps, and dwell durations.
- **Store-Wide Intelligence:** Generates live metrics for:
  - Total unique shoppers seen.
  - Currently active shoppers in the building.
  - Average dwell time across all visitors.
  - Zone popularity breakdown (total visits + average dwell time per zone).

### 8. Real-Time Frame Hub & MJPEG Streaming
- **Zero-Latency In-Memory Buffer:** `central/frame_hub.py` manages thread-safe frame queues for each camera feed.
- **Native Browser Streaming:** Feeds are exposed as multipart MJPEG HTTP streams at `/stream/{camera_id}`. Standard HTML `<img>` elements display live video feeds with bounding boxes and track labels without requiring client-side decoding libraries.

### 9. Operations Dashboard
- **Dark Ops-Room Interface:** Built with high-contrast, responsive CSS and typography (`Inter` & `JetBrains Mono`).
- **Live Camera Matrix:** Synchronized 2×2 grid displaying all 4 camera streams simultaneously with real-time status badges.
- **Identity Ledger:** Displays all tracked Global Shoppers with active status badges, visit counts, and dwell times. Supports filtering by *All*, *Active*, or *Inactive*.
- **Customer Journey Inspector:** Clicking any shopper card opens a modal displaying their complete journey timeline, timestamped zone visits, and cosine similarity matching confidence scores.
- **Live WebSocket Feed:** Subscribes to `/ws` for sub-second updates whenever new identities are registered or cross-camera merges occur.
- **One-Click CSV Export:** Downloads a complete CSV audit log of all shoppers, dwell durations, and zone visit histories.

### 10. SQLite Persistence & Crash Recovery
- **WAL-Mode SQLite Database:** Identity histories, exemplars, and zone visits are persisted to `data/reid_identities.db`.
- **Automatic Startup Recovery:** On server reboot, previous identities and their multi-exemplar feature vectors are restored from SQLite back into the FAISS index and in-memory registry.

### 11. API Key Authentication & Security
- **Optional Security Layer:** Protects API endpoints using Bearer token authentication (`REID_API_KEY`).
- **Development Mode:** Leave `REID_API_KEY=""` for unauthenticated local development; set a secret string in `.env` to enforce security for production deployments.

### 12. Calibration & Diagnostic Utilities
- **`scripts/calibrate_threshold.py`:** Evaluates cosine similarity distributions across known same-person vs. different-person crops and computes the optimal mathematical threshold for `config.py`.
- **`scripts/generate_sample_videos.py`:** Generates synthetic multi-camera test videos with walking colored targets when physical video feeds are unavailable.

---

## Folder Structure

```
reid_system/
├── run_demo.py                     # Master entry point (orchestrates API & workers)
├── config.py                       # Configuration dataclasses & environment defaults
├── requirements.txt                # Python package dependencies
│
├── models/
│   └── reid_resnet50_market1501.pth # Trained ResNet-50 Market-1501 weights
│
├── edge/
│   ├── detector_tracker.py         # YOLOv8 + ByteTrack / MOG2 centroid fallback
│   ├── reid_embedder.py            # ResNet-50 feature embedder + color histogram fallback
│   ├── edge_worker.py              # Camera loop, 1.2x pacing, annotation & ingest
│   └── transport.py                # HTTP & in-process payload transport
│
├── central/
│   ├── schemas.py                  # Pydantic data schemas (TrackPayload, GlobalIdentity)
│   ├── reid_engine.py              # FAISS index matching & identity registry
│   ├── persistence.py              # SQLite storage & recovery
│   ├── analytics.py                # Dwell time, zone paths, and store metrics
│   ├── frame_hub.py                # MJPEG frame buffer & stream generator
│   └── api.py                      # FastAPI REST routes & WebSocket server
│
├── dashboard/
│   ├── index.html                  # Dashboard HTML structure
│   ├── style.css                   # Ops-room styling & layout
│   └── app.js                      # WebSocket client, ledger rendering, modal inspector
│
├── scripts/
│   ├── calibrate_threshold.py      # Cosine similarity threshold calibration tool
│   └── generate_sample_videos.py   # Synthetic test video generator
│
└── tests/
    └── test_reid_engine.py         # PyTest test suite (FAISS, matching, TTL, paths)
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the web operations dashboard |
| `/health` | `GET` | Health check returning status and current shopper count |
| `/ingest` | `POST` | Ingests a `TrackPayload` from an edge camera worker |
| `/shoppers` | `GET` | Lists all Global Shopper summaries (filter by `?active=true`) |
| `/shoppers/{id}` | `GET` | Retrieves full visit history and zone paths for one shopper |
| `/analytics/store` | `GET` | Returns store-wide dwell times and zone popularity statistics |
| `/cameras` | `GET` | Returns the active camera and zone configuration |
| `/stream/{camera_id}` | `GET` | Live multipart MJPEG video stream with bounding box overlays |
| `/ws` | `WebSocket`| Real-time push updates for new identities and cross-camera merges |
| `/docs` | `GET` | Interactive Swagger UI API documentation |

---

## Configuration Reference (`config.py` & `.env`)

All parameters in `config.py` can be overridden via environment variables or a `.env` file:

| Parameter | Environment Variable | Default | Description |
|---|---|---|---|
| `video_speed` | `VIDEO_SPEED` | `1.2` | Video playback speed multiplier (e.g. 1.2 = 1.2x speed) |
| `detect_stride` | `DETECT_STRIDE` | `2` | Run YOLO detection every N frames to optimize throughput |
| `similarity_threshold` | `REID_THRESHOLD` | `0.78` | Cosine similarity cutoff for merging identities |
| `max_exemplars_per_camera` | `MAX_EXEMPLARS` | `8` | Maximum feature vectors saved per shopper per camera view |
| `inactive_identity_ttl_seconds` | `IDENTITY_TTL` | `300.0` | Seconds of inactivity before an identity is marked inactive |
| `send_interval_seconds` | `SEND_INTERVAL` | `0.5` | Minimum seconds between feature vector transmissions per track |
| `host` | `REID_HOST` | `0.0.0.0` | Web server listening address |
| `port` | `REID_PORT` | `8000` | Web server listening port |
| `api_key` | `REID_API_KEY` | `""` | Optional secret key for Bearer authentication |
| `db_path` | `REID_DB_PATH` | `data/reid_identities.db` | Path to SQLite database file |

---

## Running with Real Cameras

To switch from recorded video files to live webcams or RTSP network security cameras, update `config.py` or set environment variables:

```python
cameras = [
    # USB or Integrated Webcam
    CameraConfig(camera_id="cam_0", name="Front Door", zone="Entrance", source="0"),

    # RTSP IP Security Camera
    CameraConfig(camera_id="cam_1", name="Aisle 1", zone="Electronics", source="rtsp://admin:pass@192.168.1.50:554/h264Preview_01_main"),

    # Network Video Stream
    CameraConfig(camera_id="cam_2", name="Register", zone="Checkout", source="http://192.168.1.51:8080/video"),
]
```

The `source` string accepts anything compatible with OpenCV's `cv2.VideoCapture`.
