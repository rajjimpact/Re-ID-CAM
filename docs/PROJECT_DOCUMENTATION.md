# Distributed Multi-Camera Video Analytics & Cross-Camera Person Re-Identification System

**Project Documentation — Overview, Architecture & Implementation Plan**

---

## 1. What We Are Building

A software-only AI system that watches multiple video camera feeds at once
and recognizes when the **same person** appears in more than one camera —
even though each camera, on its own, has no idea the person it just saw is
the same one another camera saw five minutes ago.

Concretely, the system:

1. **Detects** every person in every camera's video feed (YOLOv8).
2. **Tracks** each person locally within one camera's view, frame to frame
   (ByteTrack) — this gives a *local* track ID, e.g. "Track 7 on Camera 1."
3. **Describes** each tracked person's appearance as a numeric fingerprint
   (a 512-number vector) using a ResNet50 deep learning model, fine-tuned
   specifically for this task on the Market-1501 person re-identification
   dataset.
4. **Matches** that fingerprint against every other fingerprint the system
   has seen, across all cameras, using cosine similarity search (FAISS) —
   this is what turns "Track 7 on Camera 1" and "Track 3 on Camera 4" into
   one **Global Shopper ID**, if they're actually the same person.
5. **Reports** on it: which zones a person visited, in what order, how
   long they stayed, and which zones are busiest overall — shown live on a
   web dashboard.

The retail-store framing (Entrance → Electronics → Grocery → Checkout) is
the example scenario used throughout the codebase and this document, but
nothing in the architecture is retail-specific — see [§9](#9-beyond-retail--other-use-cases).

---

## 2. Why This Project Exists (Problem Statement)

Standard CCTV / video analytics systems process each camera **in
isolation**. Every camera's tracker restarts its ID counter independently,
so the instant a person walks out of one camera's field of view and into
another's, the system loses the thread — it has no built-in concept that
"Track 7 here" and "Track 3 over there" are the same human being.

That makes basic, valuable questions impossible to answer automatically:

- *How long does the average customer actually spend in the store, start
  to finish?*
- *What path do most customers take through the space?*
- *Did the person flagged at the entrance also show up at the register?*

This project closes that gap with a **centralized Re-ID engine** that
resolves per-camera local identities into one consistent global identity,
without needing a face database, badge, or any cooperative identification
from the person being tracked — appearance alone (clothing, build, gait
silhouette) is enough.

---

## 3. Target Audience

This is a working prototype / reference architecture, so "who it's for"
splits into who would **use** the finished product and who this
**codebase and documentation** is written for.

| Audience | What they get from this |
|---|---|
| **Retail store & mall operators** | Store-wide customer flow analytics: dwell time, zone popularity, path patterns — input for merchandising and layout decisions. |
| **Loss-prevention / security teams** | The ability to follow one person of interest across a whole facility's camera network without manually scrubbing through footage from each camera separately. |
| **Facility & operations managers** (warehouses, airports, campuses, transit hubs) | The same movement/occupancy analytics, applied to non-retail spaces — see [§9](#9-beyond-retail--other-use-cases). |
| **You (the builder)** | A complete, working reference implementation of a real multi-stage computer vision pipeline — detection → tracking → embedding → vector search → analytics → live dashboard — suitable as a capstone/major project, portfolio piece, or internal proof-of-concept. |
| **Future contributors / evaluators reading this repo** | This document specifically: a single place that explains *what* the system does, *why* it's built the way it is, and *how* every piece fits together, without needing to reverse-engineer it from source code alone. |

**A note on responsible use:** this system processes video of real people
and produces cross-camera movement histories, which is meaningfully more
sensitive than single-camera CCTV. Before deploying anything built from
this codebase against real people, see [§11](#11-privacy-ethics--responsible-use).

---

## 4. System Architecture

```
 ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
 │  Camera 1  │   │  Camera 2  │   │  Camera 3  │   │  Camera 4  │
 │ (Entrance) │   │(Electronics)│   │ (Grocery)  │   │(Checkout)  │
 └─────┬──────┘   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
       │                │                │                │
       ▼                ▼                ▼                ▼
 ┌───────────────────────────────────────────────────────────────┐
 │                      EDGE  (one worker per camera)             │
 │  OpenCV ingest → YOLOv8 detect → ByteTrack local-track →       │
 │  crop → ResNet50 embed (512-d)  →  small JSON payload          │
 └───────────────────────────────┬────────────────────────────────┘
                                  │  {camera_id, local_track_id,
                                  │   timestamp, bbox, embedding[512]}
                                  ▼
 ┌───────────────────────────────────────────────────────────────┐
 │                          CENTRAL  (FastAPI)                    │
 │  /ingest → FAISS cosine similarity search →                    │
 │  match existing Global Shopper ID  OR  create a new one →      │
 │  update zone-visit history & dwell time  → broadcast /ws       │
 └───────────────────────────────┬────────────────────────────────┘
                                  │  live JSON updates (WebSocket)
                                  │  + REST for initial state
                                  ▼
 ┌───────────────────────────────────────────────────────────────┐
 │                          FRONTEND (dashboard)                  │
 │  Camera wall (live annotated feeds) · Shopper ledger (global   │
 │  IDs + path + dwell) · Store stats (unique visitors, zone       │
 │  popularity)                                                    │
 └───────────────────────────────────────────────────────────────┘
```

**The key architectural decision:** only a small structured payload
(camera ID, local track ID, timestamp, bounding box, 512-number vector —
a few KB) crosses the network from edge to center, never the raw video
frame. This is what lets the architecture scale to many cameras without
the central server becoming a video-processing bottleneck.

---

## 5. Technology Stack

| Layer | Technology | Role |
|---|---|---|
| Video ingestion | OpenCV | Reads camera/video-file frames |
| Detection | YOLOv8 (`ultralytics`) | Locates people in each frame |
| Local tracking | ByteTrack (built into `ultralytics`) | Keeps a stable ID per person within one camera |
| Re-ID embedding | PyTorch, ResNet50, **fine-tuned on Market-1501** | Turns a person crop into a 512-d appearance fingerprint |
| Similarity search | FAISS (`IndexFlatIP`) | Cosine-similarity matching of embeddings at scale |
| Backend API | FastAPI + Uvicorn | REST endpoints + WebSocket for live updates |
| Backend transport | HTTP (`requests`) or in-process call | Edge → Central payload delivery |
| Frontend | HTML5 / CSS3 / vanilla JavaScript | Live dashboard: camera wall, shopper ledger, store stats |
| Testing | Python `unittest` | Correctness of the matching/identity-persistence logic |

Every ML-heavy dependency (`ultralytics`, `torch`, `faiss`, `fastapi`) has
an automatic dependency-light fallback baked into the code, so the system
runs even on a machine without a GPU or those packages installed — see
`README.md` in the codebase for details. This documentation assumes
you're running the **real** path with your trained model.

---

## 6. Folder Structure

```
reid_system/
├── README.md                      # setup & quick-start (existing)
├── PROJECT_DOCUMENTATION.md        # ← this file
├── requirements.txt
├── config.py                       # all tunable thresholds + camera list
│
├── models/
│   └── reid_resnet50_market1501.pth   # ← your trained checkpoint goes here
│
├── edge/
│   ├── __init__.py
│   ├── detector_tracker.py         # YOLOv8 + ByteTrack (+ fallback tracker)
│   ├── reid_embedder.py            # ResNet50 embedder — UPDATED to load your checkpoint
│   ├── edge_worker.py              # orchestrates one camera end to end
│   └── transport.py                # HTTP sink for multi-machine deployments
│
├── central/
│   ├── __init__.py
│   ├── schemas.py                  # shared payload/identity data structures
│   ├── reid_engine.py              # Global Shopper ID registry + FAISS matching
│   ├── analytics.py                # dwell time, zone popularity, paths
│   ├── frame_hub.py                # live annotated-frame buffer for the dashboard
│   └── api.py                      # FastAPI app: /ingest, /shoppers, /analytics, /ws, /stream
│
├── dashboard/
│   ├── index.html                  # dashboard shell
│   ├── style.css                   # dark "ops room" visual theme
│   └── app.js                      # camera wall + live ledger + stats, via REST + WebSocket
│
├── sample_videos/                  # synthetic demo footage (4 camera feeds)
├── scripts/
│   └── generate_sample_videos.py
│
├── tests/
│   └── test_reid_engine.py
│
└── run_demo.py                     # single entry point: runs everything, one process
```

**What's new for this phase:** the `models/` folder and the updated
`edge/reid_embedder.py` — see [§10](#10-connecting-your-trained-model).
Everything else is the architecture already built; this phase is about
wiring your trained weights into it, not building it from scratch.

---

## 7. Backend — Logic & Responsibilities

The backend has two distinct halves that run as separate concerns even
when they're in the same process (as in `run_demo.py`):

### 7.1 Edge logic (`edge/`)

Runs once per camera. Each `EdgeWorker`:

1. Opens its video source with OpenCV (`cv2.VideoCapture`) — a file path
   today, an `rtsp://` URL in production, transparently (same API call).
2. Passes every frame through `detector_tracker.py`, which runs YOLOv8
   detection and ByteTrack tracking together (`model.track(...)`), giving
   a list of `(track_id, bbox, confidence)` per frame.
3. For each tracked person, throttled to twice a second
   (`send_interval_seconds` in `config.py`) so the network isn't hammered
   every frame:
   - crops the bounding box out of the frame,
   - runs it through `reid_embedder.py`'s `ResNet50Embedder.embed()` to
     get a 512-d, L2-normalized appearance vector,
   - packages `{camera_id, local_track_id, timestamp, bbox, embedding}`
     into a `TrackPayload` (`central/schemas.py`),
   - sends it to the central engine (in-process function call in the
     single-machine demo, or an HTTP POST via `edge/transport.py` for a
     real distributed deployment).

### 7.2 Central logic (`central/`)

Runs once, regardless of how many cameras there are. On every incoming
`TrackPayload` (`POST /ingest` in `central/api.py`):

1. `reid_engine.py`'s `GlobalIdentityRegistry.ingest()` first expires any
   identity that's been silent past `inactive_identity_ttl_seconds`
   (5 minutes by default) — closing its currently-open zone visit without
   deleting its history, which is what gives the system **identity
   persistence** through brief occlusions/blind spots.
2. It searches the FAISS index for the closest existing embedding by
   cosine similarity.
   - **Above threshold** (`similarity_threshold`, default 0.78): the
     payload is folded into that existing `GlobalIdentity` — its stored
     embedding is updated as a running average, and if the camera's zone
     differs from the identity's currently-open zone visit, that visit is
     closed and a new one opened (this is what builds the
     `Entrance → Electronics → Checkout` path).
   - **Below threshold**: a brand-new `GlobalIdentity` is registered with
     the next Global Shopper ID.
3. `analytics.py` computes derived numbers on demand (dwell time per
   shopper, per-zone visit counts and average dwell, currently-active
   count) — nothing here is precomputed/cached, it's recalculated from
   the registry each time `/analytics/store` or `/shoppers` is called,
   which is simple and fast enough at prototype scale.
4. The resolved identity update is broadcast to every connected dashboard
   over the `/ws` WebSocket, and the annotated camera frame (drawn by
   `frame_hub.py`) is available at `/stream/{camera_id}` as an MJPEG feed
   any `<img>` tag can consume directly.

### 7.3 Backend API surface

| Endpoint | Method | Purpose |
|---|---|---|
| `/ingest` | POST | Edge workers submit a `TrackPayload` here |
| `/shoppers` | GET | List every Global Shopper ID with a summary (path, dwell, active state) |
| `/shoppers/{id}` | GET | Full visit-by-visit history for one shopper |
| `/analytics/store` | GET | Store-wide numbers: unique shoppers, currently active, zone popularity |
| `/cameras` | GET | Configured camera/zone list, for the dashboard to build its camera wall |
| `/stream/{camera_id}` | GET | Live MJPEG preview of one camera, annotated with detection boxes |
| `/ws` | WebSocket | Live push of every identity resolution event, as it happens |
| `/` | GET | Serves the dashboard itself |

---

## 8. Frontend — Logic & Responsibilities

The dashboard (`dashboard/`) is a single page with three coordinated
regions, all driven from `app.js` with no build step or framework:

| Region | Data source | Behavior |
|---|---|---|
| **Camera wall** | `GET /cameras` once, then `<img src="/stream/{id}">` per tile | Each tile is a live MJPEG stream; the backend does all the annotation (boxes + local track ID), the browser just displays it. |
| **Shopper ledger** | `GET /shoppers` on load, then `/ws` for live updates | A `Map<global_id, summary>` in memory; every `identity_update` message from the WebSocket updates one entry and re-renders the list, newest-active-first. A brief highlight animation marks identities that just got a fresh cross-camera match. |
| **Store stats footer** | `GET /analytics/store`, polled every 4s | Unique-shopper count, currently-active count, average dwell time, and a zone-popularity bar chart built from plain `<div>` widths — no charting library needed at this scale. |

**Connection resilience:** the WebSocket client auto-reconnects on drop
(`ws.onclose` schedules a retry), and the connection-status indicator in
the header reflects `live` / `reconnecting…` honestly rather than silently
going stale.

**Design intent:** a night-operations monitoring-room aesthetic (near-black
background, cyan for live/active state, amber for a just-happened match
event, monospace for all telemetry values like IDs/timestamps/durations)
rather than a generic SaaS dashboard look — because this is genuinely
operational data a technician reads, not a marketing surface.

---

## 9. Beyond Retail — Other Use Cases

Nothing in `config.py`'s `CameraConfig` list or `reid_engine.py`'s matching
logic assumes "retail." Swapping the zone names and camera sources
repurposes the same pipeline for:

- **Airports** — passenger flow from check-in → security → gate, dwell
  time at gates, congestion hotspots.
- **Warehouses** — worker/forklift movement patterns, dwell time at
  specific stations, safety-zone occupancy.
- **Campuses / transit hubs** — footfall patterns across buildings or
  platforms, peak-time analysis.

---

## 10. Connecting Your Trained Model

You trained `edge/reid_embedder.py`'s ResNet50 backbone + 512-d projection
on Market-1501 in Colab (see the companion notebook) and downloaded
`reid_resnet50_market1501.pth`. Here's exactly how it plugs in.

### 10.1 Place the checkpoint

```
reid_system/models/reid_resnet50_market1501.pth
```

### 10.2 Add a config field

In `config.py`, add one field to `SystemConfig`:

```python
reid_checkpoint_path: str = "models/reid_resnet50_market1501.pth"
```

### 10.3 Update `ResNet50Embedder` to load it

Replace the fixed random projection in `edge/reid_embedder.py` with a
checkpoint loader:

```python
class ResNet50Embedder(BaseEmbedder):
    def __init__(self, embedding_dim: int = 512, device: str = None,
                 checkpoint_path: str = None):
        self.embedding_dim = embedding_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        self.backbone = backbone.to(self.device).eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.backbone.load_state_dict(ckpt["backbone_state_dict"])
            proj = ckpt["projection"]                      # already (2048, 512)
            print(f"Loaded trained Re-ID weights from {checkpoint_path} "
                  f"(eval results at export time: {ckpt.get('eval_results')})")
        else:
            # fallback: fixed random projection (untrained prototype behavior)
            rng = np.random.default_rng(seed=42)
            proj = rng.normal(size=(2048, embedding_dim)).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)

        self.projection = torch.from_numpy(proj).to(self.device)
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    # embed() is unchanged
```

### 10.4 Update the factory function

```python
def build_embedder(embedding_dim: int = 512, checkpoint_path: str = None) -> BaseEmbedder:
    if _HAS_TORCH:
        try:
            return ResNet50Embedder(embedding_dim=embedding_dim, checkpoint_path=checkpoint_path)
        except Exception:
            pass
    return HistogramFallbackEmbedder(embedding_dim=embedding_dim)
```

### 10.5 Pass the path through from the edge worker

In `edge/edge_worker.py`, the line that builds the embedder becomes:

```python
self.embedder = build_embedder(
    embedding_dim=CONFIG.embedding_dim,
    checkpoint_path=CONFIG.reid_checkpoint_path,
)
```

### 10.6 Re-calibrate the similarity threshold

`config.py`'s `similarity_threshold: 0.78` was a reasonable placeholder
for an *untrained random projection*. Your trained model's score
distribution for same-person vs. different-person pairs will look
different — before trusting it in production, run a handful of known
same-person and different-person crops through the new embedder, look at
the cosine similarities you actually get, and set the threshold between
those two clusters.

---

## 11. Privacy, Ethics & Responsible Use

Cross-camera Re-ID is meaningfully more sensitive than single-camera CCTV
because it produces a **movement history** of a real person across an
entire facility, not just isolated clips. Before deploying anything built
from this codebase against real people:

- Check local/regional law on video surveillance and biometric-adjacent
  processing (appearance-embedding-based Re-ID is not facial recognition,
  but some privacy laws treat any individual-tracking system similarly —
  this varies by jurisdiction).
- Post clear signage disclosing camera-based analytics, consistent with
  what applies in your jurisdiction/venue.
- Set a real data-retention policy — the prototype's default 5-minute
  identity TTL only affects *live* re-identification; decide separately
  how long resolved histories should be stored, and delete or anonymize
  them on a set schedule.
- Restrict access to the dashboard and API to authorized personnel only —
  neither currently ships with authentication (see roadmap below).

---

## 12. Implementation Roadmap

| Phase | Status | Description |
|---|---|---|
| 1. Core pipeline architecture | ✅ Done | Edge/central split, detection, tracking, matching, analytics, dashboard — all built and tested against synthetic video. |
| 2. Model training | ✅ Done | ResNet50 fine-tuned on Market-1501 in Colab; checkpoint downloaded. |
| 3. Model integration | ⬜ Next | Apply the [§10](#10-connecting-your-trained-model) patch; verify real embeddings improve match quality vs. the random-projection baseline. |
| 4. Threshold calibration | ⬜ | Re-tune `similarity_threshold` against the trained model's real score distribution. |
| 5. Real footage validation | ⬜ | Swap `sample_videos/` for actual camera footage of the target space; confirm detection/tracking quality on real people, real lighting. |
| 6. Frontend polish (optional) | ⬜ | Historical playback, per-shopper detail view, date-range filters on analytics, CSV export. |
| 7. Access control (optional, recommended before any real deployment) | ⬜ | Add authentication to the FastAPI app and dashboard. |
| 8. Deployment | ⬜ | Real RTSP cameras, persistent storage in place of in-memory registry, containerized deployment. |

---

## 13. Summary

| Question | Answer |
|---|---|
| **What is it?** | A prototype that re-identifies people across multiple camera views using detection + tracking + a fine-tuned appearance-embedding model + vector similarity search. |
| **Why build it?** | Standard CCTV can't answer "how did this person move through my whole space," only "what did one camera see." |
| **Who is it for?** | Retail/facility operators (analytics), security teams (cross-camera tracking), and as a reference implementation / capstone project for whoever built it. |
| **What's left to do?** | Wire the trained checkpoint into `edge/reid_embedder.py` ([§10](#10-connecting-your-trained-model)), re-calibrate the threshold, then validate against real footage. |
