"""
scripts/generate_sample_videos.py — Create 4 synthetic demo camera feeds.

Each video is 30 seconds at 15 fps (450 frames).
"People" are coloured rectangles that move with simple random walk across the frame.
Colours are intentionally distinct so the random-projection embedder (if used
as fallback) produces separable histogram features per track, giving visible
tracking behaviour even without real video.

Run from reid_system/:
    python scripts/generate_sample_videos.py
"""
from __future__ import annotations
import os
import random
import math
import sys

try:
    import cv2
    import numpy as np
except ImportError:
    print("OpenCV and numpy are required. Run: pip install opencv-python numpy")
    sys.exit(1)

# Output directory (relative to reid_system/)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAMERAS = [
    ("cam_entrance",   "ENTRANCE",   (0, 180, 200)),   # teal bg tint
    ("cam_electronics","ELECTRONICS",(30, 20, 80)),     # deep purple tint
    ("cam_grocery",    "GROCERY",    (10, 60, 10)),     # dark green tint
    ("cam_checkout",   "CHECKOUT",   (80, 40, 10)),     # dark amber tint
]

W, H   = 640, 480
FPS    = 15
FRAMES = 450   # 30 s

PERSON_COLOURS = [
    (240, 100, 80),   # blue-ish (BGR)
    (60, 200, 80),    # green-ish
    (200, 80, 240),   # purple-ish
    (80, 200, 200),   # yellow-ish
    (200, 150, 60),   # cyan-ish
]


class Person:
    """A coloured rectangle that wanders around the frame."""

    def __init__(self, person_id: int, colour, frame_w: int, frame_h: int):
        self.person_id = person_id
        self.colour = colour
        self.pw = random.randint(40, 70)   # person width
        self.ph = random.randint(80, 130)  # person height
        # Start at random position
        self.x = float(random.randint(0, frame_w - self.pw))
        self.y = float(random.randint(0, frame_h - self.ph))
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-1.0, 1.0)
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.visible_start = random.randint(0, FRAMES // 3)
        self.visible_end   = random.randint(2 * FRAMES // 3, FRAMES)

    def step(self):
        # Random walk with slight mean-revert to keep in frame
        self.vx += random.gauss(0, 0.3)
        self.vy += random.gauss(0, 0.2)
        # Clamp speed
        self.vx = max(-3.0, min(3.0, self.vx))
        self.vy = max(-2.0, min(2.0, self.vy))
        self.x += self.vx
        self.y += self.vy
        # Bounce off walls
        if self.x < 0:           self.x, self.vx = 0.0, abs(self.vx)
        if self.x + self.pw > self.frame_w: self.x, self.vx = self.frame_w - self.pw, -abs(self.vx)
        if self.y < 0:           self.y, self.vy = 0.0, abs(self.vy)
        if self.y + self.ph > self.frame_h: self.y, self.vy = self.frame_h - self.ph, -abs(self.vy)

    def draw(self, frame: np.ndarray, frame_idx: int):
        if not (self.visible_start <= frame_idx < self.visible_end):
            return
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = x1 + self.pw, y1 + self.ph
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colour, -1)
        # Darker inner rectangle — gives the histogram-based fallback something
        inner_col = tuple(max(0, c - 60) for c in self.colour)
        cv2.rectangle(frame, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), inner_col, -1)


def make_video(cam_id: str, zone_label: str, bg_tint: tuple):
    out_path = os.path.join(OUTPUT_DIR, f"{cam_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    # Background: dark-grey with subtle colour tint
    base_bg = np.array([20, 20, 20], dtype=np.uint8)
    bg_colour = np.clip(base_bg + np.array(bg_tint, dtype=np.int32), 0, 255).astype(np.uint8)

    n_people = random.randint(2, len(PERSON_COLOURS))
    people = [
        Person(i, PERSON_COLOURS[i], W, H)
        for i in range(n_people)
    ]

    for fi in range(FRAMES):
        frame = np.full((H, W, 3), bg_colour, dtype=np.uint8)

        # Add subtle grid lines for depth cue
        for gx in range(0, W, 80):
            cv2.line(frame, (gx, 0), (gx, H), tuple(int(c) + 8 for c in bg_colour), 1)
        for gy in range(0, H, 80):
            cv2.line(frame, (0, gy), (W, gy), tuple(int(c) + 8 for c in bg_colour), 1)

        for p in people:
            p.step()
            p.draw(frame, fi)

        # Zone label overlay
        cv2.putText(frame, zone_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 190), 2, cv2.LINE_AA)
        # Frame counter (tiny, bottom-right)
        cv2.putText(frame, f"{fi+1:03d}/{FRAMES}", (W - 80, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 100, 100), 1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(f"  [OK] {out_path}  ({FRAMES} frames @ {FPS} fps, {n_people} persons)")


if __name__ == "__main__":
    print(f"Generating {len(CAMERAS)} synthetic demo videos -> {OUTPUT_DIR}/\n")
    random.seed(99)
    np.random.seed(99)
    for cam_id, label, tint in CAMERAS:
        make_video(cam_id, label, tint)
    print("\nDone. Run python run_demo.py to start the system.")
