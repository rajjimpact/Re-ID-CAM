"""
central/reid_engine.py — Global Identity Registry + multi-exemplar FAISS matching.

Phase 9 (multi-exemplar gallery):
  ─────────────────────────────────────────────────────────────────────────────
  Each GlobalIdentity now holds a *gallery* of real, unblended L2-normalised
  embedding vectors rather than a single running-average vector.  The gallery is
  organised as a two-level dict:

      gallery_per_cam[global_id][camera_id] = List[np.ndarray]

  Each camera bucket is capped at `max_exemplars_per_camera` (default 8).
  When the bucket is full, new vectors are silently dropped (oldest exemplars
  are the most reliable; we never blend).

  Matching algorithm
  ──────────────────
  query vector  →  cosine similarity against *every* gallery exemplar
  best score    =  max over all exemplars across all cameras
  → above threshold  : merge (append to bucket if room, update metadata)
  → below threshold  : register new GlobalIdentity

  FAISS index
  ───────────
  One row per exemplar (not one row per identity).  `_emb_ids` is a parallel
  list of global_id strings (duplicates are fine — best score wins the search).

  Persistence
  ───────────
  Gallery is serialised as `gallery_json` (JSON list-of-lists) and written to
  SQLite by central.persistence.  The old `embedding_sum` column is migrated
  transparently on first read.

Public API (unchanged from Phase 8):
  ingest(payload)  → (global_id, is_cross_camera, match_score)
  get_all()        → List[GlobalIdentity]
  get_by_id(id)    → Optional[GlobalIdentity]
  identity_count() → int
  reset()          → None   ← NEW: flush everything, restart counter from 1
  exemplar_stats() → List[dict]  ← NEW: per-identity gallery breakdown
"""
from __future__ import annotations
import json
import time
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from central.schemas import GlobalIdentity, TrackPayload

try:
    import faiss as _faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    print("[reid_engine] faiss not installed — using brute-force numpy cosine search.")


class GlobalIdentityRegistry:
    """
    Thread-safe registry that resolves per-camera local track IDs into
    persistent Global Shopper IDs using multi-exemplar appearance-embedding
    similarity.

    Phase 9: Gallery-based matching — never blends embeddings.
             Accepts optional IdentityStore for SQLite persistence.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        similarity_threshold: float = 0.78,
        inactive_ttl_seconds: float = 300.0,
        max_exemplars_per_camera: int = 8,
        store=None,          # central.persistence.IdentityStore or None
    ) -> None:
        self.embedding_dim            = embedding_dim
        self.similarity_threshold     = similarity_threshold
        self.inactive_ttl             = inactive_ttl_seconds
        self.max_exemplars_per_camera = max_exemplars_per_camera
        self._store                   = store

        self._lock       = threading.Lock()
        self._identities: Dict[str, GlobalIdentity] = {}
        self._next_id: int = 1

        # gallery_per_cam[global_id][camera_id] = list of np.ndarray (L2-normed)
        self._gallery_per_cam: Dict[str, Dict[str, List[np.ndarray]]] = {}

        # Flat arrays for FAISS / brute-force search
        self._emb_matrix: np.ndarray = np.empty((0, embedding_dim), dtype=np.float32)
        self._emb_ids: List[str]     = []   # parallel list of global_id (may repeat)

        # FAISS index (rebuilt lazily)
        self._index       = None
        self._index_dirty = True

        # Restore from SQLite if a store is provided
        if store is not None:
            self._restore_from_store(store)

    # ── Restore from DB ───────────────────────────────────────────────────────

    def _restore_from_store(self, store) -> None:
        loaded = store.load_all()
        for identity in loaded:
            self._identities[identity.global_id] = identity

            # Track next_id counter
            try:
                n = int(identity.global_id.split("-")[-1])
                if n >= self._next_id:
                    self._next_id = n + 1
            except (ValueError, IndexError):
                pass

            # Rebuild gallery from persisted gallery_json
            gid = identity.global_id
            self._gallery_per_cam[gid] = {}

            if identity.gallery_json:
                # Stored as list-of-[cam_id, vector_list] pairs OR flat list of vectors
                # (flat list = legacy embedding_sum migration — treat as cam "unknown")
                raw = identity.gallery_json
                if raw and isinstance(raw[0], list):
                    # Modern format: [[cam_id_str, [f0, f1, ...]], ...]
                    for entry in raw:
                        if isinstance(entry, list) and len(entry) == 2:
                            cam_id, vec = entry[0], entry[1]
                            emb = np.array(vec, dtype=np.float32)
                            norm = np.linalg.norm(emb)
                            if norm > 0:
                                emb /= norm
                            bucket = self._gallery_per_cam[gid].setdefault(cam_id, [])
                            bucket.append(emb)
                            self._emb_matrix = np.vstack(
                                [self._emb_matrix, emb[np.newaxis, :]]
                            )
                            self._emb_ids.append(gid)
                elif isinstance(raw, list) and raw:
                    # Legacy: flat float list (old embedding_sum) → single exemplar
                    emb = np.array(raw, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb /= norm
                    bucket = self._gallery_per_cam[gid].setdefault("legacy", [])
                    bucket.append(emb)
                    self._emb_matrix = np.vstack(
                        [self._emb_matrix, emb[np.newaxis, :]]
                    )
                    self._emb_ids.append(gid)

        self._index_dirty = True
        if loaded:
            total_ex = sum(
                sum(len(b) for b in cam_dict.values())
                for cam_dict in self._gallery_per_cam.values()
            )
            print(
                f"[reid_engine] Restored {len(loaded)} identities "
                f"({total_ex} total exemplars) from SQLite."
            )

    # ── FAISS index management ────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        if not _HAS_FAISS or self._emb_matrix.shape[0] == 0:
            self._index      = None
            self._index_dirty = False
            return
        idx = _faiss.IndexFlatIP(self.embedding_dim)
        idx.add(self._emb_matrix)
        self._index      = idx
        self._index_dirty = False

    def _cosine_search(self, query: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Returns (best_global_id, cosine_score) or (None, 0.0).
        Scores the query against *every* exemplar; best score wins.
        """
        if self._emb_matrix.shape[0] == 0:
            return None, 0.0

        if _HAS_FAISS:
            if self._index_dirty or self._index is None:
                self._rebuild_index()
            if self._index is None:
                return None, 0.0
            # k = min(exemplar count, 5) — take the highest-scoring row
            k = min(self._emb_matrix.shape[0], 5)
            scores, idxs = self._index.search(query[np.newaxis, :], k=k)
            best_score = float(scores[0][0])
            best_id    = self._emb_ids[int(idxs[0][0])]
        else:
            # Brute-force fallback
            sims = self._emb_matrix @ query          # (N,)
            best_idx   = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            best_id    = self._emb_ids[best_idx]

        return best_id, best_score

    # ── Gallery helpers ───────────────────────────────────────────────────────

    def _add_exemplar(self, global_id: str, camera_id: str, emb: np.ndarray) -> bool:
        """
        Append `emb` to the per-camera bucket for `global_id` if room remains.
        Returns True if the exemplar was accepted (updates FAISS matrix too).
        """
        cam_dict = self._gallery_per_cam.setdefault(global_id, {})
        bucket   = cam_dict.setdefault(camera_id, [])
        if len(bucket) >= self.max_exemplars_per_camera:
            return False   # bucket full — silently drop, never blend
        bucket.append(emb)
        self._emb_matrix = np.vstack([self._emb_matrix, emb[np.newaxis, :]])
        self._emb_ids.append(global_id)
        self._index_dirty = True
        return True

    def _gallery_to_json(self, global_id: str) -> list:
        """Serialise the gallery to a JSON-safe list of [cam_id, vec] pairs."""
        result = []
        for cam_id, bucket in self._gallery_per_cam.get(global_id, {}).items():
            for emb in bucket:
                result.append([cam_id, emb.tolist()])
        return result

    def _total_exemplars(self, global_id: str) -> int:
        cam_dict = self._gallery_per_cam.get(global_id, {})
        return sum(len(b) for b in cam_dict.values())

    # ── TTL expiry ────────────────────────────────────────────────────────────

    def _expire_stale(self, now: float) -> None:
        for identity in self._identities.values():
            if identity.active and (now - identity.last_seen) > self.inactive_ttl:
                identity.active = False
                identity.close_current_visit(at=identity.last_seen)
                if self._store:
                    self._store.save(identity)

    # ── Main ingest ───────────────────────────────────────────────────────────

    def ingest(self, payload: TrackPayload) -> Tuple[str, bool, float]:
        """
        Returns (global_id, is_cross_camera_match, match_score).
        match_score is the best cosine similarity from the gallery search
        (0.0 for brand-new identities).
        """
        emb = np.array(payload.embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm

        now = payload.timestamp

        with self._lock:
            self._expire_stale(now)

            best_id, score = self._cosine_search(emb)
            is_cross_camera = False

            if best_id and score >= self.similarity_threshold:
                # ── Merge into existing identity ──────────────────────────────
                identity = self._identities[best_id]

                # Cross-camera match flag
                if identity.last_camera_id and identity.last_camera_id != payload.camera_id:
                    is_cross_camera = True

                identity.last_seen      = now
                identity.last_camera_id = payload.camera_id
                identity.last_bbox      = payload.bbox
                identity.active         = True

                # Append exemplar to per-camera bucket (if room); never blend
                accepted = self._add_exemplar(best_id, payload.camera_id, emb)
                if accepted:
                    identity.gallery_json    = self._gallery_to_json(best_id)
                    identity.embedding_count = self._total_exemplars(best_id)
                    if self._store:
                        self._store.save(identity)

                # Zone-visit tracking
                identity.open_zone_visit(payload.zone, payload.camera_id, now)

                if self._store:
                    self._store.save(identity)

                return best_id, is_cross_camera, score

            else:
                # ── New identity ──────────────────────────────────────────────
                global_id = f"GSI-{self._next_id:04d}"
                self._next_id += 1

                identity = GlobalIdentity(
                    global_id=global_id,
                    first_seen=now,
                    last_seen=now,
                    last_camera_id=payload.camera_id,
                    last_bbox=payload.bbox,
                    gallery_json=[],
                    embedding_count=0,
                )
                identity.open_zone_visit(payload.zone, payload.camera_id, now)

                self._identities[global_id]    = identity
                self._gallery_per_cam[global_id] = {}

                # Register first exemplar
                self._add_exemplar(global_id, payload.camera_id, emb)
                identity.gallery_json    = self._gallery_to_json(global_id)
                identity.embedding_count = 1

                if self._store:
                    self._store.save(identity)

                return global_id, False, 0.0

    # ── Registry management ───────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Flush everything in-memory and truncate the SQLite tables.
        Identity counter restarts from 1.
        Thread-safe.
        """
        with self._lock:
            self._identities.clear()
            self._gallery_per_cam.clear()
            self._emb_matrix  = np.empty((0, self.embedding_dim), dtype=np.float32)
            self._emb_ids      = []
            self._index        = None
            self._index_dirty  = True
            self._next_id      = 1
            if self._store:
                self._store.truncate()
        print("[reid_engine] Registry reset — all identities cleared.")

    def exemplar_stats(self) -> List[dict]:
        """
        Return per-identity gallery breakdown for the Re-ID Register panel.
        Thread-safe read.
        """
        with self._lock:
            result = []
            for gid, identity in self._identities.items():
                cam_breakdown = {
                    cam_id: len(bucket)
                    for cam_id, bucket in self._gallery_per_cam.get(gid, {}).items()
                }
                total = sum(cam_breakdown.values())
                result.append({
                    "global_id":     gid,
                    "first_seen":    identity.first_seen,
                    "last_seen":     identity.last_seen,
                    "active":        identity.active,
                    "total_exemplars": total,
                    "per_camera":    cam_breakdown,
                })
            return sorted(result, key=lambda x: x["global_id"])

    # ── Read helpers ──────────────────────────────────────────────────────────

    def get_all(self) -> List[GlobalIdentity]:
        with self._lock:
            return list(self._identities.values())

    def get_by_id(self, global_id: str) -> Optional[GlobalIdentity]:
        with self._lock:
            return self._identities.get(global_id)

    def identity_count(self) -> int:
        with self._lock:
            return len(self._identities)
