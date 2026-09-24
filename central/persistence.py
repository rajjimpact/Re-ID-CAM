"""
central/persistence.py — SQLite-backed persistence for the identity registry.

Phase 8a: In-memory GlobalIdentityRegistry backed by SQLite.
Phase 9 : Non-destructive migration from single embedding_sum vector to
           multi-exemplar gallery_json (list of [cam_id, vector] pairs).

Schema (current)
──────────────────
  identities(global_id TEXT PK, first_seen REAL, last_seen REAL,
             last_camera_id TEXT,
             gallery_json TEXT,          ← Phase 9 (replaces embedding_sum)
             embedding_sum TEXT,         ← kept for backward compat (ignored on write)
             embedding_count INT,
             active INT)

  zone_visits(id INTEGER PK, global_id TEXT, zone TEXT, camera_id TEXT,
              enter_time REAL, exit_time REAL)

Migration
──────────
  If gallery_json column doesn't exist, it is added automatically.  On first
  read, rows that have a non-empty embedding_sum and empty gallery_json are
  migrated: the embedding_sum value is treated as a single exemplar vector from
  camera "legacy".

All writes are non-blocking (background thread queue) and never slow the ingest
path.  truncate() is synchronous (called during registry reset).
"""
from __future__ import annotations
import json
import os
import queue
import sqlite3
import threading
from typing import List

from central.schemas import GlobalIdentity, ZoneVisit


class IdentityStore:
    """
    Thread-safe SQLite persistence layer.

    Usage
    ─────
    store = IdentityStore("data/reid_identities.db")
    store.save(identity)           # non-blocking async write
    identities = store.load_all()  # blocking, called once at startup
    store.truncate()               # blocking, called on registry reset
    store.close()
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._write_queue: queue.Queue = queue.Queue()
        self._running = True

        # ── Create / migrate schema ───────────────────────────────────────────
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS identities (
                global_id       TEXT PRIMARY KEY,
                first_seen      REAL NOT NULL,
                last_seen       REAL NOT NULL,
                last_camera_id  TEXT NOT NULL DEFAULT '',
                embedding_sum   TEXT NOT NULL DEFAULT '[]',
                embedding_count INTEGER NOT NULL DEFAULT 0,
                active          INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS zone_visits (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                global_id   TEXT NOT NULL,
                zone        TEXT NOT NULL,
                camera_id   TEXT NOT NULL DEFAULT '',
                enter_time  REAL NOT NULL,
                exit_time   REAL,
                FOREIGN KEY (global_id) REFERENCES identities(global_id)
            );
            CREATE INDEX IF NOT EXISTS idx_zone_visits_gid ON zone_visits(global_id);
        """)

        # Phase 9 migration: add gallery_json column if absent
        cols = [row[1] for row in conn.execute("PRAGMA table_info(identities)").fetchall()]
        if "gallery_json" not in cols:
            conn.execute(
                "ALTER TABLE identities ADD COLUMN gallery_json TEXT NOT NULL DEFAULT '[]'"
            )
            conn.commit()
            print("[persistence] Migrated DB: added gallery_json column.")

        conn.commit()
        conn.close()
        print(f"[persistence] SQLite database at {os.path.abspath(db_path)}")

        # ── Background writer thread ──────────────────────────────────────────
        self._writer = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="persistence-writer",
        )
        self._writer.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, identity: GlobalIdentity) -> None:
        """Non-blocking: queues the identity for async write."""
        self._write_queue.put(("save", identity))

    def load_all(self) -> List[GlobalIdentity]:
        """
        Blocking. Called once at startup to restore state from DB.

        Phase 9 migration: if gallery_json is empty but embedding_sum is not,
        the embedding_sum value is wrapped into a single-exemplar gallery_json
        (camera "legacy") on the fly — no DB write needed; reid_engine will
        persist the proper gallery_json on the next ingest.
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        identities: List[GlobalIdentity] = []

        rows = conn.execute("SELECT * FROM identities").fetchall()
        for row in rows:
            visits_rows = conn.execute(
                "SELECT * FROM zone_visits WHERE global_id = ? ORDER BY enter_time",
                (row["global_id"],),
            ).fetchall()

            visits = [
                ZoneVisit(
                    zone=v["zone"],
                    camera_id=v["camera_id"],
                    enter_time=v["enter_time"],
                    exit_time=v["exit_time"],
                )
                for v in visits_rows
            ]

            # Resolve gallery_json (Phase 9) with migration fallback
            raw_gallery = json.loads(row["gallery_json"] or "[]")
            if not raw_gallery:
                # Attempt legacy migration from embedding_sum
                old_sum = json.loads(row["embedding_sum"] or "[]")
                if old_sum:
                    raw_gallery = [["legacy", old_sum]]

            identity = GlobalIdentity(
                global_id=row["global_id"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                last_camera_id=row["last_camera_id"],
                gallery_json=raw_gallery,
                embedding_count=row["embedding_count"],
                zone_visits=visits,
                active=bool(row["active"]),
            )
            identities.append(identity)

        conn.close()
        return identities

    def truncate(self) -> None:
        """
        Blocking synchronous truncation — called during registry reset.
        Deletes all rows from identities and zone_visits; resets the
        autoincrement counter for zone_visits.
        """
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("DELETE FROM zone_visits")
        conn.execute("DELETE FROM identities")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='zone_visits'")
        conn.commit()
        conn.close()
        print("[persistence] All identity data truncated.")

    def close(self) -> None:
        """Flush pending writes and shut down the writer thread."""
        self._write_queue.put(("stop", None))
        self._writer.join(timeout=5.0)

    # ── Internal: background writer ───────────────────────────────────────────

    def _writer_loop(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        while self._running:
            try:
                cmd, payload = self._write_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if cmd == "stop":
                break
            if cmd == "save" and payload is not None:
                self._do_save(conn, payload)
        conn.close()

    def _do_save(self, conn: sqlite3.Connection, identity: GlobalIdentity) -> None:
        try:
            conn.execute(
                """
                INSERT INTO identities
                    (global_id, first_seen, last_seen, last_camera_id,
                     gallery_json, embedding_count, active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(global_id) DO UPDATE SET
                    last_seen       = excluded.last_seen,
                    last_camera_id  = excluded.last_camera_id,
                    gallery_json    = excluded.gallery_json,
                    embedding_count = excluded.embedding_count,
                    active          = excluded.active
                """,
                (
                    identity.global_id,
                    identity.first_seen,
                    identity.last_seen,
                    identity.last_camera_id,
                    json.dumps(identity.gallery_json),
                    identity.embedding_count,
                    int(identity.active),
                ),
            )

            # Sync zone visits: delete and re-insert (visits are small)
            conn.execute(
                "DELETE FROM zone_visits WHERE global_id = ?",
                (identity.global_id,),
            )
            conn.executemany(
                """
                INSERT INTO zone_visits
                    (global_id, zone, camera_id, enter_time, exit_time)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (identity.global_id, v.zone, v.camera_id, v.enter_time, v.exit_time)
                    for v in identity.zone_visits
                ],
            )
            conn.commit()
        except Exception as e:
            print(f"[persistence] Write error for {identity.global_id}: {e}")
