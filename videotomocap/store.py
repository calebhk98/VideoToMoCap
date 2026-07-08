"""Distributed-scale clip state: an append-only, claim-based store.

The default JSON manifest (``ingest.Manifest``) rewrites the *whole* file after
every clip -- O(n^2) I/O -- and holds every clip in RAM. That's right for a family
and fatal for 10^4 people / 10^6 hours. At that scale the state layer has to:

  (a) update one clip without rewriting the rest,
  (b) let many workers -- across machines -- claim work atomically, with no shared
      in-memory manifest and no central coordinator, and
  (c) stream clips by status instead of loading them all.

:class:`ClipStore` is that interface. :class:`SqliteClipStore` implements it on
stdlib ``sqlite3`` in WAL mode (concurrent readers + serialized writers; an atomic
claim via a single guarded ``UPDATE``) -- no new dependency, cross-platform, and it
runs locally with zero infra. Point every worker at one shared DB file for a small
cluster, or drop in a Postgres-backed ``ClipStore`` behind the same interface for a
big one; the pipeline never learns which. Selected by config exactly like an HMR
backend, with the JSON manifest staying the default so nothing at family scale moves.

The claim/lease model is what makes the work *distributable*: a worker claims the
next pending clip (marking it in-progress with its own id + a timestamp), does the
GPU work independently (clips are already independent -- own scratch, own npz), then
marks it done/failed. A crashed worker's lease goes stale and :meth:`release_stale`
returns the clip to the pool -- the same crash-resumability the manifest gives, but
without a single writer.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

# Clip lifecycle. Kept as plain strings (not an enum) so a Postgres/other backend
# stores them verbatim and the values survive a JSON round-trip unchanged.
PENDING = "pending"
IN_PROGRESS = "in_progress"
DONE = "done"
FAILED = "failed"


@dataclass
class ClipRecord:
    """One clip's coordination state. Pipeline-specific fields live in ``data`` (a
    JSON blob) so the store schema stays generic across HMR/refine/dataset stages."""

    clip_id: str
    status: str = PENDING
    source: str = ""            # e.g. the source video path (rel/posix)
    worker: Optional[str] = None
    updated: float = 0.0
    data: dict = field(default_factory=dict)


class ClipStore(ABC):
    """Backend-agnostic clip state + work queue. Implementations must make
    :meth:`claim_next` atomic under concurrent callers (that's the whole point)."""

    @abstractmethod
    def add_clips(self, records: Iterable[ClipRecord]) -> int:
        """Insert new clips (existing ``clip_id``s are left untouched -- a re-scan
        must never reset progress). Returns the number actually inserted."""

    @abstractmethod
    def claim_next(self, worker: str, statuses: Iterable[str] = (PENDING,)) -> Optional[ClipRecord]:
        """Atomically take one clip in ``statuses``, mark it in-progress for
        ``worker``, and return it. ``None`` when the queue is drained. Two workers
        calling this concurrently never receive the same clip."""

    @abstractmethod
    def update_status(self, clip_id: str, status: str,
                      data: Optional[dict] = None, worker: Optional[str] = None) -> None:
        """Set a clip's status (and merge ``data``) without touching other clips."""

    @abstractmethod
    def get(self, clip_id: str) -> Optional[ClipRecord]:
        """Fetch one clip, or ``None``."""

    @abstractmethod
    def iter_by_status(self, *statuses: str) -> Iterator[ClipRecord]:
        """Stream clips (optionally filtered by status) without loading them all."""

    @abstractmethod
    def counts(self) -> Dict[str, int]:
        """Cheap ``{status: n}`` aggregate for progress reporting."""

    @abstractmethod
    def release_stale(self, timeout_s: float) -> int:
        """Return in-progress clips whose lease is older than ``timeout_s`` back to
        pending (a crashed worker's work). Returns how many were reclaimed."""

    def close(self) -> None:  # pragma: no cover - trivial default
        """Release resources (a no-op for in-memory backends)."""


class SqliteClipStore(ClipStore):
    """WAL-mode sqlite implementation -- concurrent readers, atomic claims, O(1)
    per-clip updates. Works on a local file today; a shared file (or a networked
    engine behind the same interface) scales it to many workers."""

    def __init__(self, path: Path, now_fn: Callable[[], float] = time.time):
        # now_fn is injectable so lease-expiry logic is deterministic in tests.
        self._now = now_fn
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One sqlite Connection per thread. A single Connection shared across threads
        # can't run concurrent transactions (its txn state is per-connection), which
        # is exactly the collision a multi-worker claim loop would hit. Separate
        # connections + WAL coordinate at the *file* level -- the same model a real
        # multi-process/multi-machine deployment uses. Track them so close() is clean.
        self._local = threading.local()
        self._conns: List[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        db = self._conn()
        db.execute(
            "CREATE TABLE IF NOT EXISTS clips ("
            "clip_id TEXT PRIMARY KEY, status TEXT NOT NULL, source TEXT, "
            "worker TEXT, updated REAL, data TEXT)"
        )
        db.execute("CREATE INDEX IF NOT EXISTS ix_status ON clips(status)")

    def _new_conn(self) -> sqlite3.Connection:
        # isolation_level=None -> autocommit, so our explicit BEGIN IMMEDIATE is the
        # only transaction (no implicit one to collide with). busy_timeout makes a
        # contended writer wait for the WAL write-lock instead of erroring out.
        conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _conn(self) -> sqlite3.Connection:
        """The calling thread's own connection, created on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        conn = self._new_conn()
        self._local.conn = conn
        with self._conns_lock:
            self._conns.append(conn)
        return conn

    def add_clips(self, records: Iterable[ClipRecord]) -> int:
        rows = [(r.clip_id, r.status, r.source, r.worker, r.updated or self._now(),
                 json.dumps(r.data)) for r in records]
        if not rows:
            return 0
        # INSERT OR IGNORE: a re-scan adds only genuinely new clips, never clobbers
        # a clip already mid-flight or done. total_changes deltas give the insert count.
        db = self._conn()
        before = db.total_changes
        db.executemany(
            "INSERT OR IGNORE INTO clips(clip_id, status, source, worker, updated, data) "
            "VALUES (?, ?, ?, ?, ?, ?)", rows)
        return db.total_changes - before

    def claim_next(self, worker: str, statuses: Iterable[str] = (PENDING,)) -> Optional[ClipRecord]:
        db = self._conn()
        placeholders = ",".join("?" for _ in statuses)
        params = list(statuses)
        # BEGIN IMMEDIATE grabs the write lock up front, so two concurrent claimers
        # (on their own connections) can't both read the same pending row then race.
        db.execute("BEGIN IMMEDIATE")
        try:
            row = db.execute(
                f"SELECT * FROM clips WHERE status IN ({placeholders}) ORDER BY clip_id LIMIT 1",
                params).fetchone()
            if row is None:
                db.execute("COMMIT")
                return None
            now = self._now()
            db.execute("UPDATE clips SET status=?, worker=?, updated=? WHERE clip_id=?",
                       (IN_PROGRESS, worker, now, row["clip_id"]))
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
        rec = _row_to_record(row)
        rec.status, rec.worker, rec.updated = IN_PROGRESS, worker, now
        return rec

    def update_status(self, clip_id: str, status: str,
                      data: Optional[dict] = None, worker: Optional[str] = None) -> None:
        db = self._conn()
        now = self._now()
        if data is None:
            db.execute("UPDATE clips SET status=?, updated=? WHERE clip_id=?", (status, now, clip_id))
            return
        # Merge into the existing data blob rather than overwrite, so stage-by-stage
        # fields (quality, person_id, ...) accumulate on one row.
        merged = self._merged_data(clip_id, data)
        db.execute(
            "UPDATE clips SET status=?, updated=?, data=?, worker=COALESCE(?, worker) WHERE clip_id=?",
            (status, now, json.dumps(merged), worker, clip_id))

    def _merged_data(self, clip_id: str, updates: dict) -> dict:
        row = self._conn().execute("SELECT data FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
        base = json.loads(row["data"]) if row and row["data"] else {}
        base.update(updates)
        return base

    def get(self, clip_id: str) -> Optional[ClipRecord]:
        row = self._conn().execute("SELECT * FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
        return _row_to_record(row) if row else None

    def iter_by_status(self, *statuses: str) -> Iterator[ClipRecord]:
        db = self._conn()
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            cur = db.execute(
                f"SELECT * FROM clips WHERE status IN ({placeholders}) ORDER BY clip_id", list(statuses))
        else:
            cur = db.execute("SELECT * FROM clips ORDER BY clip_id")
        for row in cur:   # cursor streams; the whole table is never materialized
            yield _row_to_record(row)

    def counts(self) -> Dict[str, int]:
        cur = self._conn().execute("SELECT status, COUNT(*) AS n FROM clips GROUP BY status")
        return {row["status"]: row["n"] for row in cur}

    def release_stale(self, timeout_s: float) -> int:
        db = self._conn()
        cutoff = self._now() - timeout_s
        before = db.total_changes
        db.execute("UPDATE clips SET status=?, worker=NULL WHERE status=? AND updated < ?",
                   (PENDING, IN_PROGRESS, cutoff))
        return db.total_changes - before

    def close(self) -> None:
        with self._conns_lock:
            for conn in self._conns:
                conn.close()
            self._conns.clear()


def _row_to_record(row: sqlite3.Row) -> ClipRecord:
    """Map a DB row to a ClipRecord, decoding the JSON ``data`` blob."""
    return ClipRecord(
        clip_id=row["clip_id"], status=row["status"], source=row["source"] or "",
        worker=row["worker"], updated=row["updated"] or 0.0,
        data=json.loads(row["data"]) if row["data"] else {},
    )


def open_store(path: Path, kind: str = "sqlite", now_fn: Callable[[], float] = time.time) -> ClipStore:
    """Open a clip store by kind. Only 'sqlite' ships in-repo; the seam is here so a
    Postgres/cloud backend is a registry entry, not a pipeline change."""
    if kind == "sqlite":
        return SqliteClipStore(path, now_fn=now_fn)
    raise ValueError(f"unknown clip store kind {kind!r} (have: 'sqlite')")


def import_manifest(store: ClipStore, manifest, source_attr: str = "source_path") -> int:
    """Seed a store from an existing JSON ``Manifest`` (the migration path off the
    family-scale format). Maps each clip's status best-effort; unknown -> pending."""
    records = []
    for clip in getattr(manifest, "clips", []):
        raw = getattr(clip, "status", PENDING)
        records.append(ClipRecord(
            clip_id=clip.clip_id, status=_map_status(str(raw)),
            source=str(getattr(clip, source_attr, "") or ""),
        ))
    return store.add_clips(records)


def _map_status(status: str) -> str:
    """Best-effort map of a manifest status string onto the store lifecycle."""
    s = status.lower()
    if "fail" in s:
        return FAILED
    if "done" in s or "pose" in s or "complete" in s:
        return DONE
    return PENDING
