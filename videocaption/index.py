"""Step 5 -- the searchable index over the captioned archive.

Rows are ``(video_id, rel_path, start_time, end_time, description, tags)``. Stored
as SQLite (queryable; a FTS5 virtual table when the sqlite build has it, else a
plain table with LIKE search) and/or a portable json file. Pure standard library
-- no heavy deps -- so building and querying the index needs nothing installed.

A search hit carries the video + time range, so the UI can jump straight to the
matching moment.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List

from .config import CaptionConfig
from .store import CaptionRow


def _rows_payload(rows: List[CaptionRow]) -> dict:
    return {"rows": [r.to_dict() for r in rows]}


def write_json_index(rows: List[CaptionRow], path: Path) -> Path:
    """Write the portable json index (one flat list of rows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_rows_payload(rows), indent=2))
    tmp.replace(path)
    return path


def _has_fts(conn: sqlite3.Connection) -> bool:
    """Whether this SQLite build ships FTS5 (better free-text search when present)."""
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


def write_sqlite_index(rows: List[CaptionRow], path: Path) -> Path:
    """(Re)build the SQLite index from ``rows``. Uses FTS5 when available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()  # full rebuild: the manifest is the source of truth, not this cache
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE captions (video_id TEXT, rel_path TEXT, seg_index INTEGER, "
            "start REAL, end REAL, description TEXT, tags TEXT)"
        )
        conn.executemany(
            "INSERT INTO captions VALUES (?,?,?,?,?,?,?)",
            [(r.video_id, r.rel_path, r.seg_index, r.start, r.end, r.description, " ".join(r.tags))
             for r in rows],
        )
        if _has_fts(conn):
            conn.execute("CREATE VIRTUAL TABLE captions_fts USING fts5(description, tags, content='captions', content_rowid='rowid')")
            conn.execute("INSERT INTO captions_fts(rowid, description, tags) SELECT rowid, description, tags FROM captions")
        conn.commit()
    finally:
        conn.close()
    return path


def build_index(cfg: CaptionConfig, rows: Iterable[CaptionRow]) -> dict:
    """Write the index in the configured format(s); return a small summary."""
    rows = list(rows)
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    out = {"n_rows": len(rows), "n_videos": len({r.video_id for r in rows})}
    if cfg.index_format in ("json", "both"):
        out["json"] = str(write_json_index(rows, cfg.index_dir / "captions.json"))
    if cfg.index_format in ("sqlite", "both"):
        out["sqlite"] = str(write_sqlite_index(rows, cfg.index_dir / "captions.db"))
    return out


def search(db_path: Path, query: str, *, limit: int = 20) -> List[dict]:
    """Free-text/keyword search over the SQLite index; newest table wins.

    Uses FTS5 MATCH when the index was built with it, else a LIKE scan over
    description + tags. Returns rows as dicts (video, time range, text) so a hit
    can jump to the moment.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        has_fts = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='captions_fts'"
        ).fetchone() is not None
        if has_fts:
            sql = ("SELECT c.* FROM captions_fts f JOIN captions c ON c.rowid = f.rowid "
                   "WHERE captions_fts MATCH ? ORDER BY rank LIMIT ?")
            params = (query, limit)
        else:
            like = f"%{query}%"
            sql = ("SELECT * FROM captions WHERE description LIKE ? OR tags LIKE ? LIMIT ?")
            params = (like, like, limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()
