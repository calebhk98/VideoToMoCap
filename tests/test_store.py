"""Unit tests for the distributed-scale clip store (videotomocap.store).

GPU-free and dependency-free (stdlib sqlite3). The load-bearing test is
`test_concurrent_claims_never_double_assign`: many threads hammer claim_next and
every clip must be handed out exactly once -- that atomicity is what lets workers
across machines share one queue without a coordinator. Runs under pytest OR
directly: `python tests/test_store.py`.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap.store import (
    DONE, FAILED, IN_PROGRESS, PENDING,
    ClipRecord, SqliteClipStore, import_manifest, open_store,
)


def _clips(n: int) -> list:
    return [ClipRecord(clip_id=f"clip{i:04d}", source=f"vid{i}.mp4") for i in range(n)]


def _store(tmp: Path, **kw) -> SqliteClipStore:
    return SqliteClipStore(tmp / "clips.db", **kw)


def test_add_is_idempotent_and_counts():
    with TemporaryDirectory() as t:
        s = _store(Path(t))
        assert s.add_clips(_clips(5)) == 5
        assert s.add_clips(_clips(5)) == 0          # re-scan adds nothing new
        assert s.counts() == {PENDING: 5}


def test_add_never_clobbers_progress():
    with TemporaryDirectory() as t:
        s = _store(Path(t))
        s.add_clips(_clips(3))
        s.update_status("clip0000", DONE)
        s.add_clips(_clips(3))                       # a re-scan must not reset clip0000
        assert s.get("clip0000").status == DONE


def test_claim_marks_in_progress_and_drains():
    with TemporaryDirectory() as t:
        s = _store(Path(t))
        s.add_clips(_clips(2))
        a = s.claim_next("w1")
        assert a.status == IN_PROGRESS and a.worker == "w1"
        b = s.claim_next("w1")
        assert b.clip_id != a.clip_id
        assert s.claim_next("w1") is None            # queue drained
        assert s.counts()[IN_PROGRESS] == 2


def test_update_merges_data_blob():
    with TemporaryDirectory() as t:
        s = _store(Path(t))
        s.add_clips(_clips(1))
        s.update_status("clip0000", IN_PROGRESS, data={"quality": "ok"})
        s.update_status("clip0000", DONE, data={"person_id": "person_00"})
        rec = s.get("clip0000")
        assert rec.status == DONE and rec.data == {"quality": "ok", "person_id": "person_00"}


def test_release_stale_reclaims_crashed_worker():
    clock = {"t": 1000.0}
    with TemporaryDirectory() as t:
        s = _store(Path(t), now_fn=lambda: clock["t"])
        s.add_clips(_clips(1))
        s.claim_next("w1")                           # leased at t=1000
        clock["t"] = 1000.0 + 100                     # 100s later
        assert s.release_stale(timeout_s=120) == 0    # lease still fresh
        clock["t"] = 1000.0 + 500                     # long gone
        assert s.release_stale(timeout_s=120) == 1    # reclaimed
        assert s.get("clip0000").status == PENDING and s.get("clip0000").worker is None


def test_iter_by_status_streams_filtered():
    with TemporaryDirectory() as t:
        s = _store(Path(t))
        s.add_clips(_clips(4))
        s.update_status("clip0000", DONE)
        s.update_status("clip0001", FAILED)
        pending = [r.clip_id for r in s.iter_by_status(PENDING)]
        assert pending == ["clip0002", "clip0003"]
        assert len(list(s.iter_by_status())) == 4     # no filter -> all


def test_concurrent_claims_never_double_assign():
    with TemporaryDirectory() as t:
        s = _store(Path(t))
        n = 200
        s.add_clips(_clips(n))
        claimed: list = []
        lock = threading.Lock()

        def drain(worker: str):
            while True:
                rec = s.claim_next(worker)
                if rec is None:
                    return
                with lock:
                    claimed.append(rec.clip_id)

        threads = [threading.Thread(target=drain, args=(f"w{i}",)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert len(claimed) == n                      # each clip claimed...
        assert len(set(claimed)) == n                 # ...exactly once (no double-assign)


def test_open_store_factory_and_unknown_kind():
    with TemporaryDirectory() as t:
        s = open_store(Path(t) / "c.db", "sqlite")
        assert isinstance(s, SqliteClipStore)
        try:
            open_store(Path(t) / "c.db", "postgres")
            raise AssertionError("expected ValueError for unknown kind")
        except ValueError:
            pass


def test_import_manifest_maps_statuses():
    from types import SimpleNamespace

    with TemporaryDirectory() as t:
        s = _store(Path(t))
        manifest = SimpleNamespace(clips=[
            SimpleNamespace(clip_id="a", status="pending", source_path="a.mp4"),
            SimpleNamespace(clip_id="b", status="pose_done", source_path="b.mp4"),
            SimpleNamespace(clip_id="c", status="failed", source_path="c.mp4"),
        ])
        assert import_manifest(s, manifest) == 3
        assert s.get("a").status == PENDING
        assert s.get("b").status == DONE
        assert s.get("c").status == FAILED
        assert s.get("b").source == "b.mp4"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} store tests passed")


if __name__ == "__main__":
    _run_all()
