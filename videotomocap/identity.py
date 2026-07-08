"""Cross-clip identity: cluster per-track SMPL betas into people (multi_person).

Body shape (``betas``) is a stable per-person signature, so clustering the betas
of every recovered track across the corpus recovers "who is who" for a small
closed set of consenting people (a family). This is the ``person_assignment:
shape`` path. Betas live only in the consent-gated identity store
(``cfg.identity_dir``) -- never in the exported dataset, which stays shape-neutral.

Pure NumPy: a small deterministic k-means, seeded, so assignment is reproducible.
For faces/gait/appearance re-ID (stronger cues when shape is ambiguous) this is
the seam to extend -- the assignment just needs to return a {unit_id: person_id}
map; the rest of the pipeline doesn't care how it was produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import PipelineConfig
from .ingest import Manifest


def _betas_path(cfg: PipelineConfig, unit_id: str) -> Path:
    return cfg.identity_dir / f"{unit_id}.npz"


def save_track_betas(cfg: PipelineConfig, unit_id: str, betas: Optional[np.ndarray]) -> None:
    """Retain a track's raw betas for identity (no-op if the backend gave none)."""
    if betas is None:
        return
    cfg.identity_dir.mkdir(parents=True, exist_ok=True)
    np.savez(_betas_path(cfg, unit_id), betas=np.asarray(betas, np.float32).reshape(-1))


def load_all_betas(cfg: PipelineConfig, unit_ids: List[str]) -> Dict[str, np.ndarray]:
    """Load retained betas for the given tracks (skips any without a stored vector)."""
    out: Dict[str, np.ndarray] = {}
    for uid in unit_ids:
        path = _betas_path(cfg, uid)
        if path.exists():
            out[uid] = np.load(path)["betas"]
    return out


def _kmeans(x: np.ndarray, k: int, *, seed: int = 0, iters: int = 50) -> np.ndarray:
    """Tiny deterministic k-means -> per-row cluster label. Pure NumPy."""
    rng = np.random.default_rng(seed)
    k = max(1, min(k, len(x)))
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        new = dists.argmin(axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for c in range(k):
            members = x[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)
    return labels


def _cluster_auto(x: np.ndarray, gap_ratio: float) -> np.ndarray:
    """Discover the number of people from the shapes themselves -> per-row label.

    Single-linkage (MST) agglomerative + a gap cut: build the minimum spanning tree
    over the betas, then cut the edges *above* the largest relative jump in merge
    distance. Within-person merges are tight and similar; the jump to the first
    between-person merge is large -- so ONE person (a single tight blob, no jump)
    stays one cluster, and distinct people separate cleanly. Robust where a fixed-k
    or variance heuristic would over-split a single person's natural jitter.
    """
    n = len(x)
    if n <= 1:
        return np.zeros(n, dtype=int)

    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    iu = np.triu_indices(n, 1)
    order = np.argsort(d[iu])
    edges = [(int(iu[0][k]), int(iu[1][k]), float(d[iu][k])) for k in order]

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    mst = []  # (i, j, dist) in ascending merge order (single linkage)
    for i, j, dist in edges:
        if find(i) != find(j):
            parent[find(i)] = find(j)
            mst.append((i, j, dist))

    n_cut = _cut_count(np.array([e[2] for e in mst]), gap_ratio)
    keep = mst[: len(mst) - n_cut]  # drop the longest (between-person) edges

    parent = list(range(n))
    for i, j, _ in keep:
        parent[find(i)] = find(j)
    roots: Dict[int, int] = {}
    labels = np.zeros(n, dtype=int)
    for a in range(n):
        labels[a] = roots.setdefault(find(a), len(roots))
    return labels


def _cut_count(dists: np.ndarray, gap_ratio: float) -> int:
    """How many of the longest MST edges to cut: those past the largest relative
    jump, but only if that jump is a real gap (>= ``gap_ratio``x). Else 0 -> one
    person. Two tracks with no baseline default to merged (conservative)."""
    if len(dists) < 2:
        return 0
    ratios = dists[1:] / np.maximum(dists[:-1], 1e-9)
    idx = int(np.argmax(ratios))
    if ratios[idx] < gap_ratio:
        return 0
    return len(dists) - (idx + 1)


def cluster_betas(betas_by_unit: Dict[str, np.ndarray], max_people: int,
                  gap_ratio: float = 3.0) -> Dict[str, int]:
    """Group tracks by body shape -> {unit_id: cluster_index}. Deterministic.

    ``max_people`` > 0 fixes the count (k-means); 0 = auto-discover it (MST gap).
    """
    if not betas_by_unit:
        return {}
    unit_ids = sorted(betas_by_unit)
    x = np.stack([betas_by_unit[u] for u in unit_ids]).astype(np.float64)
    if max_people and max_people > 0:
        labels = _kmeans(x, min(max_people, len(x)))
    else:
        labels = _cluster_auto(x, gap_ratio)
    return {uid: int(lbl) for uid, lbl in zip(unit_ids, labels)}


def assign_people(cfg: PipelineConfig, manifest: Manifest) -> Dict[str, int]:
    """Cluster every track's betas into people and write ``person_id`` onto tracks.

    Returns {person_id: n_tracks}. Person ids are stable ``person_NN`` labels the
    operator can rename in the registry. Only runs for ``person_assignment:
    shape``; 'manual' leaves assignment to the CLI, 'single' maps all tracks to one
    person.
    """
    from . import ingest, people

    units = [(f"{c.clip_id}__{t.track_id}", c, t)
             for c in manifest.by_status(ingest.POSE_DONE) for t in c.tracks]
    if not units:
        return {}

    if cfg.person_assignment == "single":
        labels = {uid: 0 for uid, _, _ in units}
    else:  # 'shape' (manual assignment is done via the CLI, not here)
        betas = load_all_betas(cfg, [uid for uid, _, _ in units])
        labels = cluster_betas(betas, cfg.max_people, gap_ratio=cfg.shape_gap_ratio)

    registry = people.load_registry(cfg)
    counts: Dict[str, int] = {}
    for uid, _clip, track in units:
        cluster = labels.get(uid)
        person_id = f"person_{cluster:02d}" if cluster is not None else None
        track.person_id = person_id
        if person_id is not None:
            people.ensure_person(registry, person_id)
            counts[person_id] = counts.get(person_id, 0) + 1
    people.save_registry(cfg, registry)
    people.append_audit(cfg, {"event": "assign", "method": cfg.person_assignment,
                              "people": counts})
    manifest.save(cfg.manifest_path)
    return counts
