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


def _choose_k(x: np.ndarray, max_people: int) -> int:
    """Pick the number of people. ``max_people`` when set; else a simple gap
    heuristic on the sorted pairwise spread, capped so a few clips can't over-split."""
    n = len(x)
    if max_people and max_people > 0:
        return min(max_people, n)
    if n <= 2:
        return n
    # Heuristic: distinct people show up as a shape spread well above within-person
    # noise. Grow k while the tightest cluster stays separated; cap at ~sqrt(n).
    cap = max(1, min(n, int(round(np.sqrt(n)))))
    best_k, best_score = 1, -np.inf
    for k in range(1, cap + 1):
        labels = _kmeans(x, k)
        score = _separation(x, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _separation(x: np.ndarray, labels: np.ndarray) -> float:
    """Between-cluster spread minus within-cluster spread (higher = cleaner split)."""
    within = 0.0
    for c in np.unique(labels):
        members = x[labels == c]
        if len(members) > 1:
            within += np.linalg.norm(members - members.mean(axis=0), axis=1).mean()
    centers = np.stack([x[labels == c].mean(axis=0) for c in np.unique(labels)])
    if len(centers) < 2:
        return -within
    between = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    between = between[between > 0].min()
    return float(between - within)


def cluster_betas(betas_by_unit: Dict[str, np.ndarray], max_people: int) -> Dict[str, int]:
    """Group tracks by body shape -> {unit_id: cluster_index}. Deterministic."""
    if not betas_by_unit:
        return {}
    unit_ids = sorted(betas_by_unit)
    x = np.stack([betas_by_unit[u] for u in unit_ids]).astype(np.float64)
    k = _choose_k(x, max_people)
    labels = _kmeans(x, k)
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
        labels = cluster_betas(betas, cfg.max_people)

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
