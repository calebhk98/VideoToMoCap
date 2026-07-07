"""Unsupervised action pseudo-labels for the motion clips.

Your footage has no action labels, but the motion model (Pipeline 2) wants
*something* to condition on. Rather than hand-label thousands of clips, we group
them by how they move -- a small k-means over per-clip motion statistics -- and
hand each clip its cluster id. Those pseudo-labels ("cluster 3" ~ walking,
"cluster 7" ~ reaching, etc.) give MDM a free conditioning signal; you can name
the clusters later if you want, or just condition on the id.

Pure NumPy, deterministic (seeded), no external deps. It's clustering, not
recognition -- the clusters are only as meaningful as the motion statistics that
feed them, so treat the ids as coarse buckets, not ground-truth actions.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .pose import SMPL_NJOINTS, SmplMotion


def clip_features(motion: SmplMotion) -> np.ndarray:
    """A fixed-length motion fingerprint for one clip (per-joint speed stats).

    Captures *how* the body moves (which joints, how fast, how variably) without
    caring where it is or which way it faces -- so clustering groups by action,
    not by location/orientation.
    """
    dim = SMPL_NJOINTS * 2 + 2
    if motion.n_frames < 2:
        return np.zeros(dim, np.float32)
    poses = motion.poses.reshape(motion.n_frames, SMPL_NJOINTS, 3)
    speed = np.linalg.norm(np.diff(poses, axis=0), axis=2)   # (T-1, 24) per-joint angular speed
    root = np.linalg.norm(np.diff(motion.trans, axis=0), axis=1)
    feat = np.concatenate([speed.mean(0), speed.std(0), [root.mean(), root.std()]])
    return feat.astype(np.float32)


def _standardize(x: np.ndarray) -> np.ndarray:
    """Z-score each feature column so no single dimension dominates the distance."""
    mean = x.mean(0, keepdims=True)
    std = x.std(0, keepdims=True)
    return (x - mean) / np.where(std < 1e-8, 1.0, std)


def kmeans(x: np.ndarray, k: int, *, seed: int = 0, iters: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Plain Lloyd's k-means. Returns (labels, centers). k is clamped to n_samples."""
    n = x.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for step in range(iters):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # (n, k)
        new_labels = dist.argmin(1)
        if step > 0 and np.array_equal(new_labels, labels):
            break  # converged
        labels = new_labels
        for c in range(k):
            members = labels == c
            if members.any():
                centers[c] = x[members].mean(0)
    return labels, centers


def cluster_clips(features_by_id: Dict[str, np.ndarray], k: int, *, seed: int = 0) -> Dict[str, int]:
    """Assign each clip a cluster id in [0, k). Empty input -> {}."""
    if not features_by_id:
        return {}
    ids = list(features_by_id)
    x = _standardize(np.stack([features_by_id[i] for i in ids]).astype(np.float64))
    labels, _ = kmeans(x, k, seed=seed)
    return {clip_id: int(label) for clip_id, label in zip(ids, labels)}
