"""Scalability tests for cross-clip identity clustering (videotomocap.identity).

The default (family-scale) paths build an N*N or N*k block; these tests cover the
paths that let identity assignment reach millions of tracks: mini-batch k-means
(bounded memory) and a loud refusal of the O(N^2) auto path at scale. GPU-free.
Runs under pytest OR directly: `python tests/test_identity_scale.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap import identity


def _blobs(n_per: int, means: list, dim: int = 16, noise: float = 0.5, seed: int = 0):
    """Well-separated Gaussian blobs -> (x, true_label per row)."""
    rng = np.random.default_rng(seed)
    xs, truth = [], []
    for lbl, m in enumerate(means):
        xs.append(rng.normal(m, noise, size=(n_per, dim)))
        truth += [lbl] * n_per
    return np.vstack(xs), np.array(truth)


def _purity(pred: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of rows whose predicted cluster's majority true-blob they match."""
    correct = 0
    for c in np.unique(pred):
        members = truth[pred == c]
        correct += np.bincount(members).max()
    return correct / len(truth)


def test_assign_chunked_matches_bruteforce():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 8))
    centers = rng.normal(size=(5, 8))
    chunked = identity._assign_chunked(x, centers, chunk=64)
    brute = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
    assert np.array_equal(chunked, brute)


def test_minibatch_kmeans_recovers_separated_blobs():
    x, truth = _blobs(300, means=[0.0, 50.0, 100.0])   # 900 rows, far apart
    labels = identity._minibatch_kmeans(x, k=3)
    assert len(np.unique(labels)) == 3
    assert _purity(labels, truth) > 0.98               # clean separation


def test_minibatch_kmeans_is_deterministic():
    x, _ = _blobs(200, means=[0.0, 30.0])
    a = identity._minibatch_kmeans(x, k=2)
    b = identity._minibatch_kmeans(x, k=2)
    assert np.array_equal(a, b)


def test_cluster_betas_routes_to_minibatch_above_threshold():
    # 2700 tracks > MINIBATCH_THRESHOLD, fixed k=3 -> mini-batch path, no N*N blowup.
    x, truth = _blobs(900, means=[0.0, 50.0, 100.0])
    betas = {f"u{i:05d}": x[i] for i in range(len(x))}
    assert len(x) > identity.MINIBATCH_THRESHOLD
    labels = identity.cluster_betas(betas, max_people=3)
    assert len(set(labels.values())) == 3
    pred = np.array([labels[f"u{i:05d}"] for i in range(len(x))])
    assert _purity(pred, truth) > 0.98


def test_auto_discovery_refused_at_scale():
    betas = {f"u{i}": np.zeros(16) for i in range(identity.MAX_AUTO_PAIRWISE + 1)}
    try:
        identity.cluster_betas(betas, max_people=0)     # auto path builds N*N -> refuse
        raise AssertionError("expected ValueError at scale")
    except ValueError as exc:
        assert "max_people>0" in str(exc) and "N*N" in str(exc)


def test_small_corpus_still_uses_exact_paths():
    # Below the thresholds, exact k-means + auto-discovery still run (family scale).
    x, truth = _blobs(20, means=[0.0, 40.0])
    betas = {f"u{i:03d}": x[i] for i in range(len(x))}
    fixed = identity.cluster_betas(betas, max_people=2)
    assert len(set(fixed.values())) == 2
    auto = identity.cluster_betas(betas, max_people=0)   # MST gap-cut, N small
    assert len(set(auto.values())) == 2


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} identity-scale tests passed")


if __name__ == "__main__":
    _run_all()
