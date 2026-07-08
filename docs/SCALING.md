# Scaling to many people / many hours

The default pipeline is built for **personal/family scale** — one JSON manifest,
one machine, exact clustering. That's the right default and stays the default.
This doc is the path to **10⁴ people / 10⁶ hours**, where three assumptions break
and the fixes that replace them.

## The physics (no code removes this)

- **1M hours ≈ 10¹¹ frames.** HMR runs near real-time, so the recovery step is
  ~50–60 GPU-years on 2×3090 — inherently a **cluster** job. Auto-scaling here means
  scaling *across machines*, not squeezing one box.
- **Storage:** anonymized SMPL-H pose is ~40–50 MB/hour → **~45 TB** for 1M hours
  (raw video far more). The dataset must be **sharded on object storage**, not a
  `work/` folder.

So the goal of this layer is: make the state, the work distribution, and the
identity/dataset steps **not depend on everything fitting in one file or one RAM**.

## The three walls and their fixes

| Wall (family-scale code) | Breaks at | Fix (this layer) | Status |
|---|---|---|---|
| Manifest is one JSON, rewritten per clip (`ingest.py`, `pipeline.py`) → O(n²) I/O + all clips in RAM | ~10⁶ clips | **`store.py`**: append-only, per-row updates, atomic claim, streaming reads | **built + tested** |
| Identity clusters a full N×N betas matrix (`identity.py`) | millions of tracks | **mini-batch k-means** (bounded memory) + loud refusal of the O(N²) auto path | **built + tested** |
| Dataset aggregates in-memory into one `index.json` (`dataset.py`) | ~10⁷ clips | sharded AMASS output + sharded index + hash-split | **spec below** |

## Built now

### `videotomocap/store.py` — the coordination backbone

A `ClipStore` interface with a stdlib-sqlite implementation (`SqliteClipStore`,
WAL mode). It replaces "rewrite one JSON per clip" with:

- **append-only inserts** and **O(1) per-clip status updates** (no full rewrite),
- **atomic `claim_next(worker)`** — the work-queue primitive: many workers, *across
  machines*, take distinct clips with no central coordinator (proven by
  `test_concurrent_claims_never_double_assign`),
- **`release_stale(timeout)`** — a crashed worker's lease expires and its clip
  returns to the pool (crash-resumability without a single writer),
- **streaming `iter_by_status` / `counts`** — never loads the whole corpus.

It runs locally with zero infra (one `.db` file). To scale out: point every worker
at one shared DB file (small cluster), or implement `ClipStore` over Postgres/a
cloud queue behind the *same interface* — `open_store(kind=...)` is the seam, and
`import_manifest()` migrates an existing family-scale run into it.

The distribution model this enables (each worker loops):

```
while (clip := store.claim_next(worker_id)):
    run backend on clip            # independent: own scratch, own npz
    store.update_status(clip.clip_id, DONE | FAILED, data={...})
# a supervisor periodically calls store.release_stale(timeout) to reclaim crashes
```

Clips are already independent (the pipeline's design), so this is a wrapping of the
existing per-clip work, not a rewrite of it.

### `videotomocap/identity.py` — scalable person assignment

- `_minibatch_kmeans` scores only a `batch×k` block per step and labels in chunks,
  so peak memory is independent of track count — the path for millions of tracks.
- `cluster_betas` auto-routes: exact k-means / MST below the thresholds
  (`MINIBATCH_THRESHOLD`, `MAX_AUTO_PAIRWISE`), mini-batch above, and a clear error
  if you ask for O(N²) auto-discovery at scale (set `max_people>0` instead).

## Plugs in next (staged, same philosophy)

1. **Wire `store.py` into `pipeline.run_hmr`** behind a config switch
   (`state_backend: manifest|sqlite`), keeping the manifest default. The worker loop
   above replaces the thread-pool-over-a-manifest; per-node it's identical work.
2. **Sharded streaming dataset** (`dataset.py`): write AMASS npz into hashed shards
   (`amass/<shard>/<clip>.npz`), emit a **sharded index** (Parquet/JSONL shards, or
   WebDataset tars), and do the train/val split by a **hash of `clip_id`** so it's
   stable and needs no global permutation. The overfit guard's `corpus_stats` then
   streams the sharded index instead of loading one file.
3. **Distributed execution adapter**: a thin `ClipStore`-over-Postgres (or SQS/Redis)
   so workers span machines; `claim_next`/`release_stale` already define the contract.
4. **Blocked identity at extreme scale**: LSH/coarse-key blocking before
   `_minibatch_kmeans` so even the mini-batch assignment never sees the whole corpus
   at once; ANN (FAISS) for nearest-person lookup as new footage arrives.

## Training side (Pipeline 2) at scale

With 10⁴ donors you train **one conditioned model** (`conditioning: person`) over
the sharded corpus, not 10⁴ fine-tunes. The overfitting guard is already per-donor
(`motion_model/overfit.py`): it reports per-donor coverage, flags thin donors, and
*relaxes* the risk verdict as diversity grows — it just needs the sharded-index
reader from step 2 to stream rather than load one `index.json`.

## Invariants that still hold at scale

Privacy is unchanged: `betas` never enter `store.py` (it carries status + a small
JSON blob of labels/paths, never a biometric), the dataset stays betas-neutral, and
body shape remains only in the consent-gated identity store. Determinism holds:
mini-batch k-means and the hash-split are seeded/stable.
