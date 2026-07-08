"""Multi-person plumbing shared by the orchestrator (opt-in ``multi_person``).

Keeps ``pipeline.py`` focused: the motion-unit abstraction (a clip's single
motion, or one of its per-person tracks), consent gating, and per-person dataset
export live here. HMR itself (``save_tracks``) stays in ``pipeline`` because it
needs the finalize/anonymize path; everything here operates on already-written
pose npz + the manifest, so there's no back-dependency on the orchestrator.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import ingest
from .config import PipelineConfig
from .dataset import build_dataset
from .ingest import Clip, Manifest


def pose_path(cfg: PipelineConfig, clip: Clip) -> Path:
    """Single-subject pose npz path (``pose/<clip_id>.npz``)."""
    return cfg.pose_dir / f"{clip.clip_id}.npz"


@dataclass
class MotionUnit:
    """One recovered motion + who/where it belongs to. A clip's single motion
    (single-subject) or one of its per-person tracks (multi_person) -- so the
    analysis passes and ``build`` iterate both the same way."""

    clip: Clip
    track: Optional[object]  # ingest.Track or None (single-subject)
    pose_path: Path

    @property
    def unit_id(self) -> str:
        return f"{self.clip.clip_id}__{self.track.track_id}" if self.track else self.clip.clip_id

    @property
    def person_id(self) -> Optional[str]:
        return self.track.person_id if self.track else None


def iter_units(cfg: PipelineConfig, manifest: Manifest, *statuses):
    """Yield a :class:`MotionUnit` per existing pose npz (per-track in multi mode)."""
    for clip in manifest.by_status(*(statuses or (ingest.POSE_DONE,))):
        if clip.tracks:
            for track in clip.tracks:
                p = cfg.pose_dir / track.pose_rel if track.pose_rel else None
                if p and p.exists():
                    yield MotionUnit(clip, track, p)
            continue
        p = pose_path(cfg, clip)
        if p.exists():
            yield MotionUnit(clip, None, p)


def set_mirror(unit: MotionUnit, *, suspected: Optional[bool] = None, corrected: bool = False) -> None:
    """Write mirror flags onto the track (multi) or the clip (single)."""
    target = unit.track if unit.track else unit.clip
    if suspected is not None:
        target.suspected_mirrored = suspected
    if corrected:
        target.mirrored = True


def apply_result(clip: Clip, result) -> None:
    """Fold a ``_compute_and_save`` result (int single, or List[Track] multi) onto the clip."""
    if isinstance(result, list):
        clip.tracks = result
        clip.n_frames = sum(t.n_frames or 0 for t in result)
    else:
        clip.n_frames = result


def consent_gate(cfg: PipelineConfig, units: List[MotionUnit]) -> List[MotionUnit]:
    """Drop tracks whose assigned person hasn't consented (fail-closed). No-op in
    single-subject mode or when ``consent_required`` is off."""
    if not (cfg.multi_person and cfg.consent_required):
        return units
    from . import people

    registry = people.load_registry(cfg)
    kept = [u for u in units if people.consent_ok(registry, u.person_id, cfg.unassigned_policy)]
    dropped = len(units) - len(kept)
    if dropped:
        print(f"consent gate: dropped {dropped} track(s) without granted consent")
    return kept


def build_per_person(cfg: PipelineConfig, units: List[MotionUnit]) -> None:
    """Also emit one dataset per person, so Pipeline 2 can fine-tune 'moves like X'."""
    groups: dict = defaultdict(list)
    for u in units:
        if u.person_id:
            groups[u.person_id].append(u)
    for person_id, group in groups.items():
        extra = {u.unit_id: {"person_id": person_id} for u in group}
        build_dataset(
            [u.pose_path for u in group],
            cfg.dataset_dir / "by_person" / person_id,
            gender=cfg.gender, val_fraction=cfg.val_fraction,
            min_frames=cfg.min_clip_frames, extra_per_clip=extra,
        )
