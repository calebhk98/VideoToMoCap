"""The generate -> critique -> refine loop.

Because the feed-forward lifters (IDOL/LHM/...) can't iteratively refine one 3D
asset, "refinement" here means: generate a candidate, judge the resulting MESH with a
critic, and if it doesn't clear ``accept_score``, regenerate with a fresh seed (and,
for image-native backends, you'd swap the concept image between rounds). The loop
keeps the best candidate across all attempts. GPU-free with the ``noop`` backend +
``noop`` critic, so the whole control flow is unit-tested without weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .base import Character, CharacterBackend
from .config import CharacterConfig
from .critic import CritiqueResult, MeshCritic


@dataclass
class RefineResult:
    """Outcome of a refine run: the best character and how it got there."""

    character: Optional[Character]
    result: Optional[CritiqueResult]
    attempts: List[Tuple[int, int, float, bool]] = field(default_factory=list)  # (attempt, cand, score, passed)
    accepted: bool = False

    @property
    def best_score(self) -> Optional[float]:
        return self.result.score if self.result else None


def refine(cfg: CharacterConfig, backend: CharacterBackend, critic: MeshCritic) -> RefineResult:
    """Generate + critique up to ``max_attempts``x``n_candidates``, keep the best,
    stop as soon as one clears ``accept_score``."""
    best_char: Optional[Character] = None
    best_res: Optional[CritiqueResult] = None
    attempts: List[Tuple[int, int, float, bool]] = []

    for attempt in range(max(1, cfg.max_attempts)):
        cfg.seed = cfg.seed + attempt          # fresh seed per round so refinement differs
        for cand in range(max(1, cfg.n_candidates)):
            char = backend.generate()
            res = critic.score(char, cfg.prompt)
            attempts.append((attempt, cand, round(res.score, 4), res.passed))
            if best_res is None or res.score > best_res.score:
                best_char, best_res = char, res
            if res.passed:
                return RefineResult(best_char, best_res, attempts, accepted=True)
    return RefineResult(best_char, best_res, attempts, accepted=False)


def format_refine(r: RefineResult) -> str:
    """Render a RefineResult for the CLI."""
    head = "ACCEPTED" if r.accepted else "best-effort (never cleared accept_score)"
    lines = [f"Character refine: {head}",
             f"  best score: {r.best_score:.3f}" if r.best_score is not None else "  no candidate produced"]
    for attempt, cand, score, passed in r.attempts:
        mark = "PASS" if passed else "..."
        lines.append(f"  attempt {attempt} cand {cand}: {score:.3f} {mark}")
    if r.character is not None:
        drive = "SMPL-X-native (drives with no retarget)" if r.character.native_smplx else f"rig={r.character.rig}"
        lines.append(f"  -> {r.character.asset_path} [{drive}]")
    return "\n".join(lines)
