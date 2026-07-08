"""Backends for tools whose code/weights aren't publicly released yet.

Registered so ``method: anigs`` is a recognized choice that fails LOUD with why and
what to use instead, rather than an "unknown method" error. Flip them to real
adapters if/when the upstream repos actually ship code.
"""

from __future__ import annotations

from .base import BackendError, Character, CharacterBackend


class _BlockedBackend(CharacterBackend):
    name = "blocked"
    reason = "no public code/weights released as of 2026-07"
    alternative = ""

    def generate(self) -> Character:
        raise BackendError(
            f"{self.name}: {self.reason}. " + (f"Use {self.alternative} instead." if self.alternative else "")
            + " Re-check the upstream repo and swap this for a real adapter when it ships.")


class AniGSBackend(_BlockedBackend):
    name = "anigs"
    role = "single-image animatable Gaussian avatar (CODE NOT RELEASED)"
    reason = "the aigc3d/AniGS repo ships only a README; the authors point to LHM as the working replacement"
    alternative = "method: lhm"


class HumanOrbitBackend(_BlockedBackend):
    name = "humanorbit"
    role = "360-orbit single-image human reconstruction (CODE NOT RELEASED)"
    reason = "HumanOrbit (arXiv 2602.24148) has no public repo/weights"
    alternative = "method: idol or lhm"


BACKENDS = {"anigs": AniGSBackend, "humanorbit": HumanOrbitBackend}
