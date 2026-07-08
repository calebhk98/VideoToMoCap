"""People registry + consent ledger (multi_person mode).

A ``person_id`` is a **project-local label** ("dad", "person_00") -- never a stored
biometric. This registry holds, per person, a display name and a consent record;
the dataset ``build`` gates every person's motion on that consent (fail-closed).
Because the owner's rule is "consenting to participate means consenting to
biometric use" (identity clustering needs SMPL shape), a granted person's body
shape may be retained in the separate identity store -- but even then it never
enters the exported dataset, which stays shape-neutral.

Kept as JSON (stdlib, atomic writes, hand-editable) so the core stays light and
the multi-person tests need no YAML. Every consent change is also appended to an
immutable JSONL audit log -- the "who agreed, when, to what" trail.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .config import PipelineConfig


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def load_registry(cfg: PipelineConfig) -> Dict[str, dict]:
    """Load the people registry ({person_id: {display_name, consent}}); {} if none."""
    path = cfg.people_path
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("people", {})


def save_registry(cfg: PipelineConfig, registry: Dict[str, dict]) -> None:
    """Persist the registry atomically."""
    _atomic_write(cfg.people_path, json.dumps({"people": registry}, indent=2, sort_keys=True))


def append_audit(cfg: PipelineConfig, event: dict) -> None:
    """Append one event to the immutable consent/assignment audit log (JSONL)."""
    cfg.consent_log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"at": _now(), **event})
    with cfg.consent_log_path.open("a") as fh:
        fh.write(line + "\n")


def ensure_person(registry: Dict[str, dict], person_id: str, display_name: Optional[str] = None) -> dict:
    """Return the person record, creating a fail-closed (consent-denied) one if new."""
    person = registry.get(person_id)
    if person is None:
        person = {"display_name": display_name or person_id,
                  "consent": {"granted": False, "scope": "all", "granted_on": None, "method": None}}
        registry[person_id] = person
    elif display_name:
        person["display_name"] = display_name
    return person


def set_consent(cfg: PipelineConfig, person_id: str, *, granted: bool,
                method: str = "cli", recorded_by: str = "operator", scope: str = "all") -> Dict[str, dict]:
    """Grant/revoke a person's consent and record it in the audit log. Returns the registry."""
    registry = load_registry(cfg)
    person = ensure_person(registry, person_id)
    person["consent"] = {
        "granted": bool(granted),
        "scope": scope,
        "granted_on": _now() if granted else None,
        "method": method,
    }
    save_registry(cfg, registry)
    append_audit(cfg, {"event": "consent", "person_id": person_id,
                       "granted": bool(granted), "by": recorded_by, "scope": scope})
    return registry


def is_granted(registry: Dict[str, dict], person_id: Optional[str]) -> bool:
    """Whether ``person_id`` has consent currently granted."""
    if person_id is None:
        return False
    person = registry.get(person_id)
    return bool(person and person.get("consent", {}).get("granted"))


def consent_ok(registry: Dict[str, dict], person_id: Optional[str], unassigned_policy: str) -> bool:
    """Whether a track owned by ``person_id`` may export.

    An assigned person needs granted consent. An unassigned track (no person_id)
    follows ``unassigned_policy`` -- 'exclude' (fail-closed) or 'include' (for a
    single-consenting-user project where assignment is moot)."""
    if person_id is None:
        return unassigned_policy == "include"
    return is_granted(registry, person_id)
