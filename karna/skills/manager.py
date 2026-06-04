"""Minimal skill loader.

Sprint 2 scope: load a skill's YAML definition and expose the active
version's prompts and tunables. Sprint 6 will extend this to handle:
    - A/B evaluation
    - Auto-promote winner
    - Version history with quality + cost metrics
    - Dependency precheck
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


SKILL_LIB = Path(__file__).resolve().parent / "library"


@dataclass
class SkillVersion:
    name: str               # e.g. "v1"
    status: str             # "active" | "retired"
    method: str
    quality: float | None
    fields: dict[str, Any]  # everything else from the YAML version block


@dataclass
class Skill:
    name: str
    version_label: str
    active: SkillVersion
    raw: dict[str, Any]

    def prompt(self, key: str, **fmt) -> str:
        """Render a prompt field on the active version with **fmt kwargs."""
        template = self.active.fields.get(key)
        if not template:
            raise KeyError(f"Skill {self.name!r} v{self.version_label} has no '{key}' prompt")
        return template.format(**fmt)


@lru_cache(maxsize=64)
def load(skill_name: str) -> Skill:
    """Load `<library>/<skill_name>.yaml` and return the active version.

    Cached because skills are static during a process run.
    """
    path = SKILL_LIB / f"{skill_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No skill at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    active_label = raw.get("active") or "v1"
    versions = raw.get("versions", {})
    if active_label not in versions:
        raise ValueError(f"Skill {skill_name!r} active points to missing version {active_label!r}")

    v = versions[active_label]
    active = SkillVersion(
        name=active_label,
        status=v.get("status", "active"),
        method=v.get("method", ""),
        quality=v.get("quality"),
        fields={k: val for k, val in v.items() if k not in {"status", "method", "quality"}},
    )
    return Skill(name=skill_name, version_label=active_label, active=active, raw=raw)
