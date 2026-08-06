"""Skills file manager: load, save, dedup, and format for system prompt injection.

The on-disk format is a simple YAML file:

    skills:
      - trigger: "when asked to summarize"
        strategy: "lead with the conclusion, then supporting points"
        evidence: "user re-asked for shorter summary"
        added: "2026-05-24T10:30:00+00:00"

Only this module knows the file layout — callers operate on `Skill` objects.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

MAX_SKILLS = 30
DEFAULT_PATH = Path("skills.yaml")
DEFAULT_DEDUP_THRESHOLD = 0.7


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Skill:
    trigger: str
    strategy: str
    evidence: str = ""
    added: str = field(default_factory=_now_iso)

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        return cls(
            trigger=str(data.get("trigger", "")).strip(),
            strategy=str(data.get("strategy", "")).strip(),
            evidence=str(data.get("evidence", "")).strip(),
            added=str(data.get("added") or _now_iso()),
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


def load_skills(path: Path | str = DEFAULT_PATH) -> list[Skill]:
    path = Path(path)
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or {}
    items = raw.get("skills") or []
    return [Skill.from_dict(item) for item in items if item.get("strategy")]


def save_skills(skills: list[Skill], path: Path | str = DEFAULT_PATH) -> None:
    payload = {"skills": [s.to_dict() for s in skills]}
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(_normalize(a).split()), set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def is_duplicate(
    candidate: Skill,
    existing: Iterable[Skill],
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> bool:
    """A candidate is a duplicate when both trigger and strategy overlap heavily
    with an existing skill. Both must match — a shared trigger with a novel
    strategy is still worth keeping."""
    for s in existing:
        trigger_overlap = _token_overlap(candidate.trigger, s.trigger)
        strategy_overlap = _token_overlap(candidate.strategy, s.strategy)
        if trigger_overlap >= threshold and strategy_overlap >= threshold:
            return True
    return False


def add_skill(
    new: Skill,
    path: Path | str = DEFAULT_PATH,
    max_skills: int = MAX_SKILLS,
) -> bool:
    """Append `new` to the skillbook unless it duplicates an existing entry.

    Returns True if the skill was added, False if it was suppressed as a duplicate.
    """
    skills = load_skills(path)
    if is_duplicate(new, skills):
        return False
    skills.append(new)
    if len(skills) > max_skills:
        # FIFO eviction. Replace with confidence-weighted eviction once
        # reflector outputs include a quality score.
        skills = skills[-max_skills:]
    save_skills(skills, path)
    return True


def remove_skills_by_indices(
    indices: set[int],
    path: Path | str = DEFAULT_PATH,
) -> int:
    """Remove the skills at the given 0-based indices. Returns the number
    actually removed. Out-of-range indices are silently ignored."""
    skills = load_skills(path)
    if not skills or not indices:
        return 0
    keep = [s for i, s in enumerate(skills) if i not in indices]
    removed = len(skills) - len(keep)
    if removed:
        save_skills(keep, path)
    return removed


def format_for_prompt(skills: list[Skill]) -> str:
    if not skills:
        return ""
    lines = [
        "Learned strategies from prior conversations "
        "(apply when the trigger matches the current request):"
    ]
    for i, s in enumerate(skills, 1):
        lines.append(f"{i}. {s.trigger}: {s.strategy}")
    return "\n".join(lines)


def inject_skills(messages: list[dict], skills: list[Skill]) -> list[dict]:
    """Return a new message list with the skills block merged into the
    system message (or prepended as a new system message if none exists)."""
    if not skills:
        return messages
    block = format_for_prompt(skills)
    out = list(messages)
    if out and out[0].get("role") == "system":
        existing = out[0].get("content", "") or ""
        merged = f"{existing}\n\n{block}" if existing else block
        out[0] = {**out[0], "content": merged}
    else:
        out.insert(0, {"role": "system", "content": block})
    return out
