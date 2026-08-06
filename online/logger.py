"""Conversation logger.

Writes each completed chat to `conversations/{timestamp}.json` and tracks
which ones the reflector has already reviewed via a sidecar `.reviewed`
marker file (one filename per line).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DIR = Path("conversations")
REVIEWED_MARKER = ".reviewed"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def log_conversation(
    messages: list[dict],
    response: dict,
    directory: Path | str = DEFAULT_DIR,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_timestamp()}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
        "response": response,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def list_conversations(directory: Path | str = DEFAULT_DIR) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def _reviewed_set(directory: Path) -> set[str]:
    marker = directory / REVIEWED_MARKER
    if not marker.exists():
        return set()
    return {line.strip() for line in marker.read_text().splitlines() if line.strip()}


def get_unreviewed(directory: Path | str = DEFAULT_DIR) -> list[Path]:
    directory = Path(directory)
    reviewed = _reviewed_set(directory)
    return [p for p in list_conversations(directory) if p.name not in reviewed]


def mark_reviewed(paths: list[Path], directory: Path | str = DEFAULT_DIR) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reviewed = _reviewed_set(directory) | {p.name for p in paths}
    (directory / REVIEWED_MARKER).write_text("\n".join(sorted(reviewed)))
