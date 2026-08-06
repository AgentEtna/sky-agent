"""Strategy extractor.

Walks unreviewed conversation logs, asks the same local LLM to extract a
concrete lesson per conversation, and appends novel lessons to the skillbook.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx

from .logger import DEFAULT_DIR, get_unreviewed, mark_reviewed
from .skills import (
    DEFAULT_PATH,
    Skill,
    add_skill,
    load_skills,
    remove_skills_by_indices,
)

REFLECTION_PROMPT = """\
Review this conversation. Look for:
- Did the user correct the assistant? What was wrong?
- Did the user re-ask or rephrase? What was unclear?
- Did the user seem satisfied?

If you find a concrete lesson, output it as:
trigger: <when this applies>
strategy: <what to do differently>
evidence: <what happened in the conversation>

If the conversation went fine and there's nothing to learn, output: NO_LESSON

Conversation:
{conversation}
"""

PRUNE_PROMPT = """\
Below is the current list of learned strategies (skills) and a sample of recent
conversations the assistant produced. Identify any skills that should be REMOVED
because they:
- Caused the assistant to produce a wrong, irrelevant, or unhelpful response
- Directly contradict another skill in the list
- Are too vague or generic to ever be actionable
- Are clearly wrong

Be conservative. Only flag a skill if you are confident it is harmful or useless.
If the skill list looks fine, output NONE.

Current skills (numbered):
{skill_list}

Recent conversations:
{conversation_block}

Respond with one line per skill to remove, in this exact format:
REMOVE: <number>

Or, if nothing should be removed:
NONE
"""

# Cap the number of recent conversations we feed into the prune prompt so the
# request stays small. Five is enough to catch contradictions without ballooning
# context.
PRUNE_CONVERSATION_SAMPLE = 5


def _format_conversation(payload: dict) -> str:
    lines: list[str] = []
    for m in payload.get("messages", []):
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            # OpenAI multimodal content arrays: concatenate text parts.
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        lines.append(f"[{role}] {content}")

    response = payload.get("response") or {}
    for choice in response.get("choices", []):
        text = (choice.get("message") or {}).get("content") or ""
        if text:
            lines.append(f"[assistant] {text}")
    return "\n".join(lines)


def _parse_lesson(text: str) -> Skill | None:
    text = (text or "").strip()
    if not text or "NO_LESSON" in text.upper():
        return None

    fields: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"\s*(trigger|strategy|evidence)\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            fields[m.group(1).lower()] = m.group(2).strip()

    if not fields.get("trigger") or not fields.get("strategy"):
        return None
    return Skill(
        trigger=fields["trigger"],
        strategy=fields["strategy"],
        evidence=fields.get("evidence", ""),
    )


def _parse_prune_indices(text: str, n_skills: int) -> set[int]:
    """Extract 0-based indices from `REMOVE: <n>` lines (input is 1-indexed)."""
    if not text or "NONE" in text.upper().split():
        return set()
    out: set[int] = set()
    for line in text.splitlines():
        m = re.match(r"\s*REMOVE\s*:\s*(\d+)\s*$", line, re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n_skills:
                out.add(idx)
    return out


async def _prune_skills(
    client: httpx.AsyncClient,
    upstream: str,
    model: str,
    payloads: list[dict],
    skills_path: Path | str,
    headers: dict[str, str] | None = None,
) -> int:
    """Ask the judge which existing skills (if any) should be deleted, then
    delete them silently. Returns the number removed."""
    skills = load_skills(skills_path)
    if not skills:
        return 0

    skill_list = "\n".join(
        f"{i + 1}. {s.trigger}: {s.strategy}" for i, s in enumerate(skills)
    )
    sample = payloads[-PRUNE_CONVERSATION_SAMPLE:] or []
    if not sample:
        return 0
    convo_block = "\n\n---\n\n".join(_format_conversation(p) for p in sample)

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": PRUNE_PROMPT.format(
                    skill_list=skill_list,
                    conversation_block=convo_block,
                ),
            }
        ],
        "stream": False,
        "temperature": 0,
    }
    r = await client.post(
        f"{upstream.rstrip('/')}/v1/chat/completions",
        json=body,
        headers=headers or {},
        timeout=120.0,
    )
    r.raise_for_status()
    text = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
    indices = _parse_prune_indices(text, len(skills))
    if not indices:
        return 0

    # Safety valve: refuse to wipe the entire skillbook in a single prune call.
    # If the judge wants to delete everything, something is wrong with the model
    # output — let the human use `autoswarm skills clear` instead.
    if len(indices) >= len(skills):
        return 0

    return remove_skills_by_indices(indices, skills_path)


async def _reflect_one(
    client: httpx.AsyncClient,
    upstream: str,
    model: str,
    payload: dict,
    headers: dict[str, str] | None = None,
) -> Skill | None:
    convo = _format_conversation(payload)
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": REFLECTION_PROMPT.format(conversation=convo)}
        ],
        "stream": False,
        "temperature": 0,
    }
    r = await client.post(
        f"{upstream.rstrip('/')}/v1/chat/completions",
        json=body,
        headers=headers or {},
        timeout=120.0,
    )
    r.raise_for_status()
    data = r.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return _parse_lesson(text)


async def reflect(
    upstream: str,
    model: str,
    *,
    conversations_dir: Path | str = DEFAULT_DIR,
    skills_path: Path | str = DEFAULT_PATH,
    limit: int | None = None,
    api_key: str | None = None,
) -> dict:
    """Review unreviewed conversations and append novel skills.

    If `api_key` is provided (or `OPENAI_API_KEY` is set in the environment),
    it's forwarded as `Authorization: Bearer ...` — needed when reflecting
    against a hosted upstream like OpenAI. Local LLMs (Ollama, vLLM, LM
    Studio) don't need this.

    After the add pass, the same judge is asked which existing skills (if any)
    should be deleted. Flagged skills are removed silently from `skills_path`.

    Returns a summary dict: {"reviewed": N, "added": N, "skipped": N, "pruned": N}.
    """
    paths = get_unreviewed(conversations_dir)
    if limit is not None:
        paths = paths[-limit:]

    key = api_key or os.environ.get("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else None

    added = 0
    skipped = 0
    pruned = 0
    processed: list[Path] = []
    processed_payloads: list[dict] = []

    async with httpx.AsyncClient() as client:
        for path in paths:
            try:
                payload = json.loads(path.read_text())
            except Exception as exc:
                print(f"[reflector] {path.name}: unreadable ({exc})")
                continue

            try:
                skill = await _reflect_one(client, upstream, model, payload, headers)
            except Exception as exc:
                # Network/LLM error — leave unmarked so we retry next time.
                print(f"[reflector] {path.name}: reflection error ({exc})")
                continue

            if skill and add_skill(skill, skills_path):
                added += 1
            else:
                skipped += 1
            processed.append(path)
            processed_payloads.append(payload)

        if processed_payloads:
            try:
                pruned = await _prune_skills(
                    client,
                    upstream,
                    model,
                    processed_payloads,
                    skills_path,
                    headers,
                )
            except Exception as exc:
                # Pruning is best-effort. Never let a bad judge response stop
                # the reflection pass from completing.
                print(f"[reflector] prune error ({exc})")

    if processed:
        mark_reviewed(processed, conversations_dir)

    return {
        "reviewed": len(processed),
        "added": added,
        "skipped": skipped,
        "pruned": pruned,
    }
