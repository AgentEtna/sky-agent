"""Local LLM auto-discovery.

Probes well-known OpenAI-compatible endpoints in parallel and extracts the
loaded model list from `/v1/models`. Used by `autoswarm start` (auto-detect
upstream + model when flags are omitted) and `autoswarm doctor` (diagnostics).
"""

from __future__ import annotations

import asyncio
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx

KNOWN_UPSTREAMS: list[tuple[str, str]] = [
    ("LM Studio", "http://localhost:1234"),
    ("Ollama", "http://localhost:11434"),
    ("vLLM", "http://localhost:8000"),
]

PROBE_TIMEOUT_SEC = 0.5

# Substrings (case-insensitive) that mark a model as non-chat. We skip these
# during auto-detect so e.g. `text-embedding-nomic-embed-text-v1.5` loaded in
# LM Studio alongside a chat model doesn't get picked as the default.
_NON_CHAT_MARKERS = ("embed", "rerank", "whisper", "tts")


def _is_chat_model(model_id: str) -> bool:
    mid = model_id.lower()
    return not any(marker in mid for marker in _NON_CHAT_MARKERS)


@dataclass
class ProbeResult:
    name: str
    url: str
    reachable: bool
    models: list[str]
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.reachable and bool(self.models)


async def _probe_one(client: httpx.AsyncClient, name: str, url: str) -> ProbeResult:
    try:
        r = await client.get(f"{url}/v1/models", timeout=PROBE_TIMEOUT_SEC)
    except httpx.ConnectError:
        return ProbeResult(name, url, False, [], "connection refused")
    except httpx.TimeoutException:
        return ProbeResult(name, url, False, [], "timeout")
    except Exception as exc:
        return ProbeResult(name, url, False, [], str(exc))

    if r.status_code != 200:
        return ProbeResult(name, url, False, [], f"HTTP {r.status_code}")
    try:
        data = r.json().get("data") or []
        all_ids = [m.get("id", "") for m in data if isinstance(m, dict) and m.get("id")]
    except Exception as exc:
        return ProbeResult(name, url, False, [], f"bad json: {exc}")
    chat_models = [mid for mid in all_ids if _is_chat_model(mid)]
    return ProbeResult(name, url, True, chat_models)


async def _probe_all_async(candidates: list[tuple[str, str]]) -> list[ProbeResult]:
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(
            *(_probe_one(client, n, u) for n, u in candidates)
        )


def probe_upstreams(
    candidates: list[tuple[str, str]] | None = None,
) -> list[ProbeResult]:
    """Probe all known upstreams in parallel. Total wall time ≈ PROBE_TIMEOUT_SEC."""
    return asyncio.run(_probe_all_async(candidates or KNOWN_UPSTREAMS))


def detect_upstream() -> ProbeResult | None:
    """Return the first reachable upstream (prefers ones with a model loaded)."""
    results = probe_upstreams()
    for r in results:
        if r.ready:
            return r
    for r in results:
        if r.reachable:
            return r
    return None


def detect_model(upstream: str, timeout_sec: float = 2.0) -> str | None:
    """Query `/v1/models` on `upstream`, return the first model id, or None."""

    async def _q() -> str | None:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            try:
                r = await client.get(f"{upstream.rstrip('/')}/v1/models")
            except Exception:
                return None
            if r.status_code != 200:
                return None
            try:
                data = r.json().get("data") or []
            except Exception:
                return None
            for m in data:
                mid = m.get("id") if isinstance(m, dict) else None
                if mid and _is_chat_model(mid):
                    return mid
            return None

    return asyncio.run(_q())


def find_lm_studio_app() -> Path | None:
    """Best-effort detection of LM Studio install location (macOS only for now)."""
    if platform.system() != "Darwin":
        return None
    for p in (
        Path("/Applications/LM Studio.app"),
        Path.home() / "Applications" / "LM Studio.app",
    ):
        if p.exists():
            return p
    return None


def find_lms_cli() -> Path | None:
    """Locate the `lms` CLI on PATH, or None."""
    p = shutil.which("lms")
    return Path(p) if p else None
