"""OpenAI-compatible proxy server with skill injection.

The proxy:
  - exposes `/v1/chat/completions` and `/v1/models`
  - on every chat request, loads `skills.yaml` and merges learned strategies
    into the system message
  - forwards to an upstream OpenAI-compatible endpoint (Ollama, vLLM, LM Studio)
  - logs the (original) request + reconstructed response to `conversations/`
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .logger import DEFAULT_DIR as DEFAULT_CONVERSATIONS_DIR, log_conversation
from .skills import DEFAULT_PATH as DEFAULT_SKILLS_PATH, inject_skills, load_skills

# Hop-by-hop headers (RFC 7230 §6.1) plus a few we must always recompute or drop.
_DROP_HEADERS = {
    "host", "content-length", "content-encoding",
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "expect",
    "accept-encoding",
}


def _forward_headers(request: Request) -> dict[str, str]:
    """Pass through the client's headers (notably Authorization) to upstream."""
    return {
        k: v for k, v in request.headers.items()
        if k.lower() not in _DROP_HEADERS
    }


def create_app(
    upstream: str,
    default_model: str | None = None,
    skills_path: Path | str = DEFAULT_SKILLS_PATH,
    conversations_dir: Path | str = DEFAULT_CONVERSATIONS_DIR,
) -> FastAPI:
    upstream = upstream.rstrip("/")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(timeout=None)
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="AutoSwarm online proxy", lifespan=lifespan)
    app.state.upstream = upstream
    app.state.default_model = default_model
    app.state.skills_path = Path(skills_path)
    app.state.conversations_dir = Path(conversations_dir)

    @app.get("/v1/models")
    async def models():
        try:
            r = await app.state.client.get(f"{upstream}/v1/models")
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if default_model and not body.get("model"):
            body["model"] = default_model

        original_messages = body.get("messages", [])
        skills = load_skills(app.state.skills_path)
        body["messages"] = inject_skills(original_messages, skills)

        url = f"{upstream}/v1/chat/completions"
        headers = _forward_headers(request)

        if not body.get("stream"):
            r = await app.state.client.post(url, json=body, headers=headers)
            try:
                data = r.json()
            except Exception:
                return JSONResponse({"error": r.text}, status_code=r.status_code)
            _safe_log(app, original_messages, data)
            return JSONResponse(data, status_code=r.status_code)

        return StreamingResponse(
            _stream_and_log(app, url, body, original_messages, headers),
            media_type="text/event-stream",
        )

    return app


def _safe_log(app: FastAPI, messages: list[dict], response: dict) -> None:
    try:
        log_conversation(messages, response, app.state.conversations_dir)
    except Exception as exc:
        print(f"[logger] failed: {exc}")


async def _stream_and_log(
    app: FastAPI,
    url: str,
    body: dict,
    original_messages: list[dict],
    headers: dict[str, str],
):
    """Proxy an SSE stream while accumulating delta content for the log."""
    pieces: list[str] = []
    buffer = b""

    async with app.state.client.stream("POST", url, json=body, headers=headers) as r:
        async for raw in r.aiter_bytes():
            yield raw
            buffer += raw
            # SSE events are newline-delimited; lines can split across chunks.
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    pieces.append(piece)

    reconstructed = {
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(pieces)},
                "finish_reason": "stop",
            }
        ],
        "streamed": True,
    }
    _safe_log(app, original_messages, reconstructed)


def run(
    upstream: str,
    model: str | None,
    host: str,
    port: int,
    skills_path: Path | str = DEFAULT_SKILLS_PATH,
    conversations_dir: Path | str = DEFAULT_CONVERSATIONS_DIR,
) -> None:
    import uvicorn

    app = create_app(
        upstream,
        default_model=model,
        skills_path=skills_path,
        conversations_dir=conversations_dir,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
