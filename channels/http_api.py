"""HTTP channel: talk to your agent with a POST request.

    curl -X POST https://your-app.up.railway.app/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "hello!"}'

Set HTTP_API_KEY in the environment to require
`Authorization: Bearer <key>` on /chat and /reset.
Add your own endpoints below — it's a normal FastAPI app.
"""

import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from agent import memory
from agent.loop import respond


class ChatRequest(BaseModel):
    message: str
    chat_id: str = "api"  # pass your own id to keep separate conversations


class ResetRequest(BaseModel):
    chat_id: str = "api"


def create_app(config: dict) -> FastAPI:
    app = FastAPI(title=config.get("name", "Agent"))
    required_key = os.environ.get("HTTP_API_KEY")

    def check_auth(authorization: Optional[str]) -> None:
        if required_key and authorization != f"Bearer {required_key}":
            raise HTTPException(status_code=401, detail="Bad or missing API key")

    @app.get("/")
    def health() -> dict:
        return {"status": "ok", "agent": config.get("name", "Agent")}

    @app.post("/chat")
    def chat(req: ChatRequest, authorization: Optional[str] = Header(default=None)) -> dict:
        check_auth(authorization)
        return {"reply": respond(req.chat_id, req.message, config)}

    @app.post("/reset")
    def reset(req: ResetRequest, authorization: Optional[str] = Header(default=None)) -> dict:
        check_auth(authorization)
        memory.clear(req.chat_id)
        return {"status": "cleared", "chat_id": req.chat_id}

    return app
