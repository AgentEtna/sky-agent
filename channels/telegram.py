"""Telegram channel via long polling.

No webhooks, no public URL, no TLS setup — the bot pulls its own
messages, so it runs anywhere a container can run.

Get a token in 30 seconds: message @BotFather on Telegram, send
/newbot, follow the prompts, copy the token.
"""

import os
import time

import requests

from agent import memory
from agent.loop import respond

MAX_LEN = 4096  # Telegram's per-message limit


def _api(token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


def _send(token: str, chat_id, text: str) -> None:
    for i in range(0, len(text), MAX_LEN):
        requests.post(
            _api(token, "sendMessage"),
            json={"chat_id": chat_id, "text": text[i : i + MAX_LEN]},
            timeout=30,
        )


def start(config: dict) -> None:
    """Blocking long-poll loop. Run this in a thread (see main.py)."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    name = config.get("name", "your agent")
    # Lock the bot to specific chats: set TELEGRAM_ALLOWED_CHAT_IDS to a
    # comma-separated list of chat ids. Unset = anyone can talk to it.
    allowed = {
        c.strip()
        for c in os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",")
        if c.strip()
    }
    offset = None
    print("[telegram] polling started")

    while True:
        try:
            resp = requests.get(
                _api(token, "getUpdates"),
                params={"timeout": 50, **({"offset": offset} if offset else {})},
                timeout=60,
            )
            for update in resp.json().get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message") or {}
                text = (msg.get("text") or "").strip()
                chat = msg.get("chat", {}).get("id")
                if not text or chat is None:
                    continue

                if allowed and str(chat) not in allowed:
                    _send(
                        token,
                        chat,
                        f"This is a private bot. (Your chat id is {chat} — "
                        "if this is your bot, add that id to "
                        "TELEGRAM_ALLOWED_CHAT_IDS.)",
                    )
                    continue

                if text == "/start":
                    _send(token, chat, f"Hi! I'm {name}. Just talk to me. Send /reset to wipe my memory of this chat.")
                    continue
                if text == "/reset":
                    memory.clear(str(chat))
                    _send(token, chat, "Memory cleared. Fresh start.")
                    continue

                # show "typing…" while the model thinks
                requests.post(
                    _api(token, "sendChatAction"),
                    json={"chat_id": chat, "action": "typing"},
                    timeout=10,
                )
                try:
                    reply = respond(str(chat), text, config)
                except Exception as exc:  # keep the bot alive on model errors
                    reply = f"Something went wrong: {exc}"
                _send(token, chat, reply)

        except Exception as exc:  # network blip, Telegram hiccup, etc.
            print(f"[telegram] error: {exc} — retrying in 3s")
            time.sleep(3)
