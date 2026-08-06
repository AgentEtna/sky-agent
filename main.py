"""Entrypoint. Reads agent.yaml, starts whichever channels are enabled.

    python main.py
"""

import os
import sys

if sys.version_info < (3, 9):
    sys.exit(
        f"This agent needs Python 3.9+ (you have {sys.version.split()[0]}).\n"
        "Tip: create the venv with a newer interpreter, e.g.\n"
        "  python3.12 -m venv .venv && source .venv/bin/activate"
    )

import threading
import time

import yaml


def load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader so local dev needs zero extra tooling."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> None:
    load_dotenv()

    with open("agent.yaml") as f:
        config = yaml.safe_load(f)

    if not any(
        os.environ.get(k)
        for k in ("LLM-API-KEY", "LLM_API_KEY", "OPENAI_API_KEY")
    ):
        print(
            "[warn] No LLM-API-KEY set. Fine for local providers like Ollama; "
            "cloud providers (Anthropic, OpenAI, OpenRouter, ...) will reject "
            "requests until you set one."
        )

    channels = config.get("channels", {}) or {}
    started = []

    # telegram: auto (default) = on only if a token is set; true/false force it
    tg_setting = channels.get("telegram", "auto")
    has_token = bool(os.environ.get("TELEGRAM_BOT_TOKEN"))
    if tg_setting is True and not has_token:
        print(
            "[warn] telegram is set to true in agent.yaml but TELEGRAM_BOT_TOKEN "
            "is not set — skipping. Get a token from @BotFather."
        )
    elif (tg_setting is True or str(tg_setting).lower() == "auto") and has_token:
        from channels import telegram

        t = threading.Thread(target=telegram.start, args=(config,), daemon=True)
        t.start()
        started.append("telegram")

    if channels.get("http"):
        import uvicorn

        from channels.http_api import create_app

        started.append("http")
        port = int(os.environ.get("PORT", 8080))
        print(f"[agent] '{config.get('name', 'Agent')}' running: {', '.join(started)}")
        uvicorn.run(create_app(config), host="0.0.0.0", port=port)  # blocks
        return

    if not started:
        sys.exit("No channels running. Enable at least one in agent.yaml.")

    print(f"[agent] '{config.get('name', 'Agent')}' running: {', '.join(started)}")
    while True:  # keep the process alive for the telegram thread
        time.sleep(60)


if __name__ == "__main__":
    main()
