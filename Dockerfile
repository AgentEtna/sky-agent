FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Python deps only (cached layer; code copied below)
COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["python", "main.py"]
