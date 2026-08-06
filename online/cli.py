"""AutoSwarm CLI — `autoswarm` script entry point.

This module only handles argument parsing and dispatches to `online.*`.
Benchmark-mode commands continue to run via Harbor's CLI directly.
"""

from __future__ import annotations

import asyncio
import platform as _platform
import sys
from pathlib import Path

import click


@click.group()
@click.version_option(package_name="autoswarm")
def main() -> None:
    """AutoSwarm — self-improving local LLM proxy."""


def _resolve_upstream_and_model(
    upstream: str | None,
    model: str | None,
) -> tuple[str, str]:
    """Fill in `upstream` and `model` via auto-detect when either is omitted.

    Exits the process with a helpful message if detection fails.
    """
    from .discovery import detect_model, detect_upstream

    if upstream is None:
        click.echo("Auto-detecting upstream...", err=True)
        result = detect_upstream()
        if result is None:
            click.echo(
                "✗ No local LLM detected on common ports (1234, 11434, 8000).",
                err=True,
            )
            click.echo("  Run `autoswarm doctor` for diagnostics.", err=True)
            sys.exit(1)
        upstream = result.url
        click.echo(f"  ✓ detected {result.name} at {result.url}", err=True)
        if model is None and result.models:
            model = result.models[0]

    if model is None:
        click.echo(f"Auto-detecting model at {upstream}...", err=True)
        model = detect_model(upstream)
        if model is None:
            click.echo(
                "✗ Upstream is reachable but no models are loaded.",
                err=True,
            )
            click.echo(
                "  Load a model in your local LLM client (e.g. LM Studio), then re-run.",
                err=True,
            )
            sys.exit(1)

    click.echo(f"  ✓ using model: {model}", err=True)
    return upstream, model


# --------------------------------------------------------------------------- #
# autoswarm start
# --------------------------------------------------------------------------- #
@main.command()
@click.option(
    "--upstream",
    default=None,
    help="Upstream OpenAI-compatible LLM endpoint. "
    "Auto-detected if omitted (LM Studio :1234 / Ollama :11434 / vLLM :8000).",
)
@click.option(
    "--model",
    default=None,
    help="Default model name (used when the client omits 'model'). "
    "Auto-detected from /v1/models if omitted.",
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8080, show_default=True, type=int)
@click.option(
    "--skills",
    "skills_path",
    default="skills.yaml",
    show_default=True,
    type=click.Path(),
    help="Path to the skills YAML file",
)
@click.option(
    "--conversations",
    "conversations_dir",
    default="conversations",
    show_default=True,
    type=click.Path(),
    help="Directory for conversation logs",
)
def start(
    upstream: str | None,
    model: str | None,
    host: str,
    port: int,
    skills_path: str,
    conversations_dir: str,
) -> None:
    """Start the online proxy server."""
    from .server import run

    upstream, model = _resolve_upstream_and_model(upstream, model)

    run(
        upstream=upstream,
        model=model,
        host=host,
        port=port,
        skills_path=Path(skills_path),
        conversations_dir=Path(conversations_dir),
    )


# --------------------------------------------------------------------------- #
# autoswarm reflect
# --------------------------------------------------------------------------- #
@main.command()
@click.option(
    "--upstream",
    default=None,
    help="Upstream OpenAI-compatible LLM endpoint. Auto-detected if omitted.",
)
@click.option(
    "--model",
    default=None,
    help="Model used for reflection. Auto-detected from /v1/models if omitted.",
)
@click.option(
    "--skills",
    "skills_path",
    default="skills.yaml",
    show_default=True,
    type=click.Path(),
)
@click.option(
    "--conversations",
    "conversations_dir",
    default="conversations",
    show_default=True,
    type=click.Path(),
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Only review the N most recent unreviewed conversations",
)
@click.option(
    "--api-key",
    default=None,
    envvar="OPENAI_API_KEY",
    help="Bearer token forwarded to upstream (defaults to $OPENAI_API_KEY); "
    "leave unset for local LLMs that don't require auth",
)
def reflect(
    upstream: str | None,
    model: str | None,
    skills_path: str,
    conversations_dir: str,
    limit: int | None,
    api_key: str | None,
) -> None:
    """Review unreviewed conversations and extract new strategies."""
    from .reflector import reflect as run_reflect

    upstream, model = _resolve_upstream_and_model(upstream, model)

    summary = asyncio.run(
        run_reflect(
            upstream=upstream,
            model=model,
            skills_path=Path(skills_path),
            conversations_dir=Path(conversations_dir),
            limit=limit,
            api_key=api_key,
        )
    )
    click.echo(
        f"reviewed={summary['reviewed']} "
        f"added={summary['added']} "
        f"skipped={summary['skipped']} "
        f"pruned={summary['pruned']}"
    )


# --------------------------------------------------------------------------- #
# autoswarm doctor
# --------------------------------------------------------------------------- #
@main.command()
def doctor() -> None:
    """Diagnose local LLM availability and print copy-paste fixes."""
    from .discovery import (
        find_lm_studio_app,
        find_lms_cli,
        probe_upstreams,
    )

    # Header.
    click.echo(f"Python:    {_platform.python_version()} on {_platform.system()}")
    try:
        from importlib.metadata import version

        click.echo(f"autoswarm: {version('autoswarm')}")
    except Exception:
        click.echo("autoswarm: (unknown — running from source?)")
    click.echo("")

    # Probe known upstreams.
    click.echo("Probing local LLM servers...")
    results = probe_upstreams()
    ready = None
    reachable_no_model = None
    for r in results:
        if r.ready:
            models_str = ", ".join(r.models)
            click.echo(f"  ✓ {r.name:<10} {r.url}  →  models: {models_str}")
            ready = ready or r
        elif r.reachable:
            click.echo(f"  ! {r.name:<10} {r.url}  →  reachable but no model loaded")
            reachable_no_model = reachable_no_model or r
        else:
            click.echo(f"  ✗ {r.name:<10} {r.url}  →  {r.error}")
    click.echo("")

    # Happy path.
    if ready:
        click.echo("✓ Ready to go. Run:")
        click.echo(f"    autoswarm start   # auto-uses {ready.name} + {ready.models[0]}")
        sys.exit(0)

    # Server reachable but no model.
    if reachable_no_model:
        click.echo(
            f"! {reachable_no_model.name} is running at {reachable_no_model.url} "
            "but no model is loaded."
        )
        click.echo("  Load a model in the app, then run: autoswarm start")
        sys.exit(1)

    # Nothing reachable — actionable suggestions.
    lms_app = find_lm_studio_app()
    lms_cli = find_lms_cli()

    if lms_app:
        click.echo(f"LM Studio is installed at {lms_app} but its server isn't running.")
        click.echo("Fix one of:")
        click.echo("  • In the app: open Developer tab → flip Status toggle to Running")
        if lms_cli:
            click.echo(f"  • In terminal (lms CLI found at {lms_cli}):")
            click.echo("      lms server start")
            click.echo("      lms ls                  # see downloaded models")
            click.echo("      lms load <model-id>     # load one")
        else:
            click.echo(
                "  • Install the `lms` CLI: open LM Studio once and accept its CLI prompt"
            )
    else:
        click.echo("No supported local LLM detected. Install one of:")
        click.echo("  • LM Studio:  https://lmstudio.ai/")
        click.echo("  • Ollama:     https://ollama.com/  (auto-runs on install)")
        click.echo("  • vLLM:       https://docs.vllm.ai/  (advanced)")

    click.echo("")
    click.echo("Then re-run: autoswarm doctor")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# autoswarm skills [list|clear]
# --------------------------------------------------------------------------- #
@main.group()
def skills() -> None:
    """Inspect or reset the skillbook."""


@skills.command("list")
@click.option(
    "--path",
    "skills_path",
    default="skills.yaml",
    show_default=True,
    type=click.Path(),
)
def skills_list(skills_path: str) -> None:
    """List learned strategies."""
    from .skills import load_skills

    items = load_skills(Path(skills_path))
    if not items:
        click.echo("(no skills yet)")
        return
    for i, s in enumerate(items, 1):
        click.echo(f"{i}. {s.trigger}: {s.strategy}")


@skills.command("clear")
@click.option(
    "--path",
    "skills_path",
    default="skills.yaml",
    show_default=True,
    type=click.Path(),
)
@click.confirmation_option(prompt="Delete all learned skills?")
def skills_clear(skills_path: str) -> None:
    """Reset the skillbook (deletes skills.yaml)."""
    p = Path(skills_path)
    if p.exists():
        p.unlink()
        click.echo(f"removed {p}")
    else:
        click.echo("(nothing to remove)")


if __name__ == "__main__":
    main()
