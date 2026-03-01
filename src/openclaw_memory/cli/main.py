"""
CLI entrypoint for openclaw-memory (ocmem).
Mirrors: src/cli/memory-cli.ts
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.text import Text

app = typer.Typer(
    name="ocmem",
    help="OpenClaw memory search and management",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_workspace(workspace: str | None) -> str:
    return workspace or os.environ.get("OPENCLAW_WORKSPACE") or os.getcwd()


def _resolve_db_path(workspace: str) -> str:
    from_env = os.environ.get("OPENCLAW_DB_PATH")
    if from_env:
        return from_env
    # When workspace is explicitly provided (common in tests and isolated runs),
    # keep the DB under that workspace to avoid cross-environment permission issues.
    if workspace:
        return os.path.join(workspace, ".openclaw", "memory", "main.sqlite")
    state_dir = os.environ.get(
        "OPENCLAW_STATE_DIR", os.path.join(os.path.expanduser("~"), ".openclaw")
    )
    return os.path.join(state_dir, "memory", "main.sqlite")


def _get_manager(workspace: str | None = None) -> Any:
    """Create a MemoryIndexManager from env config."""
    from ..manager import MemoryIndexManager

    ws = _resolve_workspace(workspace)
    db_path = _resolve_db_path(ws)
    return MemoryIndexManager.create(workspace_dir=ws, overrides={"db_path": db_path})


def _today_date_str(date_override: str | None) -> str:
    if date_override:
        return date_override.strip()
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    max_results: int | None = typer.Option(None, "--max-results", "-n", help="Maximum results"),
    min_score: float | None = typer.Option(None, "--min-score", help="Minimum score threshold"),
    workspace: str | None = typer.Option(
        None, "--workspace", "-w", help="Workspace directory", envvar="OPENCLAW_WORKSPACE"
    ),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Search the memory index."""
    try:
        mgr = _get_manager(workspace)
    except Exception as exc:
        err_console.print(f"[red]Failed to open memory index:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        with mgr:
            # Trigger sync if dirty
            if mgr._dirty:
                try:
                    mgr.sync(reason="search")
                except Exception:
                    pass  # search with stale index is better than crashing

            results = mgr.search(
                query,
                max_results=max_results,
                min_score=min_score,
            )

        if output_json:
            console.print_json(
                json.dumps(
                    [
                        {
                            "path": r.path,
                            "start_line": r.start_line,
                            "end_line": r.end_line,
                            "score": r.score,
                            "snippet": r.snippet,
                            "source": r.source,
                        }
                        for r in results
                    ]
                )
            )
            return

        if not results:
            console.print("[dim]No matches.[/dim]")
            return

        for r in results:
            score_text = Text(f"{r.score:.3f}", style="green")
            loc_text = Text(f"{r.path}:{r.start_line}-{r.end_line}", style="cyan")
            console.print(score_text, loc_text)
            console.print(Text(r.snippet, style="dim"))
            console.print()

    except Exception as exc:
        err_console.print(f"[red]Search failed:[/red] {exc}")
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


@app.command()
def get(
    path: str = typer.Argument(
        ..., help="Relative path to memory file (e.g. MEMORY.md or memory/2024-01-01.md)"
    ),
    from_line: int | None = typer.Option(None, "--from", help="Start line (1-indexed)"),
    lines: int | None = typer.Option(None, "--lines", help="Number of lines to read"),
    workspace: str | None = typer.Option(
        None, "--workspace", "-w", help="Workspace directory", envvar="OPENCLAW_WORKSPACE"
    ),
) -> None:
    """Read a memory file (or slice of it)."""
    try:
        mgr = _get_manager(workspace)
    except Exception as exc:
        err_console.print(f"[red]Failed to open memory index:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        with mgr:
            result = mgr.read_file(path, from_line=from_line, lines=lines)
        console.print(result["text"])
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        err_console.print(f"[red]Failed to read file:[/red] {exc}")
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# append-daily
# ---------------------------------------------------------------------------


@app.command(name="append-daily")
def append_daily(
    text: str = typer.Argument(..., help="Text to append to today's daily memory file"),
    workspace: str | None = typer.Option(
        None, "--workspace", "-w", help="Workspace directory", envvar="OPENCLAW_WORKSPACE"
    ),
    date: str | None = typer.Option(
        None, "--date", help="Date override (YYYY-MM-DD, default: today UTC)"
    ),
) -> None:
    """
    Append text to memory/YYYY-MM-DD.md (creates the file if needed).

    Equivalent to the OpenClaw auto-memory flush that writes daily notes.
    """
    ws = _resolve_workspace(workspace)
    date_str = _today_date_str(date)

    # Validate date format
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        err_console.print(f"[red]Invalid date format:[/red] {date_str!r} (expected YYYY-MM-DD)")
        raise typer.Exit(1)

    memory_dir = Path(ws) / "memory"
    try:
        memory_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        err_console.print(f"[red]Cannot create memory directory:[/red] {exc}")
        raise typer.Exit(1) from exc

    daily_file = memory_dir / f"{date_str}.md"

    try:
        if daily_file.exists():
            existing = daily_file.read_text(encoding="utf-8")
            separator = "\n\n" if existing.rstrip() else ""
            daily_file.write_text(existing.rstrip() + separator + text + "\n", encoding="utf-8")
        else:
            daily_file.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        err_console.print(f"[red]Failed to write daily file:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[green]Appended to[/green] [cyan]{daily_file}[/cyan]")


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


@app.command()
def index(
    force: bool = typer.Option(False, "--force", help="Force full reindex"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    workspace: str | None = typer.Option(
        None, "--workspace", "-w", help="Workspace directory", envvar="OPENCLAW_WORKSPACE"
    ),
) -> None:
    """Reindex memory files."""
    try:
        mgr = _get_manager(workspace)
    except Exception as exc:
        err_console.print(f"[red]Failed to open memory index:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        with mgr:
            if verbose:
                st = mgr.status()
                console.print(
                    f"[bold]Memory Index[/bold] [dim](workspace: {st.workspace_dir})[/dim]"
                )
                console.print(
                    f"[dim]Provider:[/dim] [cyan]{st.provider}[/cyan]  "
                    f"[dim]Model:[/dim] [cyan]{st.model or 'none'}[/cyan]"
                )
                console.print()

            completed_ref = [0]
            total_ref = [0]

            def on_progress(completed: int, total: int, label: str) -> None:
                completed_ref[0] = completed
                total_ref[0] = total
                if verbose:
                    console.print(f"  [dim]{label}[/dim] ({completed}/{total})")

            mgr.sync(reason="cli", force=force, progress=on_progress)

        console.print("[green]Memory index updated.[/green]")

    except Exception as exc:
        err_console.print(f"[red]Index failed:[/red] {exc}")
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status(
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
    deep: bool = typer.Option(False, "--deep", help="Probe embedding availability"),
    workspace: str | None = typer.Option(
        None, "--workspace", "-w", help="Workspace directory", envvar="OPENCLAW_WORKSPACE"
    ),
) -> None:
    """Show memory index status."""
    try:
        mgr = _get_manager(workspace)
    except Exception as exc:
        err_console.print(f"[red]Failed to open memory index:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        with mgr:
            st = mgr.status()
            embedding_probe = None
            if deep:
                embedding_probe = mgr.probe_embedding_availability()

        if output_json:
            data: dict[str, Any] = {
                "backend": st.backend,
                "provider": st.provider,
                "model": st.model,
                "requested_provider": st.requested_provider,
                "files": st.files,
                "chunks": st.chunks,
                "dirty": st.dirty,
                "workspace_dir": st.workspace_dir,
                "db_path": st.db_path,
                "sources": st.sources,
                "fts": st.fts,
                "cache": st.cache,
                "fallback": st.fallback,
                "custom": st.custom,
            }
            if embedding_probe:
                data["embedding_probe"] = {"ok": embedding_probe.ok, "error": embedding_probe.error}
            console.print_json(json.dumps(data))
            return

        # Human-readable output
        console.print("[bold]Memory Search[/bold]")
        console.print(f"  [dim]Provider:[/dim]  [cyan]{st.provider}[/cyan]")
        if st.model:
            console.print(f"  [dim]Model:[/dim]     [cyan]{st.model}[/cyan]")
        console.print(
            f"  [dim]Indexed:[/dim]   [green]{st.files} files · {st.chunks} chunks[/green]"
        )
        console.print(
            f"  [dim]Dirty:[/dim]     {'[yellow]yes[/yellow]' if st.dirty else '[dim]no[/dim]'}"
        )
        if st.workspace_dir:
            console.print(f"  [dim]Workspace:[/dim] [cyan]{st.workspace_dir}[/cyan]")
        if st.db_path:
            console.print(f"  [dim]Store:[/dim]     [cyan]{st.db_path}[/cyan]")
        if st.sources:
            console.print(f"  [dim]Sources:[/dim]   [cyan]{', '.join(st.sources)}[/cyan]")

        if st.fts:
            fts_state = (
                "ready"
                if st.fts.get("available")
                else ("disabled" if not st.fts.get("enabled") else "unavailable")
            )
            fts_style = (
                "green"
                if fts_state == "ready"
                else ("dim" if fts_state == "disabled" else "yellow")
            )
            console.print(f"  [dim]FTS:[/dim]       [{fts_style}]{fts_state}[/{fts_style}]")

        if embedding_probe:
            ep_state = "ready" if embedding_probe.ok else "unavailable"
            ep_style = "green" if embedding_probe.ok else "yellow"
            console.print(f"  [dim]Embeddings:[/dim] [{ep_style}]{ep_state}[/{ep_style}]")
            if embedding_probe.error:
                console.print(f"  [dim]Emb error:[/dim]  [yellow]{embedding_probe.error}[/yellow]")

        if st.fallback and st.fallback.get("from"):
            console.print(f"  [dim]Fallback:[/dim]  [yellow]{st.fallback['from']}[/yellow]")

        if st.cache:
            cache_state = "enabled" if st.cache.get("enabled") else "disabled"
            cache_style = "green" if st.cache.get("enabled") else "dim"
            suffix = (
                f" ({st.cache['entries']} entries)" if st.cache.get("entries") is not None else ""
            )
            console.print(
                f"  [dim]Cache:[/dim]     [{cache_style}]{cache_state}{suffix}[/{cache_style}]"
            )

    except Exception as exc:
        err_console.print(f"[red]Status failed:[/red] {exc}")
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
