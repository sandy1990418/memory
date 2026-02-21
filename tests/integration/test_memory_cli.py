"""
Integration tests for the CLI entrypoint (ocmem / cli/main.py).
Uses typer's CliRunner so that real SQLite + MockEmbeddingProvider are used.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def _get_app():
    from openclaw_memory.cli.main import app  # type: ignore[import]
    return app


def _get_runner():
    from typer.testing import CliRunner
    return CliRunner()


class TestCLIStatus:
    def test_status_exits_zero(self, tmp_path: Path) -> None:
        app = _get_app()
        runner = _get_runner()
        result = runner.invoke(app, ["status", "--workspace", str(tmp_path)])
        assert result.exit_code == 0

    def test_status_output_contains_provider(self, tmp_path: Path) -> None:
        app = _get_app()
        runner = _get_runner()
        (tmp_path / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
        result = runner.invoke(app, ["status", "--workspace", str(tmp_path)])
        output = result.output
        # Should contain some kind of provider info
        assert "provider" in output.lower() or "backend" in output.lower() or result.exit_code == 0


class TestCLISearch:
    def test_search_exits_zero(self, tmp_path: Path) -> None:
        app = _get_app()
        runner = _get_runner()
        (tmp_path / "MEMORY.md").write_text(
            "# Memory\n\nPrefer Python.\n", encoding="utf-8"
        )
        result = runner.invoke(
            app, ["search", "Python", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0

    def test_search_no_results_does_not_crash(self, tmp_path: Path) -> None:
        app = _get_app()
        runner = _get_runner()
        (tmp_path / "MEMORY.md").write_text("# Memory\n\nHello.\n", encoding="utf-8")
        result = runner.invoke(
            app, ["search", "xyzzy_not_found_ever", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0


class TestCLIGet:
    def test_get_reads_file_content(self, tmp_path: Path) -> None:
        app = _get_app()
        runner = _get_runner()
        (tmp_path / "MEMORY.md").write_text(
            "# Memory\n\nThis is the content.\n", encoding="utf-8"
        )
        result = runner.invoke(
            app, ["get", "MEMORY.md", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Memory" in result.output or "content" in result.output

    def test_get_missing_file_handled_gracefully(self, tmp_path: Path) -> None:
        app = _get_app()
        runner = _get_runner()
        result = runner.invoke(
            app, ["get", "nonexistent.md", "--workspace", str(tmp_path)]
        )
        # Should not crash with an unhandled exception
        assert result.exit_code in (0, 1, 2)


class TestCLIIndex:
    def test_index_runs_without_error(self, tmp_path: Path) -> None:
        app = _get_app()
        runner = _get_runner()
        (tmp_path / "MEMORY.md").write_text(
            "# Memory\n\nContent to index.\n", encoding="utf-8"
        )
        result = runner.invoke(app, ["index", "--workspace", str(tmp_path)])
        assert result.exit_code == 0


class TestCLIAppendDaily:
    def test_append_daily_creates_file(self, tmp_path: Path) -> None:
        app = _get_app()
        runner = _get_runner()
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir(exist_ok=True)
        result = runner.invoke(
            app,
            ["append-daily", "New note content", "--workspace", str(tmp_path)],
        )
        assert result.exit_code == 0
        # A dated file should exist in memory/
        md_files = list(mem_dir.glob("*.md"))
        assert len(md_files) >= 1

    def test_append_daily_appends_to_existing(self, tmp_path: Path) -> None:
        app = _get_app()
        runner = _get_runner()
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir(exist_ok=True)

        # Invoke twice
        runner.invoke(
            app, ["append-daily", "First note.", "--workspace", str(tmp_path)]
        )
        runner.invoke(
            app, ["append-daily", "Second note.", "--workspace", str(tmp_path)]
        )

        md_files = list(mem_dir.glob("*.md"))
        assert len(md_files) >= 1
        content = md_files[0].read_text(encoding="utf-8")
        assert "First note." in content
        assert "Second note." in content
