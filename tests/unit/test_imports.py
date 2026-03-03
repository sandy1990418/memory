"""
Test that all architecture modules can be imported without errors.

This verifies the import graph is consistent and no circular imports exist.
Uses unittest.mock to stub missing optional dependencies so tests never skip.
"""

import importlib
import sys
import unittest
from unittest.mock import MagicMock


# Modules that should be importable without heavy deps
PURE_MODULES = [
    # Utils
    "openclaw_memory.utils",
    "openclaw_memory.utils.similarity",
    "openclaw_memory.utils.text",
    "openclaw_memory.utils.tokens",
    # Core types
    "openclaw_memory.core",
    "openclaw_memory.core.types",
    # Config
    "openclaw_memory.config",
    # Models
    "openclaw_memory.models",
    "openclaw_memory.models.session",
    "openclaw_memory.models.search",
    "openclaw_memory.models.memory",
    # Pipeline: ingest
    "openclaw_memory.pipeline",
    "openclaw_memory.pipeline.ingest",
    "openclaw_memory.pipeline.ingest.normalize",
    "openclaw_memory.pipeline.ingest.sensory",
    # Pipeline: retrieval (no DB imports at module level)
    "openclaw_memory.pipeline.retrieval",
    "openclaw_memory.pipeline.retrieval.hybrid",
    "openclaw_memory.pipeline.retrieval.ranking",
    "openclaw_memory.pipeline.retrieval.answer",
]

# Modules that require psycopg to import
DB_MODULES = [
    "openclaw_memory.db",
    "openclaw_memory.db.queries",
    "openclaw_memory.memory",
    "openclaw_memory.memory.working",
    "openclaw_memory.memory.episodic",
    "openclaw_memory.memory.semantic",
    "openclaw_memory.pipeline.ingest.conflict",
    "openclaw_memory.pipeline.ingest.extraction",
    "openclaw_memory.pipeline.retrieval.search",
    "openclaw_memory.consolidation",
    "openclaw_memory.consolidation.dedup",
]

# Modules that need psycopg_pool
POOL_MODULES = [
    "openclaw_memory.db.connection",
    "openclaw_memory.dependencies",
]

# Modules that need fastapi
FASTAPI_MODULES = [
    "openclaw_memory.routers",
    "openclaw_memory.routers.health",
    "openclaw_memory.routers.session",
    "openclaw_memory.routers.memory",
    "openclaw_memory.routers.search",
    "openclaw_memory.app",
]

# Optional packages that may not be installed — we mock them so import
# tests always run instead of being skipped.
_OPTIONAL_PACKAGES = [
    "psycopg",
    "psycopg.rows",
    "psycopg_pool",
    "fastapi",
    "fastapi.responses",
    "fastapi.routing",
    "starlette",
    "starlette.status",
    "uvicorn",
]


def _ensure_mock_packages():
    """Insert MagicMock stubs for any missing optional packages into sys.modules."""
    for pkg in _OPTIONAL_PACKAGES:
        if pkg not in sys.modules:
            sys.modules[pkg] = MagicMock()


def _clear_test_modules(module_names):
    """Remove previously imported test-target modules so they can be re-imported
    with fresh mocked dependencies."""
    for name in list(sys.modules):
        for target in module_names:
            if name == target or name.startswith(target + "."):
                del sys.modules[name]
                break


class TestModuleImports(unittest.TestCase):
    """Test that all modules can be imported (mocking optional deps if missing)."""

    def test_pure_python_modules(self):
        """Modules with no DB/web dependencies should import cleanly."""
        for mod_name in PURE_MODULES:
            with self.subTest(module=mod_name):
                try:
                    mod = importlib.import_module(mod_name)
                    self.assertIsNotNone(mod)
                except ImportError as e:
                    self.fail(f"Failed to import {mod_name}: {e}")

    def test_db_modules(self):
        """Modules requiring psycopg should import (mocked if not installed)."""
        _ensure_mock_packages()
        _clear_test_modules(DB_MODULES)
        for mod_name in DB_MODULES:
            with self.subTest(module=mod_name):
                try:
                    mod = importlib.import_module(mod_name)
                    self.assertIsNotNone(mod)
                except ImportError as e:
                    self.fail(f"Failed to import {mod_name}: {e}")

    def test_pool_modules(self):
        """Modules requiring psycopg_pool should import (mocked if not installed)."""
        _ensure_mock_packages()
        _clear_test_modules(POOL_MODULES)
        for mod_name in POOL_MODULES:
            with self.subTest(module=mod_name):
                try:
                    mod = importlib.import_module(mod_name)
                    self.assertIsNotNone(mod)
                except ImportError as e:
                    self.fail(f"Failed to import {mod_name}: {e}")

    def test_fastapi_modules(self):
        """Modules requiring fastapi should import (mocked if not installed)."""
        _ensure_mock_packages()
        _clear_test_modules(FASTAPI_MODULES)
        for mod_name in FASTAPI_MODULES:
            with self.subTest(module=mod_name):
                try:
                    mod = importlib.import_module(mod_name)
                    self.assertIsNotNone(mod)
                except ImportError as e:
                    self.fail(f"Failed to import {mod_name}: {e}")


if __name__ == "__main__":
    unittest.main()
