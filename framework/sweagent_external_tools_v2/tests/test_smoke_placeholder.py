from __future__ import annotations

import importlib.util

import pytest


def test_smoke_dependency_gate() -> None:
    """Keep pytest exit code stable when smoke dependencies are unavailable."""
    missing = [name for name in ("swerex", "sweagent") if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f"Optional smoke dependencies missing: {', '.join(missing)}")
    assert True
