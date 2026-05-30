#!/usr/bin/env python
"""Test runner for Agent-mem MVP."""

import sys
from pathlib import Path

import pytest


def run_tests() -> int:
    """Run core pytest suite and return pytest exit code."""
    root = Path(__file__).resolve().parent
    return pytest.main(["-q", str(root / "agent_mem" / "tests")])


if __name__ == "__main__":
    print("Running Agent-mem MVP tests...")
    raise SystemExit(run_tests())
