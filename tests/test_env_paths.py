"""Tests for the D:-only environment constraint (Requirement 1).

These assert that the running interpreter and its package locations do not
resolve to the C: drive. They are meant to be run from the project-local
`.venv` on D:.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make scripts/verify_paths.py importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import verify_paths  # noqa: E402


def _drive(path: str) -> str:
    return os.path.splitdrive(os.path.abspath(path))[0].upper()


def test_no_c_drive_helper_detects_c_paths():
    assert verify_paths._on_c_drive(r"C:\\Python311\\python.exe") is True
    assert verify_paths._on_c_drive(r"D:\\projects\\metadata_VSL\\.venv") is False
    assert verify_paths._on_c_drive(None) is False
    assert verify_paths._on_c_drive("") is False


def test_collect_paths_returns_interpreter_and_package_locations():
    paths = verify_paths.collect_paths()
    assert "sys.executable" in paths
    assert "purelib" in paths
    assert "platlib" in paths


@pytest.mark.skipif(
    _drive(sys.prefix) != "D:",
    reason="Interpreter is not on D: (running outside the project .venv).",
)
def test_environment_paths_resolve_under_d():
    paths = verify_paths.collect_paths()
    offenders = {name: p for name, p in paths.items() if verify_paths._on_c_drive(p)}
    assert not offenders, f"Paths resolve to C: drive: {offenders}"
