"""Pytest configuration: make the ``src`` layout importable.

Adds ``src/`` to ``sys.path`` so tests can ``import qipedc2vsl400`` without an
editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
