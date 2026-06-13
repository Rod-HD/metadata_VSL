"""Verify the Python interpreter and installed packages resolve under D:.

This enforces Requirement 1: nothing may live on the C: drive. It is run by
`setup_env.ps1` after installation and can also be run standalone:

    .venv\\Scripts\\python.exe scripts\\verify_paths.py

Exits non-zero (raising AssertionError) if any relevant path is on C:.
"""

from __future__ import annotations

import os
import site
import sys
import sysconfig


def _on_c_drive(path: str | None) -> bool:
    if not path:
        return False
    return os.path.splitdrive(os.path.abspath(path))[0].upper() == "C:"


def collect_paths() -> dict[str, str]:
    """Gather the interpreter and package-location paths to check."""
    paths: dict[str, str] = {
        "sys.executable": sys.executable,
        "sys.prefix": sys.prefix,
        "purelib": sysconfig.get_path("purelib"),
        "platlib": sysconfig.get_path("platlib"),
        "scripts": sysconfig.get_path("scripts"),
    }

    # pip cache location (should have been redirected to D: via PIP_CACHE_DIR).
    pip_cache = os.environ.get("PIP_CACHE_DIR")
    if pip_cache:
        paths["PIP_CACHE_DIR"] = pip_cache

    # site-packages directories.
    try:
        for i, sp in enumerate(site.getsitepackages()):
            paths[f"site-packages[{i}]"] = sp
    except AttributeError:
        # getsitepackages is unavailable in some venv configurations.
        pass

    return paths


def main() -> int:
    paths = collect_paths()
    offenders = {name: p for name, p in paths.items() if _on_c_drive(p)}

    print("Resolved environment paths:")
    for name, p in paths.items():
        marker = "  <-- ON C:!" if name in offenders else ""
        print(f"  {name:20s} = {p}{marker}")

    assert not offenders, (
        "The following paths resolve to the C: drive, violating the "
        f"no-C: constraint (Requirement 1): {offenders}"
    )

    print("\nOK: interpreter and packages all resolve under D: (no C: paths).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
