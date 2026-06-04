"""
Platform and path helpers shared by Windows, macOS, and Linux hosts.

Keep OS detection in this module so business code does not grow scattered
``sys.platform`` or host-specific path branches.
"""

from __future__ import annotations

import os
import platform as _platform
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


def system_name() -> str:
    """Return the normalized OS name reported by Python."""
    return _platform.system()


def machine_name() -> str:
    """Return the normalized CPU architecture name."""
    return _platform.machine()


def is_macos() -> bool:
    return system_name() == "Darwin"


def is_linux() -> bool:
    return system_name() == "Linux"


def is_windows() -> bool:
    return system_name() == "Windows"


def project_root(start: Optional[Path] = None) -> Path:
    """Resolve the repository root without depending on the current cwd."""
    env_root = os.environ.get("QLIB_PROJECT_ROOT")
    if env_root:
        return Path(os.path.expandvars(os.path.expanduser(env_root))).resolve()

    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "pytest.ini").exists():
            return candidate
    return Path.cwd().resolve()


def resolve_path(value: Any, base: Optional[Path] = None) -> Path:
    """Expand env/user markers and resolve relative paths from project root."""
    raw = "" if value is None else str(value)
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    if expanded.is_absolute():
        return expanded
    return (base or project_root()) / expanded


def temp_dir(*parts: str) -> Path:
    """Return a cross-platform temp directory, creating it if missing.

    Centralizes scattered ``/tmp`` hardcoding so research scripts also work
    on Windows (uses the OS temp dir on every platform).
    """
    path = Path(tempfile.gettempdir()).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_cpu_count(default: int = 2, max_workers: Optional[int] = None) -> int:
    """Return a bounded CPU count that behaves the same on macOS/Linux."""
    count = os.cpu_count() or int(default)
    if max_workers is not None:
        count = min(count, int(max_workers))
    return max(int(count), 1)


def runtime_profile() -> Dict[str, Any]:
    """Small serializable runtime profile for diagnostics/logging."""
    return {
        "system": system_name(),
        "machine": machine_name(),
        "project_root": str(project_root()),
        "cpu_count": safe_cpu_count(),
    }
