"""Runtime compatibility shims for supported Python versions."""

import importlib.util
import pkgutil


def patch_python314_compat() -> None:
    """Restore the legacy loader lookup used by older MMEngine releases."""
    if hasattr(pkgutil, "find_loader"):
        return

    def find_loader(name: str):
        try:
            return importlib.util.find_spec(name)
        except (ImportError, ModuleNotFoundError):
            return None

    pkgutil.find_loader = find_loader
