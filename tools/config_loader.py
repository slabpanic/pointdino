"""Repository-only YAML configuration loading for PointDINO."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_config_path(filename: str) -> Path:
    """Return a config shipped in the repository checkout."""
    path = PROJECT_ROOT / "config" / filename
    if not path.is_file():
        raise FileNotFoundError(f"repository config does not exist: {path}")
    return path


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a YAML path from the cwd or this repository."""
    value = Path(path).expanduser()
    candidates = ([value] if value.is_absolute() else
                  [Path.cwd() / value, PROJECT_ROOT / value])
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved.suffix.lower() not in {".yaml", ".yml"}:
                raise ValueError(
                    f"PointDINO configs must be YAML files: {resolved}")
            return resolved
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"config does not exist; searched: {searched}")


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Deep-merge two YAML mappings with MMEngine-style ``_delete_``."""
    if override.get("_delete_") is True:
        return {
            key: deepcopy(value)
            for key, value in override.items()
            if key != "_delete_"
        }

    result = deepcopy(dict(base))
    for key, value in override.items():
        if key == "_delete_":
            continue
        if (key in result and isinstance(result[key], Mapping)
                and isinstance(value, Mapping)):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_yaml(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"circular YAML config inheritance: {chain}")

    with path.open("r", encoding="utf-8") as file:
        import yaml

        current = yaml.safe_load(file) or {}
    if not isinstance(current, Mapping):
        raise TypeError(f"YAML config root must be a mapping: {path}")

    base_files = current.pop("_base_", []) if isinstance(current, dict) else []
    if isinstance(base_files, (str, Path)):
        base_files = [base_files]

    merged: dict[str, Any] = {}
    next_stack = (*stack, path)
    for base_file in base_files:
        base_path = (path.parent / str(base_file)).resolve()
        merged = _merge(merged, _load_yaml(base_path, next_stack))
    return _merge(merged, current)


def load_config(path: str | Path):
    """Load a repository YAML config into an MMEngine ``Config``."""
    from mmengine.config import Config

    config_path = resolve_config_path(path)
    return Config(_load_yaml(config_path), filename=str(config_path))


__all__ = [
    "PROJECT_ROOT",
    "default_config_path",
    "load_config",
    "resolve_config_path",
]
