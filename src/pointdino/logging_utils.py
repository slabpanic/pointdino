"""Project-wide Loguru configuration.

The MMEngine runtime and a few third-party components still emit records via
the standard :mod:`logging` and :mod:`warnings` modules.  This module bridges
those records into Loguru so the project has one console/file format.
"""

from __future__ import annotations

import inspect
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

from loguru import logger


_configured = False
_log_files: set[str] = set()


class _InterceptHandler(logging.Handler):
    """Forward standard-library logging records to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: Any = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = inspect.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage())


def _redirect_runtime_loggers() -> None:
    """Replace handlers already installed by MMEngine/MMDetection."""
    for name, candidate in logging.root.manager.loggerDict.items():
        if not (name == "mmengine" or name.startswith("mmengine.") or
                name == "mmdet" or name.startswith("mmdet.")):
            continue
        if not isinstance(candidate, logging.Logger):
            continue
        for handler in list(candidate.handlers):
            candidate.removeHandler(handler)
        candidate.addHandler(_InterceptHandler())
        candidate.setLevel(logging.NOTSET)
        candidate.propagate = False


def _show_warning(message, category, filename, lineno, file=None, line=None):
    """Forward Python warnings to Loguru."""
    logger.warning(
        "{}:{}: {}: {}", filename, lineno, category.__name__, message)


def configure_logging(
    level: str | None = None,
    log_file: str | Path | None = None,
) -> None:
    """Configure Loguru once and optionally add a rotating file sink.

    Args:
        level: Log level. Defaults to ``INFO``.
        log_file: Optional file receiving the same formatted records.
    """

    global _configured

    level = (level or "INFO").upper()
    if not _configured:
        logger.remove()
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level:<8}</level> | {message}"
            ),
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )

        logging.basicConfig(
            handlers=[_InterceptHandler()],
            level=logging.NOTSET,
            force=True,
        )
        warnings.showwarning = _show_warning
        _configured = True

    if log_file is not None:
        path = Path(log_file).expanduser().resolve()
        key = str(path)
        if key not in _log_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                path,
                level=level,
                rotation="50 MB",
                retention=5,
                encoding="utf-8",
                enqueue=True,
                backtrace=False,
                diagnose=False,
            )
            _log_files.add(key)

    # MMEngine can create its own handlers while building a Runner/model.
    # Re-run this on every call so a second configure_logging() refreshes the
    # bridge after those handlers appear.
    _redirect_runtime_loggers()


__all__ = ["configure_logging", "logger"]
