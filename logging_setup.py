#!/usr/bin/env python3
"""
logging_setup.py
================
Configures the root logger with a console handler and a rotating file
handler (``hexa.log``, 1 MB, 2 backups).

Call :func:`setup_logging` once at startup before any other ``import``
triggers ``logging.getLogger()``.
"""

import logging
from logging.handlers import RotatingFileHandler


def setup_logging(level: int = logging.INFO) -> None:
    """Apply a unified log format to both console and file output.

    Parameters
    ----------
    level : int
        Minimum severity level (e.g. ``logging.DEBUG``, ``logging.INFO``).
    """
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    fh = RotatingFileHandler("hexa.log", maxBytes=1_000_000, backupCount=2)
    fh.setFormatter(fmt)

    root.handlers.clear()
    root.addHandler(ch)
    root.addHandler(fh)

    # ── Dedicated debug file for the world collision guard ────────────
    world_logger = logging.getLogger("world")
    world_logger.setLevel(logging.DEBUG)
    dfh = RotatingFileHandler(
        "world_debug.log", maxBytes=5_000_000, backupCount=2
    )
    dfh.setLevel(logging.DEBUG)
    dfh.setFormatter(fmt)
    world_logger.addHandler(dfh)
