#!/usr/bin/env python3

import logging
from logging.handlers import RotatingFileHandler

def setup_logging(level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    fh = RotatingFileHandler("sim.log", maxBytes=1_000_000, backupCount=2)
    fh.setFormatter(fmt)

    root.handlers.clear()
    root.addHandler(ch)
    root.addHandler(fh)
