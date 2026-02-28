#!/usr/bin/env python3
"""
config.py
=========
Application-wide configuration constants.

Values can be overridden via environment variables (see :mod:`main`).
This module is a thin, import-safe leaf — it never imports from
other project packages.
"""

# ── Simulation defaults ──────────────────────────────────────────────────────
DEFAULT_VEHICLE_COUNT: int = 6
DEFAULT_TICK_RATE_HZ: float = 20.0
DEFAULT_PRIORITY_AXIS: str = "EW"

# ── V2X bus defaults ─────────────────────────────────────────────────────────
DEFAULT_DROP_RATE: float = 0.0
DEFAULT_LATENCY_MS: int = 0

# ── UI defaults ──────────────────────────────────────────────────────────────
WINDOW_WIDTH: int = 1000
WINDOW_HEIGHT: int = 700
TARGET_FPS: int = 60

# ── ML model path (relative to project root) ─────────────────────────────────
ML_MODEL_REL_PATH: str = "ml/generated/traffic_model.pkl"
