#!/usr/bin/env python3
"""Static / utility helpers used across the UI layer."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import pygame

from .types import ColorRGB


class ViewHelpers:
    """Mixin with generic data-access, parsing, and font utilities."""

    # -------------------------------------------------------------- #
    #  Generic data access                                             #
    # -------------------------------------------------------------- #
    @staticmethod
    def _get(obj: Any, *keys: str, default: Any = None) -> Any:
        if isinstance(obj, Mapping):
            for key in keys:
                if key in obj:
                    return obj[key]
        for key in keys:
            if hasattr(obj, key):
                return getattr(obj, key)
        return default

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    # -------------------------------------------------------------- #
    #  Color parsing                                                   #
    # -------------------------------------------------------------- #
    def _parse_color(self, raw: Any) -> Optional[ColorRGB]:
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("#") and len(text) == 7:
                try:
                    return (int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16))
                except ValueError:
                    return None
        if isinstance(raw, Sequence) and len(raw) >= 3:
            try:
                r = max(0, min(255, int(raw[0])))
                g = max(0, min(255, int(raw[1])))
                b = max(0, min(255, int(raw[2])))
                return (r, g, b)
            except (TypeError, ValueError):
                return None
        return None

    # -------------------------------------------------------------- #
    #  Font loading                                                    #
    # -------------------------------------------------------------- #
    @staticmethod
    def _load_font(size: int, bold: bool = False) -> pygame.font.Font:
        for name in ("JetBrains Mono", "Consolas", "Menlo", "DejaVu Sans Mono"):
            try:
                return pygame.font.SysFont(name, size, bold=bold)
            except Exception:
                continue
        return pygame.font.Font(None, size)

    # -------------------------------------------------------------- #
    #  Decision helpers                                                #
    # -------------------------------------------------------------- #
    def _normalize_decision(self, decision: Any) -> str:
        if decision is None:
            return "none"
        if isinstance(decision, Mapping):
            dec = str(decision.get("decision", "none")).strip().lower()
        else:
            dec = str(decision).strip().lower()
        if dec in {"go", "green", "true", "1", "proceed"}:
            return "go"
        if dec in {"stop", "red", "false", "0", "wait", "yield", "slow", "slow_down", "slowdown"}:
            return "stop"
        return "none"

    @staticmethod
    def _extract_confidence(raw_decision: Any, which: str) -> Optional[float]:
        """Pull a float confidence from a dict like ``{confidence_go: 0.8}``."""
        if not isinstance(raw_decision, Mapping):
            return None
        key = f"confidence_{which}"
        val = raw_decision.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
