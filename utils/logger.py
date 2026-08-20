"""
utils/logger.py
───────────────
Unified Rich + standard-library logging setup.
Call get_logger(__name__) in every module.
"""

import logging
import sys
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from core.config import settings

# ── Rich console with custom theme ───────────────────────────────────────────
custom_theme = Theme({
    "agent.researcher":   "bold cyan",
    "agent.fact_checker": "bold yellow",
    "agent.summarizer":   "bold green",
    "agent.analyst":      "bold magenta",
    "agent.critic":       "bold red",
    "agent.writer":       "bold blue",
    "info":               "dim white",
    "success":            "bold green",
    "warning":            "bold yellow",
    "error":              "bold red",
})

console = Console(theme=custom_theme)


def get_logger(name: str) -> logging.Logger:
    """Return a Rich-powered logger for the given module name."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_time=True,
        show_level=True,
        markup=True,
    )
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger