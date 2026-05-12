"""Logging helpers."""

from __future__ import annotations

import logging
from logging.config import dictConfig


def configure_logging() -> None:
    """Install a lightweight structured logging configuration."""

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                }
            },
            "root": {"handlers": ["console"], "level": "INFO"},
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance."""

    configure_logging()
    return logging.getLogger(name)
