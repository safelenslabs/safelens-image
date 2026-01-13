"""
Logging configuration using Uvicorn's logger format.
"""

import logging
import sys
from typing import Optional


# Uvicorn-style logging format with colors
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s [%(name)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(asctime)s [%(name)s] %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True,
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}


def setup_logging(level: int = logging.INFO) -> None:
    """
    Setup logging configuration using Uvicorn's format.

    Args:
        level: Logging level (default: INFO)
    """
    from logging.config import dictConfig

    # Update log level in config
    LOGGING_CONFIG["loggers"]["uvicorn"]["level"] = logging.getLevelName(level)
    LOGGING_CONFIG["loggers"]["uvicorn.error"]["level"] = logging.getLevelName(level)

    # Apply configuration
    dictConfig(LOGGING_CONFIG)

    # Set root logger level
    logging.root.setLevel(level)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger instance using Uvicorn's formatter.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Optional logging level override

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Ensure logger uses the default handler if not already configured
    if not logger.handlers and name not in LOGGING_CONFIG["loggers"]:
        from uvicorn.logging import DefaultFormatter

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            DefaultFormatter(
                fmt="%(levelprefix)s %(asctime)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                use_colors=True,
            )
        )
        logger.addHandler(handler)

    if level is not None:
        logger.setLevel(level)

    return logger
