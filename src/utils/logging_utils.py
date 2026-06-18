"""Structured logging setup for FedAgent-Chain."""

from __future__ import annotations

import logging
import platform
import sys
from typing import Any

import structlog


def setup_logging(level: str = "INFO", format: str = "json") -> None:
    """Configure structured logging for the framework.

    Parameters
    ----------
    level : str
        Logging level. One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    format : str
        Output format. 'json' for production, 'console' for development.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: object
    if format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a named structured logger.

    Parameters
    ----------
    name : str
        Logger name, typically the calling module's __name__ or class name.

    Returns
    -------
    structlog.BoundLogger
        A bound structured logger instance.

    Examples
    --------
    >>> logger = get_logger("FederatedServer")
    >>> logger.info("Round started", round=1, n_clients=4)
    """
    return structlog.get_logger(name)


def log_hardware_info() -> dict[str, Any]:
    """Log and return the current hardware configuration.

    Returns
    -------
    dict
        Hardware configuration dictionary containing CPU, RAM, GPU, and
        software version information.
    """
    import multiprocessing

    import torch

    hw_info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": multiprocessing.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        hw_info["gpu_count"] = torch.cuda.device_count()
        hw_info["gpu_name"] = torch.cuda.get_device_name(0)
        hw_info["cuda_version"] = torch.version.cuda or "unknown"

    try:
        import psutil

        hw_info["total_ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        hw_info["total_ram_gb"] = "unknown (psutil not installed)"

    logger = get_logger("hardware")
    logger.info("Hardware configuration", **hw_info)
    return hw_info
