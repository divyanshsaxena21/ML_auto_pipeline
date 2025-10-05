import logging
from typing import Optional

LOGGER_NAME = "ml_autopipeline"

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the package root."""
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")

def configure_logging(level: int = logging.INFO):
    """Configure root package logger if not already configured.

    Safe to call multiple times; only adds handlers once.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    # Avoid propagating to root to prevent duplicate prints if root configured.
    logger.propagate = False
    return logger
