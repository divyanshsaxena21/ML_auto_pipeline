import logging
import json
import time
from typing import Optional

LOGGER_NAME = "ml_autopipeline"

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the package root."""
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")

def configure_logging(level: int = logging.INFO, json_logs: bool = False, log_file: Optional[str] = None):
    """Configure root package logger if not already configured.

    Safe to call multiple times; only adds handlers once.
    """
    logger = logging.getLogger(LOGGER_NAME)
    formatter: logging.Formatter
    if json_logs:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s", datefmt="%H:%M:%S")

    # Only add handlers once (avoid duplication when CLI + library code)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    else:
        # Update formatters / level if already configured
        for h in logger.handlers:
            if json_logs and not isinstance(h.formatter, JsonFormatter):
                h.setFormatter(JsonFormatter())
            elif not json_logs and isinstance(h.formatter, JsonFormatter):
                h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s", datefmt="%H:%M:%S"))
        if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger
