import time
import logging
from contextlib import contextmanager
from typing import Callable, Any
from .logging_utils import get_logger

logger = get_logger("timing")

def time_block(label: str, callback: Callable[[float], None] | None = None):
    start = time.perf_counter()
    yield_obj = {}
    try:
        yield yield_obj
    finally:
        elapsed = time.perf_counter() - start
        if callback:
            callback(elapsed)
        else:
            logger.info(f"{label} completed in {elapsed:.3f}s")

@contextmanager
def timed(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"{label} took {elapsed:.3f}s")

def timing_decorator(label: str):
    def outer(fn: Callable):
        def inner(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logger.info(f"{label} ({fn.__name__}) took {elapsed:.3f}s")
        return inner
    return outer
