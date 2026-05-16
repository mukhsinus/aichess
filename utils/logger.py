"""
Logging configuration for the AI Chess Assistant.

Provides a single ``get_logger(name)`` factory so every module gets a
consistently formatted logger.  Call once per module at the top:

    from utils.logger import get_logger
    logger = get_logger(__name__)

Levels
------
- DEBUG : frame-level detail (FEN candidates, detection counts)
- INFO  : lifecycle events (board detected, FEN accepted, move announced)
- WARNING : recoverable issues (invalid FEN, Stockfish timeout)
- ERROR : failures (camera not opened, model file missing)
"""

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with console output.

    The root handler is installed once; subsequent calls just return a
    child logger under the given *name*.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        _CONFIGURED = True

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
