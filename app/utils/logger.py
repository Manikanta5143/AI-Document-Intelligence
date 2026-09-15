import logging
import sys


def setup_logger(name: str = "app", level: str = "INFO") -> logging.Logger:
    """Configure and return a standardized logger instance.

    Args:
        name: Name of the logger, typically __name__ or module name.
        level: Logging level as a string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
