
"""Centralized logging configuration."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
) -> None:
    """Configure application logging to console and a rotating file."""

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_format = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    formatter = logging.Formatter(log_format)

    file_handler = RotatingFileHandler(
        log_path / "pipeline.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging is called again.
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    else:
        # Configure existing handlers rather than adding duplicates.
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
