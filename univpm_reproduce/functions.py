from __future__ import annotations

import logging
import logging.config
from pathlib import Path

from lightrag.utils import logger, set_verbose_debug


def initialize_logger(log_dir: Path, log_filename: str = "step_1.log", verbose_debug: bool = False) -> Path:
    """Initialize LightRAG logger with console + rotating file handlers."""
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = (log_dir / log_filename).resolve()

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_file_path),
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    logger.setLevel(logging.INFO)
    set_verbose_debug(verbose_debug)
    return log_file_path


def write_constants_snapshot(
    output_dir: Path,
    constants: dict[str, object],
    file_name: str = "constants.txt",
) -> Path:
    """Write uppercase constants into a plain text snapshot file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / file_name

    filtered_constants: dict[str, object] = {}
    for key, value in constants.items():
        if not key.isupper():
            continue
        if callable(value):
            continue
        filtered_constants[key] = str(value) if isinstance(value, Path) else value

    lines = [f"{key}={filtered_constants[key]}" for key in sorted(filtered_constants.keys())]
    snapshot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return snapshot_path
