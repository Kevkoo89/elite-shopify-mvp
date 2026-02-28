from __future__ import annotations

import logging
import os
import re
from pathlib import Path

LOG_DIR = Path("runtime_logs")
LOG_DIR.mkdir(exist_ok=True)

_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(token\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(password\s*[=:]\s*)([^\s,;]+)"),
]


class SecretRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for pattern in _SECRET_PATTERNS:
            message = pattern.sub(r"\1***", message)
        record.msg = message
        record.args = ()
        return True


def _resolve_log_level() -> int:
    level_name = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def get_app_logger(name: str = "elite_analyst") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(_resolve_log_level())
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    redaction_filter = SecretRedactionFilter()

    file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redaction_filter)
    logger.addHandler(file_handler)

    error_handler = logging.FileHandler(LOG_DIR / "errors.log", encoding="utf-8")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    error_handler.addFilter(redaction_filter)
    logger.addHandler(error_handler)

    logger.propagate = False
    return logger
