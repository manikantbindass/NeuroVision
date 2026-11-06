"""Application logging for operational events."""

import json
import logging
from datetime import UTC, datetime

from neurovision.core.settings import LogLevel


class JsonFormatter(logging.Formatter):
    """Emit one JSON record per line with an explicit field allowlist."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


def create_cli_logger(level: LogLevel) -> logging.Logger:
    """Send CLI events to stderr independently of the host logging configuration."""
    logger = logging.Logger("neurovision.cli", level=level.value)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger
