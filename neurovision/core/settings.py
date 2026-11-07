"""Validate the process environment at application startup."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from neurovision.core.errors import ConfigurationError

_INVALID_SEED = "NEUROVISION_SEED must be an integer from 0 to 4294967295"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings with defaults suitable for a local research session."""

    seed: int = 42
    log_level: LogLevel = LogLevel.WARNING

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        values = os.environ if environ is None else environ
        defaults = cls()
        raw_seed = values.get("NEUROVISION_SEED", str(defaults.seed)).strip()
        if not raw_seed.isascii() or not raw_seed.isdecimal():
            raise ConfigurationError(_INVALID_SEED)
        digits = raw_seed.lstrip("0") or "0"
        if len(digits) > 10:
            raise ConfigurationError(_INVALID_SEED)
        seed = int(digits)
        if seed > 2**32 - 1:
            raise ConfigurationError(_INVALID_SEED)
        try:
            level = LogLevel(
                values.get("NEUROVISION_LOG_LEVEL", defaults.log_level).strip().upper()
            )
        except ValueError:
            raise ConfigurationError(
                "NEUROVISION_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL"
            ) from None
        return cls(seed=seed, log_level=level)
