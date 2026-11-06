"""Exceptions shared by the CLI and future application boundaries."""


class NeuroVisionError(Exception):
    """Base exception for expected NeuroVision failures."""


class ConfigurationError(NeuroVisionError):
    """Application configuration failed validation."""


class PipelineError(NeuroVisionError):
    """A pipeline component failed to produce a usable result."""
