"""Keystroke-injection backends, selected at runtime behind a common protocol."""

from backends.input.base import InputBackend, Key
from backends.input.factory import create_input_backend

__all__ = ["InputBackend", "Key", "create_input_backend"]
