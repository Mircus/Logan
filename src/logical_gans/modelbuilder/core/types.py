"""Three-valued truth used throughout the bounded model-building kernel."""
from __future__ import annotations

from enum import Enum


class Truth(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
