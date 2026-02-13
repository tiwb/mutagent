"""mutagent - A Python AI Agent framework for runtime self-iterating code."""

__version__ = "0.0.1"

from mutagent.base import MutagentMeta, Object
from forwardpy import impl

__all__ = ["Object", "MutagentMeta", "impl"]
