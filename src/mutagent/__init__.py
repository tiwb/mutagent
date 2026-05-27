"""mutagent - A Python AI Agent framework for runtime self-iterating code."""

__version__ = "0.9.999"

from mutagent.core.tools import Toolkit
from mutio.net.client import HttpClient

HttpClient.set_default_user_agent(f"mutagent/{__version__}")

__all__ = ["Toolkit"]
