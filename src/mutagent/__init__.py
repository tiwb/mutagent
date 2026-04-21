"""mutagent - A Python AI Agent framework for runtime self-iterating code."""

__version__ = "0.6.999"

from mutobj import Declaration, impl, field, register_module_impls, unregister_module_impls
from mutagent.tools import Toolkit
from mutio.net.client import HttpClient

HttpClient.set_default_user_agent(f"mutagent/{__version__}")

__all__ = ["Declaration", "impl", "field", "register_module_impls", "unregister_module_impls", "Toolkit"]
