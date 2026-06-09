"""Optional LLM proxy routes for mutagent.

Importing this package registers the HTTP View subclasses. Call
``configure_llm_proxy()`` to bind a ``Config`` instance before serving traffic.
"""

from .routes import configure_llm_proxy, get_llm_proxy_runtime, reset_llm_proxy_runtime
from . import routes as _routes  # noqa: F401  # pyright: ignore[reportUnusedImport]

__all__ = [
    "configure_llm_proxy",
    "get_llm_proxy_runtime",
    "reset_llm_proxy_runtime",
]
