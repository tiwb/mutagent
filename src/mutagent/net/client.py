"""兼容层 — re-export from mutio.net.client + mutio.mcp.client。"""
from mutio.net.client import *  # noqa: F401,F403  # pyright: ignore[reportWildcardImportFromLibrary]
from mutio.mcp.client import MCPClient, MCPError  # noqa: F401
