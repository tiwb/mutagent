"""mutagent.log_toolkit -- LogToolkit declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mutagent.core.tools import Toolkit

if TYPE_CHECKING:
    from mutagent.app.log_store import LogStore


class LogToolkit(Toolkit):
    """Tools for querying logs and configuring log capture.

    Attributes:
        log_store: The LogStore instance for in-memory log storage.
    """

    log_store: LogStore

    def query(
        self,
        pattern: str = "",
        level: str = "DEBUG",
        limit: int = 50,
        tool_capture: str = "",
    ) -> str:
        """Query log entries or configure logging.

        Args:
            pattern: Regex pattern to search in log messages. Empty matches all.
            level: Minimum log level filter (DEBUG/INFO/WARNING/ERROR).
            limit: Maximum number of entries to return.
            tool_capture: Set to "on" or "off" to enable/disable tool log
                capture (logs appended to tool output). Empty string = no change.

        Returns:
            Formatted log entries, newest first.
        """
        ...


from . import _log_toolkit_impl as _log_toolkit_impl  # noqa: F401, E402
