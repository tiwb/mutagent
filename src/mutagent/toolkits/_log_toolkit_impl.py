"""mutagent.toolkits._log_toolkit_impl -- LogToolkit query implementation."""

from __future__ import annotations

from datetime import datetime

import mutobj
from .log_toolkit import LogToolkit


@mutobj.impl(LogToolkit.query)
def log_toolkit_query(
    self: LogToolkit,
    pattern: str = "",
    level: str = "DEBUG",
    limit: int = 50,
    tool_capture: str = "",
) -> str:
    """Query log entries or configure logging."""
    parts: list[str] = []

    # Handle tool_capture configuration
    if tool_capture.lower() == "on":
        self.log_store.tool_capture_enabled = True
    elif tool_capture.lower() == "off":
        self.log_store.tool_capture_enabled = False

    # Status line
    capture_status = "on" if self.log_store.tool_capture_enabled else "off"
    total = self.log_store.count()
    parts.append(f"[Tool capture: {capture_status} | Total entries: {total}]")
    parts.append("")

    # Query entries
    entries = self.log_store.query(pattern=pattern, level=level, limit=limit)
    if not entries:
        parts.append("(no matching entries)")
    else:
        for entry in entries:
            ts = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            parts.append(
                f"{ts} {entry.level:<8s} {entry.logger_name:<20s} - {entry.message}"
            )
        parts.append("")
        parts.append(f"(showing {len(entries)} of {total} entries, newest first)")

    return "\n".join(parts)
