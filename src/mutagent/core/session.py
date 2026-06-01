"""mutagent.core.session -- Public AgentSession declaration."""

from __future__ import annotations

from pathlib import Path

import mutobj

from .context import AgentContext


class AgentSession(mutobj.Declaration):
    """Runtime session lifecycle manager for CLI/Web entry points."""

    id: str = ""
    dir: Path | None = None
    cwd: str = ""
    model: str = ""

    def start_new(
        self,
        *,
        session_dir: str | Path,
        cwd: str,
        model: str,
        session_id: str = "",
    ) -> None:
        """Prepare a new lazy session without creating the file yet."""
        ...

    def resume(self, value: str | Path, context: AgentContext) -> Path:
        """Load an existing session file into the given context."""
        ...

    def sync(self, context: AgentContext) -> None:
        """Append any new durable messages from context to the session file."""
        ...


from . import _session_impl as _session_impl  # noqa: F401, E402
