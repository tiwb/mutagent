"""mutagent.builtins.run_code -- run_code tool implementation."""

import io
import sys
import traceback

import mutagent
from mutagent.essential_tools import EssentialTools


@mutagent.impl(EssentialTools.run_code)
def run_code(self: EssentialTools, code: str) -> str:
    """Execute a Python code snippet and return the result."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        # Execute in a namespace with access to common modules
        namespace = {"__name__": "__run_code__", "__builtins__": __builtins__}
        exec(code, namespace)

        stdout_val = stdout_capture.getvalue()
        stderr_val = stderr_capture.getvalue()

        parts = []
        if stdout_val:
            parts.append(stdout_val.rstrip())
        if stderr_val:
            parts.append(f"[stderr]\n{stderr_val.rstrip()}")

        return "\n".join(parts) if parts else "(no output)"

    except Exception:
        stdout_val = stdout_capture.getvalue()
        tb = traceback.format_exc()
        parts = []
        if stdout_val:
            parts.append(stdout_val.rstrip())
        parts.append(tb.rstrip())
        return "\n".join(parts)

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
