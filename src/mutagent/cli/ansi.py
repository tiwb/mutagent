"""mutagent.cli.ansi -- ANSI color utilities + tool formatting + Markdown highlighting.

Merged from ``runtime/ansi.py``, ``builtins/userio_impl.py`` formatting helpers,
and ``builtins/block_handlers.py`` task colorization.
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache

# ---------------------------------------------------------------------------
# Terminal capability detection
# ---------------------------------------------------------------------------

def _enable_windows_ansi() -> bool:
    """Enable ANSI/VT processing on Windows 10+."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _color_supported() -> bool:
    """Check if the terminal supports ANSI colors."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        return _enable_windows_ansi()
    return True


# ---------------------------------------------------------------------------
# ANSI SGR codes
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_ITALIC = "\033[3m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"

# ---------------------------------------------------------------------------
# Color wrapper functions
# ---------------------------------------------------------------------------

def dim(text: str) -> str:
    """Wrap *text* with dim (faint) styling."""
    if not _color_supported():
        return text
    return f"{_DIM}{text}{_RESET}"


def bold(text: str) -> str:
    """Wrap *text* with bold styling."""
    if not _color_supported():
        return text
    return f"{_BOLD}{text}{_RESET}"


def green(text: str) -> str:
    """Wrap *text* with green foreground."""
    if not _color_supported():
        return text
    return f"{_GREEN}{text}{_RESET}"


def red(text: str) -> str:
    """Wrap *text* with red foreground."""
    if not _color_supported():
        return text
    return f"{_RED}{text}{_RESET}"


def bold_red(text: str) -> str:
    """Wrap *text* with bold red foreground."""
    if not _color_supported():
        return text
    return f"{_BOLD}{_RED}{text}{_RESET}"


def yellow(text: str) -> str:
    """Wrap *text* with yellow foreground."""
    if not _color_supported():
        return text
    return f"{_YELLOW}{text}{_RESET}"


def cyan(text: str) -> str:
    """Wrap *text* with cyan foreground."""
    if not _color_supported():
        return text
    return f"{_CYAN}{text}{_RESET}"


def bold_cyan(text: str) -> str:
    """Wrap *text* with bold cyan foreground."""
    if not _color_supported():
        return text
    return f"{_BOLD}{_CYAN}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Markdown lightweight syntax highlighting
# ---------------------------------------------------------------------------


# Line-start patterns that highlight the ENTIRE line (marker + content)
_MD_LINE_FULL = [
    re.compile(r'^#{1,6}\s'),              # headings
    re.compile(r'^>\s?'),                   # blockquote
]

# Line-start patterns that highlight ONLY the marker (content stays default)
_MD_LINE_MARKER_ONLY = [
    (re.compile(r'^(\s*[-*+]\s)(.*)$'), 1),      # unordered list
    (re.compile(r'^(\s*\d+\.\s)(.*)$'), 1),       # ordered list
]

# Inline markers (can match multiple times per line)
_MD_BOLD_RE = re.compile(r'(\*\*[^*]+\*\*|__[^_]+__)')
_MD_INLINE_CODE_RE = re.compile(r'(`[^`]+`)')


def highlight_markdown_line(line: str) -> str:
    """Apply lightweight Markdown syntax highlighting to a single line.

    - Headings and blockquotes: entire line highlighted
    - Lists: only the marker highlighted, content stays default
    - Bold and inline code: entire span highlighted
    Returns the line unchanged when color is disabled.
    """
    if not _color_supported():
        return line

    # Headings and blockquotes: highlight entire line
    for pattern in _MD_LINE_FULL:
        if pattern.match(line):
            return f"{_CYAN}{line}{_RESET}"

    # Lists: highlight only the marker
    for pattern, group_idx in _MD_LINE_MARKER_ONLY:
        m = pattern.match(line)
        if m:
            marker = m.group(group_idx)
            rest = m.group(group_idx + 1)
            rest = _apply_inline_patterns(rest)
            return f"{_CYAN}{marker}{_RESET}{rest}"

    # No line-start match -- apply inline patterns to the whole line
    return _apply_inline_patterns(line)


def _apply_inline_patterns(text: str) -> str:
    """Apply inline Markdown highlighting (bold spans, inline code)."""
    # Inline code first (takes precedence) -- yellow to distinguish from cyan headings
    text = _MD_INLINE_CODE_RE.sub(f"{_YELLOW}\\1{_RESET}", text)
    # Bold spans -- cyan
    text = _MD_BOLD_RE.sub(f"{_CYAN}\\1{_RESET}", text)
    return text


# ---------------------------------------------------------------------------
# Tool call / result formatting
# ---------------------------------------------------------------------------

_MAX_VALUE_LEN = 60       # max display length for a single parameter value
_MAX_SINGLE_LINE = 80     # max total length before switching to multi-line
_INDENT = "  "            # base indentation
_PARAM_INDENT = "      "  # parameter indentation in multi-line mode (6 spaces)
_PREVIEW_LINES = 4        # default number of result preview lines
_RESULT_INDENT = "    "   # result continuation indent (4 spaces)


def _format_value(value) -> str:
    """Format a single argument value in Python style."""
    if isinstance(value, str):
        display = value
        if len(display) > _MAX_VALUE_LEN:
            display = display[:_MAX_VALUE_LEN - 3] + "..."
        return f'"{display}"'
    r = repr(value)
    if len(r) > _MAX_VALUE_LEN:
        r = r[:_MAX_VALUE_LEN - 3] + "..."
    return r


def _format_tool_call(name: str, args: dict) -> str:
    """Format a tool call as a Python-style function call string."""
    if not args:
        return f"{_INDENT}{name}()"

    # Build parameter strings
    params = [f"{k}={_format_value(v)}" for k, v in args.items()]

    # Try single-line first
    single = f"{_INDENT}{name}({', '.join(params)})"
    if len(single) <= _MAX_SINGLE_LINE:
        return single

    # Multi-line form
    lines = [f"{_INDENT}{name}("]
    for p in params:
        lines.append(f"{_PARAM_INDENT}{p},")
    lines.append(f"{_INDENT})")
    return "\n".join(lines)


def _format_tool_result(content: str, is_error: bool) -> str:
    """Format a tool result with preview and line count."""
    color = bold_red if is_error else green
    lines = content.split("\n") if content else [""]

    if len(lines) <= _PREVIEW_LINES:
        # Short result: show everything
        first = f"{_INDENT}\u2192 {lines[0]}"
        result_lines = [color(first)]
        for extra in lines[1:]:
            result_lines.append(color(f"{_RESULT_INDENT}{extra}"))
        return "\n".join(result_lines)

    # Long result: preview + overflow indicator
    first = f"{_INDENT}\u2192 {lines[0]}"
    result_lines = [color(first)]
    for extra in lines[1:_PREVIEW_LINES]:
        result_lines.append(color(f"{_RESULT_INDENT}{extra}"))
    remaining = len(lines) - _PREVIEW_LINES
    result_lines.append(dim(f"{_RESULT_INDENT}... +{remaining} lines"))
    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Task line colorization
# ---------------------------------------------------------------------------

_TASK_CHECK_RE = re.compile(r'^(\s*(?:[-*])\s*)(\[[ x~]\])(.*)')


def _colorize_task_line(line: str) -> str:
    """Colorize task checkmark in a task list line."""
    m = _TASK_CHECK_RE.match(line)
    if not m:
        return line
    prefix, check, rest = m.group(1), m.group(2), m.group(3)
    if check == '[x]':
        return prefix + green(check) + rest
    elif check == '[~]':
        return prefix + yellow(check) + rest
    elif check == '[ ]':
        return prefix + dim(check) + rest
    return line
