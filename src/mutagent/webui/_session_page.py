"""Session management — ResumePage Declaration + implementation + helpers.

Single-file module following the _settings_mcp.py pattern.
Provides ResumePage (View) and session lifecycle functions:
start_new_session(), resume_session().
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from mutagent.core.session import AgentSession
from mutgui import Action, ActionContext, Callback, View, ViewBlock

if TYPE_CHECKING:
    from mutagent.webui.conversation import Conversation


@dataclass(slots=True)
class SessionSummary:
    path: str
    title: str
    model: str
    updated_at: float
    updated_label: str


class ResumeSessionPage(View):
    """Session resume page for the built-in WebUI."""

    conversation: Conversation | None
    entries: list[SessionSummary]

    def __init__(self, *, conversation: Conversation | None = None) -> None:
        super().__init__()
        self.id = "resume-page"
        self.conversation = conversation
        self.entries = []

    def render(self) -> ViewBlock:
        return _render_resume_page(self)

    async def activate(self) -> None:
        await self.refresh()

    async def refresh(self) -> None:
        self.entries = _scan_sessions(_default_session_dir())
        self.invalidate()

    async def close(self) -> None:
        if self.conversation is not None:
            await self.conversation.navigate_to("")


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _default_session_dir() -> Path:
    return Path.home() / ".mutagent" / "sessions"


def _format_session_time(ts: float) -> str:
    if ts <= 0:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _extract_title(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") != "message":
                continue
            message = entry.get("message", {})
            if message.get("role") != "user":
                continue
            text = "".join(
                block.get("text", "")
                for block in message.get("content", [])
                if block.get("type") == "text"
            )
            normalized = " ".join(text.replace("\r", " ").replace("\n", " ").split())
            if normalized:
                return normalized
    return "(empty)"


def _read_session_summary(path: Path) -> SessionSummary:
    model = ""
    created_at = 0.0
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") != "session":
                continue
            model = str(entry.get("model", "") or "")
            timestamp = str(entry.get("timestamp", "") or "")
            if timestamp:
                try:
                    created_at = datetime.fromisoformat(
                        timestamp.replace("Z", "+00:00")
                    ).astimezone(timezone.utc).timestamp()
                except ValueError:
                    created_at = 0.0
            break
    updated_at = path.stat().st_mtime
    display_at = created_at or updated_at
    return SessionSummary(
        path=str(path),
        title=_extract_title(path),
        model=model,
        updated_at=updated_at,
        updated_label=_format_session_time(display_at),
    )


def _scan_sessions(session_dir: Path) -> list[SessionSummary]:
    return sorted(
        [_read_session_summary(path) for path in session_dir.glob("*.jsonl")],
        key=lambda entry: entry.updated_at,
        reverse=True,
    )


def _render_resume_page(self: ResumeSessionPage) -> ViewBlock:
    back_btn: dict[str, Any] = {
        "$component": "div",
        "$id": "resume-back-btn",
        "children": "← 返回对话",
        "onClick": Callback(_on_back_click, self),
        "style": {
            "cursor": "pointer",
            "userSelect": "none",
            "fontSize": "var(--mutagent-font-size-base)",
            "color": "var(--mutgui-text-secondary)",
        },
    }
    header: dict[str, Any] = {
        "$component": "div",
        "$id": "resume-header",
        "style": {
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "12px 16px",
            "borderBottom": "1px solid var(--mutgui-border)",
            "gap": "12px",
            "flex": "0 0 auto",
        },
        "$children": [
            {
                "$component": "div",
                "$id": "resume-header-left",
                "style": {
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "4px",
                },
                "$children": [
                    back_btn,
                    {
                        "$component": "div",
                        "$id": "resume-title",
                        "style": {
                            "fontSize": "18px",
                            "fontWeight": 600,
                            "color": "var(--mutgui-text)",
                        },
                        "children": "Resume Session",
                    },
                ],
            },
        ],
    }

    body_children: list[Any] = []
    if self.entries:
        for entry in self.entries:
            body_children.append({
                "$component": "antd.Button",
                "$id": f"resume-entry-{Path(entry.path).stem}",
                "type": "text",
                "block": True,
                "style": {
                    "height": "auto",
                    "padding": "12px 16px",
                    "textAlign": "left",
                    "justifyContent": "flex-start",
                    "border": "1px solid var(--mutgui-border)",
                    "borderRadius": "10px",
                },
                "onClick": Callback(_on_resume_click, self, entry.path),
                "$children": [{
                    "$component": "div",
                    "$id": "resume-entry-content",
                    "style": {
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "6px",
                        "width": "100%",
                        "minWidth": 0,
                    },
                    "$children": [
                        {
                            "$component": "div",
                            "$id": "resume-entry-title",
                            "style": {
                                "fontSize": "var(--mutagent-font-size-base)",
                                "fontWeight": 600,
                                "color": "var(--mutgui-text)",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "whiteSpace": "nowrap",
                                "width": "100%",
                            },
                            "children": entry.title,
                        },
                        {
                            "$component": "div",
                            "$id": "resume-entry-meta",
                            "style": {
                                "fontSize": "var(--mutagent-font-size-meta)",
                                "color": "var(--mutgui-text-dim)",
                            },
                            "children": " · ".join(part for part in [entry.updated_label, entry.model] if part),
                        },
                    ],
                }],
            })
    else:
        body_children.append({
            "$component": "div",
            "$id": "resume-empty",
            "style": {
                "padding": "32px 16px",
                "fontSize": "var(--mutagent-font-size-base)",
                "color": "var(--mutgui-text-dim)",
            },
            "children": "No saved sessions yet.",
        })

    root: dict[str, Any] = {
        "$component": "div",
        "$id": "resume-page-root",
        "style": {
            "display": "flex",
            "flexDirection": "column",
            "flex": 1,
            "minHeight": 0,
            "height": "100%",
            "color": "var(--mutgui-text)",
        },
        "$children": [
            header,
            {
                "$component": "div",
                "$id": "resume-list",
                "style": {
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "12px",
                    "padding": "16px",
                    "overflow": "auto",
                    "minHeight": 0,
                    "flex": 1,
                },
                "$children": body_children,
            },
        ],
    }
    return ViewBlock([root])


async def _on_back_click(view: ResumeSessionPage, *_: Any) -> None:
    await view.close()


async def _on_resume_click(view: ResumeSessionPage, session_path: str) -> None:
    if view.conversation is not None:
        await resume_session(view.conversation, session_path)


# ═══════════════════════════════════════════════════════════════
#  Main menu Actions (via category "mutagent.main_menu" auto-discovery)
# ═══════════════════════════════════════════════════════════════


class ResumeSessionAction(Action):
    action_id = "mutagent.menu.resume_session"
    categories = ("mutagent.main_menu",)
    label = "Resume Session"

    def check_enabled(self, context: ActionContext) -> bool:
        conv = context.get("conversation")
        return not getattr(conv, "is_busy", False)

    async def execute(self, context: ActionContext) -> None:
        conv = context.get("conversation")
        if conv is not None:
            navigate_to = getattr(conv, "navigate_to", None)
            if navigate_to is not None:
                await navigate_to("resume")


class NewSessionAction(Action):
    action_id = "mutagent.menu.new_session"
    categories = ("mutagent.main_menu",)
    label = "New Session"

    def check_enabled(self, context: ActionContext) -> bool:
        conv = context.get("conversation")
        return not getattr(conv, "is_busy", False)

    async def execute(self, context: ActionContext) -> None:
        conv = context.get("conversation")
        if conv is not None:
            await start_new_session(conv)


# ═══════════════════════════════════════════════════════════════
#  Session lifecycle (called by Conversation / toolbar Actions)
# ═══════════════════════════════════════════════════════════════


def _current_session_model(conversation: Any) -> str:
    return (
        str(getattr(conversation.agent, "model", "") or "")
        or conversation.current_model
        or str(getattr(getattr(conversation.agent, "llm", None), "model", "") or "")
    )


def _start_session(conversation: Any) -> AgentSession:
    # local import to avoid circular: _conversation_impl → _session_page → _conversation_impl
    from ._conversation_impl import _cext

    ext = _cext(conversation)
    ext.session = AgentSession()
    ext.session.start_new(
        session_dir=_default_session_dir(),
        cwd=str(Path.cwd()),
        model=_current_session_model(conversation),
    )
    object.__setattr__(conversation.agent, "session", ext.session)
    return ext.session


async def start_new_session(conversation: Any) -> None:
    """Clear all messages + reset state + create new AgentSession. Navigate to main if needed."""
    from ._conversation_impl import (
        _reset_context_usage,
        _replace_items,
        _reset_runtime_state,
        _refresh_shell,
    )

    context = getattr(conversation.agent, "context", None)
    if context is not None:
        context.messages = []
        _reset_context_usage(context)
    _replace_items(conversation, [])
    _reset_runtime_state(conversation)
    _start_session(conversation)
    _refresh_shell(conversation)
    if conversation.current_route:
        await conversation.navigate_to("")
    else:
        conversation.invalidate()


async def resume_session(conversation: Any, session_path: str | Path) -> None:
    """Load a saved session into the agent context and rebuild UI items."""
    from ._conversation_impl import (
        _cext,
        _reset_context_usage,
        _replace_items,
        _rebuild_items_from_messages,
        _reset_runtime_state,
        _refresh_shell,
    )

    ext = _cext(conversation)
    context = getattr(conversation.agent, "context", None)
    if context is None:
        return
    ext.session.resume(session_path, context)
    _reset_context_usage(context)
    _replace_items(conversation, _rebuild_items_from_messages(context.messages))
    _reset_runtime_state(conversation)
    _refresh_shell(conversation)
    await conversation.navigate_to("")
