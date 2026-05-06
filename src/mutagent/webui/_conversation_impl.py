"""Default Conversation implementation."""

from __future__ import annotations

import time
from functools import partial
import logging
from typing import Any

import mutagent
from mutagent.messages import Message, StreamEvent, TextBlock
from mutagent.webui.chat_input import ChatInput
from mutagent.webui.conversation import Conversation
from mutagent.webui.messages import (
    AssistantErrorItem,
    AssistantTextItem,
    MessageList,
    ToolCallItem,
    TurnSeparatorItem,
    UserTextItem,
)
from mutagent.webui.toolbar import AgentStatusBar
from mutagent.webui.settings import LLMSettingsPanel
from mutgui import ActionContext, ActionToolbar, Callback, ViewBlock

logger = logging.getLogger(__name__)


def _now_item_id(prefix: str) -> str:
    return f"{prefix}-{time.time_ns()}"


def _extract_text(message: Message) -> str:
    return "".join(
        block.text for block in message.blocks if isinstance(block, TextBlock)
    )


def _resolve_current_model_name(agent: Any, models: list[dict[str, Any]]) -> str:
    current_id = getattr(getattr(agent, "llm", None), "model", "")
    for model in models:
        if model.get("name") == current_id or model.get("model_id") == current_id:
            return str(model.get("name", current_id))
    return current_id


def _find_last_assistant(items: list[Any]) -> AssistantTextItem | None:
    for item in reversed(items):
        if isinstance(item, AssistantTextItem):
            return item
    return None


@mutagent.impl(Conversation.__init__)
def __init__(self: Conversation, *, agent: Any, app: Any = None) -> None:
    super(Conversation, self).__init__()
    self.app = app
    self.agent = agent
    self.items: list[Any] = []
    self.message_list = MessageList(items=self.items)
    self.message_list.id = "message-list"
    self.models = agent.list_models()
    self.current_model = _resolve_current_model_name(agent, self.models)
    self.status = "idle"
    self.is_busy = False
    self._cancel_requested = False
    self._current_assistant_id = ""
    self._tool_item_ids: dict[str, str] = {}
    self._turn_input_tokens = 0
    self._turn_output_tokens = 0
    self._turn_started_at = 0.0
    self._total_cost = 0.0
    # 模块级函数绑定到 self（Conversation 类未声明这些为方法，手动用 partial 绑）
    self._handle_model_change = partial(_handle_model_change, self)
    self._handle_send = partial(_handle_send, self)
    self._handle_cancel = partial(_handle_cancel, self)
    self._handle_agent_event = partial(_handle_agent_event, self)
    self.status_bar = AgentStatusBar(
        status=self.status,
    )
    self.status_bar.id = "agent-status-bar"
    self.chat_input = ChatInput(
        on_send=self._handle_send,
        on_cancel=self._handle_cancel,
    )
    self.chat_input.id = "chat-input"
    self.chat_input.conversation = self
    self.settings_open = False
    self.settings_panel = LLMSettingsPanel(
        app=app,
        agent=agent,
        on_close=partial(_close_settings, self),
        on_saved=partial(_settings_saved, self),
    )
    self._open_settings_action = partial(_open_settings, self)
    self._close_settings_action = partial(_close_settings, self)
    self._refresh_models_action = partial(_refresh_models_from_config, self)
    self.toolbar = ActionToolbar(
        id="conversation-toolbar",
        categories=["mutagent.conversation.toolbar"],
        context=ActionContext(owner=self, data={"conversation": self}),
        label_mode="auto",
    )
    self._subscription = agent.subscribe(self._handle_agent_event)


def _refresh_shell(self: Conversation) -> None:
    self.status_bar.status = self.status
    self.status_bar.input_tokens = self._turn_input_tokens
    self.status_bar.output_tokens = self._turn_output_tokens
    self.status_bar.total_cost = self._total_cost
    # context + cache — synced from AgentContext (defensive: may be absent in tests)
    ctx = getattr(self.agent, "context", None)
    if ctx is not None:
        self.status_bar.context_used = ctx.get_context_used()
        ctx_pct = ctx.get_context_percent()
        self.status_bar.context_percent = ctx_pct if ctx_pct is not None else 0.0
        self.status_bar.context_total = ctx.context_window
        self.status_bar.cache_read_tokens = ctx.get_cache_read_tokens()
        self.status_bar.cache_write_tokens = ctx.get_cache_write_tokens()
    self.chat_input.disabled = False
    self.chat_input.is_busy = self.is_busy
    self.toolbar.context = ActionContext(owner=self, data={"conversation": self})
    self.status_bar.invalidate()
    self.chat_input.invalidate()
    self.toolbar.invalidate()


def _append_item(self: Conversation, item: Any) -> None:
    self.items.append(item)
    self.message_list.refresh()


def _touch_item(self: Conversation, item_id: str) -> None:
    self.message_list.invalidate_item(item_id)


def _find_item(self: Conversation, item_id: str) -> Any | None:
    for item in self.items:
        if getattr(item, "id", "") == item_id:
            return item
    return None


async def _open_settings(self: Conversation) -> None:
    self.settings_open = True
    reload_panel = getattr(self.settings_panel, "_reload", None)
    if callable(reload_panel):
        reload_panel()
    self.settings_panel.invalidate()
    self.invalidate()


async def _close_settings(self: Conversation) -> None:
    self.settings_open = False
    self.invalidate()


async def _refresh_models_from_config(self: Conversation, preferred_model: str = "") -> None:
    self.models = self.agent.list_models()
    available = [str(model.get("name", "")) for model in self.models if model.get("name")]
    desired = preferred_model or str(self.agent.config.get("default_model", default="") or "")
    if not desired or desired not in available:
        desired = available[0] if available else ""
    if desired and not self.is_busy:
        try:
            self.agent.select_model(desired)
        except Exception as exc:
            _append_item(
                self,
                AssistantErrorItem(
                    id=_now_item_id("error"),
                    kind="assistant.error",
                    error=str(exc),
                    timestamp=time.time(),
                ),
            )
        else:
            self.current_model = desired
    else:
        self.current_model = _resolve_current_model_name(self.agent, self.models)
    _refresh_shell(self)
    self.invalidate()


async def _settings_saved(self: Conversation, preferred_model: str) -> None:
    await _refresh_models_from_config(self, preferred_model)
    await _close_settings(self)


async def _handle_send(self: Conversation, text: str) -> None:
    if self.is_busy:
        logger.info("Conversation send ignored while busy")
        return
    logger.info("Conversation submit requested (%d chars)", len(text))
    _append_item(
        self,
        UserTextItem(
            id=_now_item_id("user"),
            kind="user.text",
            text=text,
            timestamp=time.time(),
        ),
    )
    self.is_busy = True
    self.status = "thinking"
    self._cancel_requested = False
    self._current_assistant_id = ""
    self._tool_item_ids = {}
    self._turn_input_tokens = 0
    self._turn_output_tokens = 0
    self._turn_started_at = time.time()
    _refresh_shell(self)
    self.invalidate()
    try:
        await self.agent.submit(text)
    except Exception as exc:
        logger.exception("Conversation submit failed")
        self.is_busy = False
        self.status = "idle"
        _append_item(
            self,
            AssistantErrorItem(
                id=_now_item_id("error"),
                kind="assistant.error",
                error=str(exc),
                timestamp=time.time(),
            ),
        )
        _refresh_shell(self)
        self.invalidate()


async def _handle_cancel(self: Conversation) -> None:
    if self.agent.cancel():
        logger.info("Conversation cancel requested")
        self._cancel_requested = True
        self.status = "cancelling"
        _refresh_shell(self)
        self.invalidate()


async def _handle_model_change(self: Conversation, name: str) -> None:
    logger.info("Conversation model change requested: %s", name)
    try:
        self.agent.select_model(name)
    except Exception as exc:
        _append_item(
            self,
            AssistantErrorItem(
                id=_now_item_id("error"),
                kind="assistant.error",
                error=str(exc),
                timestamp=time.time(),
            ),
        )
    else:
        self.models = self.agent.list_models()
        self.current_model = name
    _refresh_shell(self)
    self.invalidate()


def _ensure_current_assistant(self: Conversation, event: StreamEvent) -> AssistantTextItem:
    item = None
    if self._current_assistant_id:
        found = _find_item(self, self._current_assistant_id)
        if isinstance(found, AssistantTextItem):
            item = found
    if item is None:
        response = event.response.message if event.response else None
        item = AssistantTextItem(
            id=(response.id if response and response.id else _now_item_id("assistant")),
            kind="assistant.text",
            text="",
            model=(response.model if response else ""),
            timestamp=(response.timestamp if response else time.time()),
        )
        self._current_assistant_id = item.id
        _append_item(self, item)
    return item


async def _handle_agent_event(self: Conversation, event: StreamEvent) -> None:
    logger.debug("Conversation received event: %s", event.type)
    if event.type == "response_start":
        logger.info("Assistant response started")
        response = event.response.message if event.response else None
        item = AssistantTextItem(
            id=(response.id if response and response.id else _now_item_id("assistant")),
            kind="assistant.text",
            text="",
            model=(response.model if response else ""),
            timestamp=(response.timestamp if response else time.time()),
        )
        self._current_assistant_id = item.id
        _append_item(self, item)
        self.status = "thinking"
    elif event.type == "text_delta" and event.text:
        item = _ensure_current_assistant(self, event)
        item.text += event.text
        _touch_item(self, item.id)
    elif event.type == "tool_exec_start" and event.tool_call is not None:
        logger.info("Tool execution started: %s", event.tool_call.name)
        tool_call = event.tool_call
        item_id = self._tool_item_ids.get(tool_call.id)
        if item_id is None:
            item_id = f"tool-{tool_call.id or time.time_ns()}"
            self._tool_item_ids[tool_call.id] = item_id
            _append_item(
                self,
                ToolCallItem(
                    id=item_id,
                    kind="assistant.tool_group",
                    tool_id=tool_call.id,
                    name=tool_call.name,
                    input_text=str(tool_call.input),
                    status="pending",
                ),
            )
        tool_item = _find_item(self, item_id)
        if isinstance(tool_item, ToolCallItem):
            tool_item.status = "pending"
            tool_item.input_text = str(tool_call.input)
            _touch_item(self, item_id)
        self.status = "tool_calling"
    elif event.type == "tool_exec_end" and event.tool_call is not None:
        logger.info("Tool execution finished: %s", event.tool_call.name)
        tool_call = event.tool_call
        item_id = self._tool_item_ids.get(tool_call.id)
        tool_item = _find_item(self, item_id) if item_id else None
        if isinstance(tool_item, ToolCallItem):
            tool_item.status = "error" if tool_call.is_error else "success"
            tool_item.result_text = tool_call.result
            tool_item.duration = tool_call.duration
            tool_item.is_error = tool_call.is_error
            _touch_item(self, tool_item.id)
        self.status = "thinking"
    elif event.type == "response_done" and event.response is not None:
        assistant_item = _ensure_current_assistant(self, event)
        response = event.response
        response_text = _extract_text(response.message)
        if response_text and not assistant_item.text:
            assistant_item.text = response_text
        assistant_item.model = response.message.model or assistant_item.model
        assistant_item.duration += response.message.duration
        assistant_item.input_tokens += response.message.input_tokens
        assistant_item.output_tokens += response.message.output_tokens
        self._turn_input_tokens += response.message.input_tokens
        self._turn_output_tokens += response.message.output_tokens
        _touch_item(self, assistant_item.id)
    elif event.type == "error":
        logger.error("Conversation received error event: %s", event.error or "Unknown error")
        _append_item(
            self,
            AssistantErrorItem(
                id=_now_item_id("error"),
                kind="assistant.error",
                error=event.error or "Unknown error",
                timestamp=time.time(),
            ),
        )
        self.status = "idle"
        self.is_busy = False
    elif event.type == "turn_done":
        logger.info("Conversation turn finished")
        if self._cancel_requested:
            assistant_item = _find_item(self, self._current_assistant_id)
            if isinstance(assistant_item, AssistantTextItem):
                if "[interrupted]" not in assistant_item.text:
                    assistant_item.text = (
                        assistant_item.text.rstrip() + "\n\n[interrupted]"
                    ).strip()
                    _touch_item(self, assistant_item.id)
        duration = max(0.0, time.time() - self._turn_started_at) if self._turn_started_at else 0.0
        _append_item(
            self,
            TurnSeparatorItem(
                id=f"turn-{event.turn_id or time.time_ns()}",
                kind="turn_done",
                turn_id=event.turn_id,
                duration=duration,
                input_tokens=self._turn_input_tokens,
                output_tokens=self._turn_output_tokens,
            ),
        )
        self.is_busy = False
        self.status = "idle"
        self._cancel_requested = False
        self._current_assistant_id = ""
        self._tool_item_ids = {}
    _refresh_shell(self)
    self.invalidate()


@mutagent.impl(Conversation.render)
def render(self: Conversation) -> ViewBlock:
    _refresh_shell(self)
    return ViewBlock([
        {
            "$component": "div",
            "$id": "conversation-root",
            "style": {
                "--mutagent-font-size-base": "13px",
                "--mutagent-font-size-meta": "12px",
                "display": "flex",
                "flexDirection": "column",
                "height": "100vh",
                "boxSizing": "border-box",
                "color": "var(--mutgui-text)",
                "gap": 0,
                "position": "relative",
            },
            "$children": [
                {
                    "$component": "div",
                    "$id": "toolbar-shell",
                    "style": {"padding": "8px 12px"},
                    "$children": [self.toolbar],
                },
                {
                    "$component": "div",
                    "$id": "messages-shell",
                    "style": {
                        "flex": 1,
                        "minHeight": 0,
                        "display": "flex",
                        "flexDirection": "column",
                        "overflow": "hidden",
                    },
                    "$children": [self.message_list],
                },
                self.chat_input,
                {
                    "$component": "antd.Drawer",
                    "$id": "llm-settings-drawer",
                    "title": "LLM API 设置",
                    "placement": "right",
                    "open": self.settings_open,
                    "width": 560,
                    "destroyOnHidden": False,
                    "onClose": Callback(self._close_settings_action),
                    "$children": [self.settings_panel],
                },
            ],
        }
    ])
