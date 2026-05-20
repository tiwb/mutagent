"""Default Conversation implementation."""

from __future__ import annotations

import time
from functools import partial
import logging
from typing import Any

import mutagent
import mutobj
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
from mutagent.webui.settings import SettingsPage
from mutgui import ActionContext, ActionToolbar, Callback, ViewBlock
from mutgui.events import Event

logger = logging.getLogger(__name__)


# ── Extension ──────────────────────────────────────────────────────

class ConversationExt(mutobj.Extension[Conversation]):
    """Conversation 的运行时私有状态。"""
    items: list[Any] = mutobj.field(default_factory=list)
    message_list: Any = None
    status_bar: Any = None
    chat_input: Any = None
    toolbar: Any = None
    settings_page: Any = None
    cancel_requested: bool = False
    current_assistant_id: str = ""
    tool_item_ids: dict[str, str] = mutobj.field(default_factory=dict)
    turn_input_tokens: int = 0
    turn_output_tokens: int = 0
    turn_started_at: float = 0.0
    total_cost: float = 0.0
    handle_model_change: Any = None
    handle_send: Any = None
    handle_cancel: Any = None
    handle_agent_event: Any = None
    subscription: Any = None


def _cext(self: Conversation) -> ConversationExt:
    return ConversationExt.get_or_create(self)


# ── 工具函数 ──────────────────────────────────────────────────────

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


# ── 路由解析 / 构造 ───────────────────────────────────────────
#
# route 是裸字符串，遵循 "<page>" 或 "<page>/<sub>" 的扁平约定。
# URL hash 形如 "#/<page>/<sub>"，前缀 ``#/`` 给未来 ``#/history``
# 等顶级页面留扩展空间。

def _parse_hash(hash_value: str) -> str:
    """hash → route。``"#/"`` / ``""`` → ``""``；``"#/settings/llm"`` → ``"settings/llm"``。"""
    return hash_value.lstrip("#").lstrip("/")


def _hash_for_route(route: str) -> str:
    """route → hash。"""
    return f"#/{route}" if route else "#/"


# ── @impl: __init__ ──────────────────────────────────────────────


@mutagent.impl(Conversation.__init__)
def __init__(self: Conversation, *, agent: Any, app: Any = None) -> None:
    super(Conversation, self).__init__()
    ext = _cext(self)
    self.app = app
    self.agent = agent
    ext.items = []
    ext.message_list = MessageList(items=ext.items)
    ext.message_list.id = "message-list"
    self.models = agent.list_models()
    self.current_model = _resolve_current_model_name(agent, self.models)
    self.status = "idle"
    self.is_busy = False
    ext.cancel_requested = False
    ext.current_assistant_id = ""
    ext.tool_item_ids = {}
    ext.turn_input_tokens = 0
    ext.turn_output_tokens = 0
    ext.turn_started_at = 0.0
    ext.total_cost = 0.0
    # 路由：单一真相源。"" = 对话；"settings" / "settings/<id>" = 设置页
    self.current_route = ""
    # 模块级函数绑定到 self（Conversation 类未声明这些为方法，手动用 partial 绑）
    ext.handle_model_change = partial(handle_model_change, self)
    ext.handle_send = partial(handle_send, self)
    ext.handle_cancel = partial(handle_cancel, self)
    ext.handle_agent_event = partial(handle_agent_event, self)
    ext.status_bar = AgentStatusBar(
        status=self.status,
    )
    ext.status_bar.id = "agent-status-bar"
    ext.chat_input = ChatInput(
        on_send=ext.handle_send,
        on_cancel=ext.handle_cancel,
    )
    ext.chat_input.id = "chat-input"
    ext.chat_input.conversation = self
    self.refresh_models = partial(_refresh_models_from_config, self)
    # SettingsPage：注入两个 request 回调，由它请求 Conversation 切换路由
    ext.settings_page = SettingsPage(
        app=app,
        agent=agent,
        on_models_changed=self.refresh_models,
        on_request_close=partial(_on_settings_request_close, self),
        on_request_navigate=partial(_on_settings_request_navigate, self),
    )
    ext.toolbar = ActionToolbar(
        id="conversation-toolbar",
        categories=["mutagent.conversation.toolbar"],
        context=ActionContext(
            data={"conversation": self},
        ),
        label_mode="auto",
    )
    ext.subscription = agent.subscribe(ext.handle_agent_event)


# ── 回调 helpers ────────────────────────────────────────────────


async def _on_settings_request_close(self: Conversation) -> None:
    """SettingsPage 请求关闭设置页 — 走 navigate_to 让路由 + URL 同步。"""
    await self.navigate_to("")


async def _on_settings_request_navigate(self: Conversation, route: str) -> None:
    """SettingsPage 请求切到指定路由（菜单点击）— 同上。"""
    await self.navigate_to(route)


# ── 内部 helper 函数 ────────────────────────────────────────────


def _refresh_shell(self: Conversation) -> None:
    ext = _cext(self)
    ext.status_bar.status = self.status
    ext.status_bar.input_tokens = ext.turn_input_tokens
    ext.status_bar.output_tokens = ext.turn_output_tokens
    ext.status_bar.total_cost = ext.total_cost
    # context + cache — synced from AgentContext (defensive: may be absent in tests)
    ctx = getattr(self.agent, "context", None)
    if ctx is not None:
        ext.status_bar.context_used = ctx.get_context_used()
        ctx_pct = ctx.get_context_percent()
        ext.status_bar.context_percent = ctx_pct if ctx_pct is not None else 0.0
        ext.status_bar.context_total = ctx.context_window
        ext.status_bar.cache_read_tokens = ctx.get_cache_read_tokens()
        ext.status_bar.cache_write_tokens = ctx.get_cache_write_tokens()
    ext.chat_input.disabled = False
    ext.chat_input.is_busy = self.is_busy
    ext.toolbar.context = ActionContext(
        data={"conversation": self},
    )
    ext.status_bar.invalidate()
    ext.chat_input.invalidate()
    ext.toolbar.invalidate()


def _append_item(self: Conversation, item: Any) -> None:
    ext = _cext(self)
    ext.items.append(item)
    ext.message_list.refresh()


def _touch_item(self: Conversation, item_id: str) -> None:
    ext = _cext(self)
    ext.message_list.invalidate_item(item_id)


def _find_item(self: Conversation, item_id: str) -> Any | None:
    ext = _cext(self)
    for item in ext.items:
        if getattr(item, "id", "") == item_id:
            return item
    return None


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


async def handle_send(self: Conversation, text: str) -> None:
    ext = _cext(self)
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
    ext.cancel_requested = False
    ext.current_assistant_id = ""
    ext.tool_item_ids = {}
    ext.turn_input_tokens = 0
    ext.turn_output_tokens = 0
    ext.turn_started_at = time.time()
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


async def handle_cancel(self: Conversation) -> None:
    ext = _cext(self)
    if self.agent.cancel():
        logger.info("Conversation cancel requested")
        ext.cancel_requested = True
        self.status = "cancelling"
        _refresh_shell(self)
        self.invalidate()


async def handle_model_change(self: Conversation, name: str) -> None:
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
    ext = _cext(self)
    item = None
    if ext.current_assistant_id:
        found = _find_item(self, ext.current_assistant_id)
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
        ext.current_assistant_id = item.id
        _append_item(self, item)
    return item


async def handle_agent_event(self: Conversation, event: StreamEvent) -> None:
    ext = _cext(self)
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
        ext.current_assistant_id = item.id
        _append_item(self, item)
        self.status = "thinking"
    elif event.type == "text_delta" and event.text:
        item = _ensure_current_assistant(self, event)
        item.text += event.text
        _touch_item(self, item.id)
    elif event.type == "tool_exec_start" and event.tool_call is not None:
        logger.info("Tool execution started: %s", event.tool_call.name)
        tool_call = event.tool_call
        item_id = ext.tool_item_ids.get(tool_call.id)
        if item_id is None:
            item_id = f"tool-{tool_call.id or time.time_ns()}"
            ext.tool_item_ids[tool_call.id] = item_id
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
        item_id = ext.tool_item_ids.get(tool_call.id)
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
        ext.turn_input_tokens += response.message.input_tokens
        ext.turn_output_tokens += response.message.output_tokens
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
        if ext.cancel_requested:
            assistant_item = _find_item(self, ext.current_assistant_id)
            if isinstance(assistant_item, AssistantTextItem):
                if "[interrupted]" not in assistant_item.text:
                    assistant_item.text = (
                        assistant_item.text.rstrip() + "\n\n[interrupted]"
                    ).strip()
                    _touch_item(self, assistant_item.id)
        duration = max(0.0, time.time() - ext.turn_started_at) if ext.turn_started_at else 0.0
        _append_item(
            self,
            TurnSeparatorItem(
                id=f"turn-{event.turn_id or time.time_ns()}",
                kind="turn_done",
                turn_id=event.turn_id,
                duration=duration,
                input_tokens=ext.turn_input_tokens,
                output_tokens=ext.turn_output_tokens,
            ),
        )
        self.is_busy = False
        self.status = "idle"
        ext.cancel_requested = False
        ext.current_assistant_id = ""
        ext.tool_item_ids = {}
    _refresh_shell(self)
    self.invalidate()


# ── 路由 / 双模式 render / 事件 ──────────────────────────────


async def _apply_route(self: Conversation, route: str) -> None:
    """按 prev/new 是否在 settings 内，分四象限处理 panel 生命周期。

    最后写入 ``self.current_route``。``invalidate`` 由调用方负责。
    """
    ext = _cext(self)
    prev = self.current_route
    prev_in_settings = prev.startswith("settings")
    new_in_settings = route.startswith("settings")

    if new_in_settings:
        # 解析目标 panel_id；空则用默认（SettingsPage.activate 内部兜底）
        new_panel_id = ""
        if route.startswith("settings/"):
            new_panel_id = route[len("settings/"):]
        # 从 settings → settings 同模式切换：activate 内部已处理 close 旧 panel
        await ext.settings_page.activate(new_panel_id)
    elif prev_in_settings:
        # 离开 settings → close 当前 panel
        await ext.settings_page.deactivate()
    # 其他象限（conversation → conversation）无需做 panel 生命周期处理

    self.current_route = route


@mutagent.impl(Conversation.navigate_to)
async def navigate_to(self: Conversation, route: str) -> None:
    """编程式导航 — 后端主动改路由 + 同步 URL hash。

    防循环：``mutgui.setHash`` 走 ``pushState/replaceState``，W3C 规定
    不触发 ``hashchange`` 事件 → 不会回传后端 → 不会循环。
    """
    if route == self.current_route:
        return
    await _apply_route(self, route)
    await self.broadcast_command("mutgui.setHash", hash=_hash_for_route(route))
    self.invalidate()


@mutagent.impl(Conversation.on_hash_change)
async def on_hash_change(self: Conversation, hash_value: str) -> None:
    """浏览器侧 hash 变化（back / forward / 手输 / 初始握手）→ 同步状态。

    **不**回发 setHash —— URL 已经在前端是新值。
    """
    route = _parse_hash(hash_value)
    if route == self.current_route:
        return
    await _apply_route(self, route)
    await self.broadcast_command("mutgui.setHash", hash=hash_value)
    self.invalidate()


@mutagent.impl(Conversation.on_event)
async def on_event(self: Conversation, event: Event) -> bool:
    """拦截 root 级 ``$hashchange`` 系统事件，其他事件走默认子组件分发。

    ``source: []`` 才到得了 root View 的 ``on_event``（``component_id == ""``）。
    """
    if event.component_id == "" and event.name == "$hashchange":
        new_hash = ""
        if isinstance(event.data, dict):
            new_hash = str(event.data.get("hash", "") or "")
        await on_hash_change(self, new_hash)
        return True
    return await super(Conversation, self).on_event(event)


@mutagent.impl(Conversation.render)
def render(self: Conversation) -> ViewBlock:
    ext = _cext(self)
    _refresh_shell(self)
    in_settings = self.current_route.startswith("settings")
    if in_settings:
        children: list[Any] = [ext.settings_page]
    else:
        children = [
            {
                "$component": "div",
                "$id": "toolbar-shell",
                "style": {"padding": "8px 12px"},
                "$children": [ext.toolbar],
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
                "$children": [ext.message_list],
            },
            ext.chat_input,
        ]
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
            "$children": children,
        }
    ])
