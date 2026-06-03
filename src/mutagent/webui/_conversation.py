"""Conversation root view and Agent ↔ View adapter — Declaration + Implementation."""

from __future__ import annotations

import time
from functools import partial
import logging
from typing import Any, TYPE_CHECKING

import mutobj
from mutagent.core._context_impl import ContextRuntime
from mutagent.core.messages import (
    Message,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from mutagent.core.llm import LLMApiClient
from ._chat_input import ChatInput
from ._messages import (
    AssistantErrorItem,
    AssistantTextItem,
    ChatItem,
    MessageList,
    ToolCallItem,
    TurnSeparatorItem,
    UserTextItem,
)
from ._session_page import ResumeSessionPage
from ._toolbar import AgentStatusBar, AgentToolbar
from ._settings_page import SettingsPage
from mutgui import Callback, View, ViewBlock
from mutgui.events import Event

if TYPE_CHECKING:
    from mutagent.core.agent import Agent, CancelFn
    from mutagent.core.context import AgentContext
    from mutagent.core.session import AgentSession
    from mutagent.app.app import App


logger = logging.getLogger(__name__)


class Conversation(View):
    """Root conversation shell for the built-in WebUI.

    路由权威集中在此类。``current_route`` 是单一真相源：

    - ``""``                — 对话主页
    - ``"resume"``          — 历史 session 恢复页
    - ``"settings"``        — 设置页（默认 panel）
    - ``"settings/<id>"``   — 设置页的指定 panel

    URL hash 由 mutgui 的 ``mutgui.setHash`` 命令 + ``$hashchange`` 事件
    通道双向同步——本类既不直接读写 ``window.location.hash``，也不依赖
    任何防循环标记位（W3C 规定 ``pushState`` 不触发 ``hashchange``，
    天然无回环）。
    """

    current_route: str = ""
    agent: Agent
    app: App
    current_model: str
    status: str = "idle"
    is_busy: bool = False
    config_dirty: bool = False
    message_list: MessageList
    status_bar: AgentStatusBar
    chat_input: ChatInput
    toolbar: AgentToolbar
    resume_page: ResumeSessionPage
    settings_page: SettingsPage
    session: AgentSession

    def __init__(self, *, agent: Agent, app: App) -> None: ...

    def render(self) -> ViewBlock: ...

    async def navigate_to(self, route: str) -> None: ...
    async def on_hash_change(self, hash_value: str) -> None: ...
    async def on_event(self, event: Event) -> bool: ...

    def change_model(self, model_name: str) -> None: ...
    async def send(self, text: str) -> None: ...
    async def cancel(self) -> None: ...


# ── Extension ──────────────────────────────────────────────────────

class ConversationExt(mutobj.Extension[Conversation]):
    """Conversation 的运行时私有状态。"""
    cancel_requested: bool = False
    current_assistant_id: str = ""
    tool_item_ids: dict[str, str] = mutobj.field(default_factory=dict)
    turn_input_tokens: int = 0
    turn_output_tokens: int = 0
    turn_started_at: float = 0.0
    subscription: CancelFn | None = None
    pending_model: str = ""

def _cext(self: Conversation) -> ConversationExt:
    return ConversationExt.get_or_create(self)


# ── 工具函数 ──────────────────────────────────────────────────────

def _now_item_id(prefix: str) -> str:
    return f"{prefix}-{time.time_ns()}"


def _extract_text(message: Message) -> str:
    return "".join(
        block.text for block in message.blocks if isinstance(block, TextBlock)
    )


def _find_last_assistant(items: list[Any]) -> AssistantTextItem | None:
    for item in reversed(items):
        if isinstance(item, AssistantTextItem):
            return item
    return None


def _reset_context_usage(context: AgentContext) -> None:
    if context is None:
        return
    rt = ContextRuntime.get_or_create(context)
    rt.total_input_tokens = 0
    rt.total_output_tokens = 0
    rt.total_cache_read_tokens = 0
    rt.total_cache_write_tokens = 0


def _replace_items(self: Conversation, items: list[Any]) -> None:
    self.message_list.replace_items(items)


def _message_has_user_text(message: Message) -> bool:
    return message.role == "user" and any(
        isinstance(block, TextBlock) and block.text
        for block in message.blocks
    )


def _message_text(message: Message) -> str:
    return "".join(
        block.text for block in message.blocks
        if isinstance(block, TextBlock) and block.text
    )


def _assistant_turn_ends(messages: list[Message], index: int) -> bool:
    if messages[index].role != "assistant":
        return False
    if index == len(messages) - 1:
        return True
    return _message_has_user_text(messages[index + 1])


def _rebuild_items_from_messages(messages: list[Message]) -> list[ChatItem]:
    items: list[ChatItem] = []
    tool_item_ids: dict[str, str] = {}
    turn_input_tokens = 0
    turn_output_tokens = 0
    turn_duration = 0.0

    for index, message in enumerate(messages):
        text = _message_text(message)
        if message.role == "user":
            if text:
                items.append(UserTextItem(
                    id=message.id or _now_item_id("user"),
                    kind="user.text",
                    text=text,
                    timestamp=message.timestamp,
                ))
                turn_input_tokens = 0
                turn_output_tokens = 0
                turn_duration = 0.0
            for block in message.blocks:
                if not isinstance(block, ToolResultBlock):
                    continue
                item_id = tool_item_ids.get(block.tool_use_id)
                tool_item = next(
                    (
                        existing for existing in items
                        if isinstance(existing, ToolCallItem) and existing.id == item_id
                    ),
                    None,
                )
                if tool_item is None:
                    tool_item = ToolCallItem(
                        id=item_id or f"tool-{block.tool_use_id or time.time_ns()}",
                        kind="assistant.tool_group",
                        tool_id=block.tool_use_id,
                        name=block.tool_name or "(unknown tool)",
                        status="error" if block.is_error else "success",
                        result_text=block.content,
                        is_error=block.is_error,
                        duration=block.duration,
                    )
                    items.append(tool_item)
                else:
                    tool_item.status = "error" if block.is_error else "success"
                    tool_item.result_text = block.content
                    tool_item.is_error = block.is_error
                    tool_item.duration = block.duration
        elif message.role == "assistant":
            turn_input_tokens += message.input_tokens
            turn_output_tokens += message.output_tokens
            turn_duration += message.duration
            if text:
                items.append(AssistantTextItem(
                    id=message.id or _now_item_id("assistant"),
                    kind="assistant.text",
                    text=text,
                    model=message.model,
                    timestamp=message.timestamp,
                    duration=message.duration,
                    input_tokens=message.input_tokens,
                    output_tokens=message.output_tokens,
                ))
            for block in message.blocks:
                if not isinstance(block, ToolUseBlock):
                    continue
                item_id = f"tool-{block.id or time.time_ns()}"
                tool_item_ids[block.id] = item_id
                items.append(ToolCallItem(
                    id=item_id,
                    kind="assistant.tool_group",
                    tool_id=block.id,
                    name=block.name,
                    input_text=str(block.input),
                    status="pending",
                ))
            if _assistant_turn_ends(messages, index):
                items.append(TurnSeparatorItem(
                    id=f"turn-{message.id or time.time_ns()}",
                    kind="turn_done",
                    turn_id=message.id,
                    duration=turn_duration,
                    input_tokens=turn_input_tokens,
                    output_tokens=turn_output_tokens,
                ))
                turn_input_tokens = 0
                turn_output_tokens = 0
                turn_duration = 0.0
                tool_item_ids = {}
    return items


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


def _normalize_route(route: str) -> str:
    normalized = route.strip().strip("/")
    if not normalized:
        return ""
    if normalized == "resume" or normalized.startswith("resume/"):
        return "resume"
    if normalized == "settings" or normalized.startswith("settings/"):
        return normalized
    return ""


# ── @impl: __init__ ──────────────────────────────────────────────


@mutobj.impl(Conversation.__init__)
def conversation_init__(self: Conversation, *, agent: Agent, app: App) -> None:
    super(Conversation, self).__init__()
    ext = _cext(self)
    self.app = app
    self.agent = agent
    self.message_list = MessageList()
    self.current_model = agent.model
    # 模块级函数绑定到 self（Conversation 类未声明这些为方法，手动用 partial 绑）
    self.status_bar = AgentStatusBar()
    self.chat_input = ChatInput(conversation=self)
    self.resume_page = ResumeSessionPage(conversation=self)
    self.settings_page = SettingsPage(conversation=self)
    self.toolbar = AgentToolbar(conversation=self)
    from ._session_page import _start_session
    _start_session(self)
    ext.subscription = agent.subscribe(partial(handle_agent_event, self))


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
    self.status_bar.status = self.status
    self.status_bar.input_tokens = ext.turn_input_tokens
    self.status_bar.output_tokens = ext.turn_output_tokens
    ctx = self.agent.context
    rt = ContextRuntime.get_or_create(ctx)
    self.status_bar.context_used = rt.total_input_tokens
    cw = self.agent.llm.context_window or 0
    self.status_bar.context_percent = rt.total_input_tokens / cw if cw else 0.0
    self.status_bar.context_total = cw
    self.status_bar.cache_read_tokens = rt.total_cache_read_tokens
    self.status_bar.cache_write_tokens = rt.total_cache_write_tokens
    self.status_bar.invalidate()
    self.chat_input.invalidate()
    self.toolbar.invalidate()


def _reset_runtime_state(self: Conversation) -> None:
    ext = _cext(self)
    self.is_busy = False
    self.status = "idle"
    ext.cancel_requested = False
    ext.current_assistant_id = ""
    ext.tool_item_ids = {}
    ext.turn_input_tokens = 0
    ext.turn_output_tokens = 0
    ext.turn_started_at = 0.0


async def _handle_send(self: Conversation, text: str) -> None:
    ext = _cext(self)
    if ext.pending_model or self.config_dirty:
        model = ext.pending_model or self.agent.model
        if model:
            try:
                spec = self.app.config.resolve_model(model)
                if spec:
                    llm = LLMApiClient.from_spec(spec)
                    self.agent.llm = llm
                    self.agent.model = llm.model_id
            except Exception:
                logger.exception("Failed to rebuild LLM")
        ext.pending_model = ""
        self.config_dirty = False
    if self.is_busy:
        logger.info("Conversation send ignored while busy")
        return
    logger.info("Conversation submit requested (%d chars)", len(text))
    self.message_list.append_item(
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
        self.message_list.append_item(
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
        ext = _cext(self)
        logger.info("Conversation cancel requested")
        ext.cancel_requested = True
        self.status = "cancelling"
        _refresh_shell(self)
        self.invalidate()


def _ensure_current_assistant(self: Conversation, event: StreamEvent) -> AssistantTextItem:
    ext = _cext(self)
    item = None
    if ext.current_assistant_id:
        found = self.message_list.find_item(ext.current_assistant_id)
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
        self.message_list.append_item(item)
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
        self.message_list.append_item(item)
        self.status = "thinking"
    elif event.type == "text_delta" and event.text:
        item = _ensure_current_assistant(self, event)
        item.text += event.text
        self.message_list.invalidate_item(item.id)
    elif event.type == "tool_exec_start" and isinstance(event.tool_call, ToolUseBlock):
        logger.info("Tool execution started: %s", event.tool_call.name)
        tool_call = event.tool_call
        item_id = ext.tool_item_ids.get(tool_call.id)
        if item_id is None:
            item_id = f"tool-{tool_call.id or time.time_ns()}"
            ext.tool_item_ids[tool_call.id] = item_id
            self.message_list.append_item(
                ToolCallItem(
                    id=item_id,
                    kind="assistant.tool_group",
                    tool_id=tool_call.id,
                    name=tool_call.name,
                    input_text=str(tool_call.input),
                    status="pending",
                ),
            )
        tool_item = self.message_list.find_item(item_id)
        if isinstance(tool_item, ToolCallItem):
            tool_item.status = "pending"
            tool_item.input_text = str(tool_call.input)
            self.message_list.invalidate_item(item_id)
        self.status = "tool_calling"
    elif event.type == "tool_exec_end" and isinstance(event.tool_call, ToolResultBlock):
        logger.info("Tool execution finished: %s", event.tool_call.tool_name)
        tool_call = event.tool_call
        item_id = ext.tool_item_ids.get(tool_call.tool_use_id)
        tool_item = self.message_list.find_item(item_id) if item_id else None
        if isinstance(tool_item, ToolCallItem):
            tool_item.status = "error" if tool_call.is_error else "success"
            tool_item.result_text = tool_call.content
            tool_item.duration = tool_call.duration
            tool_item.is_error = tool_call.is_error
            self.message_list.invalidate_item(tool_item.id)
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
        self.message_list.invalidate_item(assistant_item.id)
    elif event.type == "error":
        logger.error("Conversation received error event: %s", event.error or "Unknown error")
        self.message_list.append_item(
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
            assistant_item = self.message_list.find_item(ext.current_assistant_id)
            if isinstance(assistant_item, AssistantTextItem):
                if "[interrupted]" not in assistant_item.text:
                    assistant_item.text = (
                        assistant_item.text.rstrip() + "\n\n[interrupted]"
                    ).strip()
                    self.message_list.invalidate_item(assistant_item.id)
        duration = max(0.0, time.time() - ext.turn_started_at) if ext.turn_started_at else 0.0
        self.message_list.append_item(
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
        try:
            self.session.sync(self.agent.context)
        except Exception as exc:
            logger.exception("Conversation session sync failed")
            self.message_list.append_item(
                AssistantErrorItem(
                    id=_now_item_id("error"),
                    kind="assistant.error",
                    error=str(exc),
                    timestamp=time.time(),
                ),
            )
    _refresh_shell(self)
    self.invalidate()


# ── 路由 / 双模式 render / 事件 ──────────────────────────────


async def _apply_route(self: Conversation, route: str) -> None:
    """按 prev/new 是否在 settings 内，分四象限处理 panel 生命周期。

    最后写入 ``self.current_route``。``invalidate`` 由调用方负责。
    """
    route = _normalize_route(route)
    prev = self.current_route
    prev_in_settings = prev.startswith("settings")
    new_in_settings = route.startswith("settings")
    prev_in_resume = prev == "resume"
    new_in_resume = route == "resume"

    if new_in_settings:
        # 解析目标 panel_id；空则用默认（SettingsPage.activate 内部兜底）
        new_panel_id = ""
        if route.startswith("settings/"):
            new_panel_id = route[len("settings/"):]
        # 从 settings → settings 同模式切换：activate 内部已处理 close 旧 panel
        await self.settings_page.activate(new_panel_id)
    elif prev_in_settings:
        # 离开 settings → close 当前 panel
        await self.settings_page.deactivate()
    if new_in_resume:
        await self.resume_page.activate()
    elif prev_in_resume:
        pass
    # 其他象限（conversation → conversation）无需做 panel 生命周期处理

    self.current_route = route


@mutobj.impl(Conversation.navigate_to)
async def conversation_navigate_to(self: Conversation, route: str) -> None:
    """编程式导航 — 后端主动改路由 + 同步 URL hash。

    防循环：``mutgui.setHash`` 走 ``pushState/replaceState``，W3C 规定
    不触发 ``hashchange`` 事件 → 不会回传后端 → 不会循环。
    """
    route = _normalize_route(route)
    if route == self.current_route:
        return
    await _apply_route(self, route)
    await self.broadcast_command("mutgui.setHash", hash=_hash_for_route(route))
    self.invalidate()


@mutobj.impl(Conversation.on_hash_change)
async def conversation_on_hash_change(self: Conversation, hash_value: str) -> None:
    """浏览器侧 hash 变化（back / forward / 手输 / 初始握手）→ 同步状态。

    **不**回发 setHash —— URL 已经在前端是新值。
    """
    route = _normalize_route(_parse_hash(hash_value))
    if route == self.current_route:
        return
    await _apply_route(self, route)
    await self.broadcast_command("mutgui.setHash", hash=_hash_for_route(route))
    self.invalidate()


@mutobj.impl(Conversation.on_event)
async def conversation_on_event(self: Conversation, event: Event) -> bool:
    """拦截 root 级 ``$hashchange`` 系统事件，其他事件走默认子组件分发。

    ``source: []`` 才到得了 root View 的 ``on_event``（``component_id == ""``）。
    """
    if event.component_id == "" and event.name == "$hashchange":
        new_hash = ""
        if isinstance(event.kwargs, dict):
            new_hash = str(event.kwargs.get("hash", "") or "")
        await conversation_on_hash_change(self, new_hash)
        return True
    return await super(Conversation, self).on_event(event)


@mutobj.impl(Conversation.render)
def conversation_render(self: Conversation) -> ViewBlock:
    _refresh_shell(self)
    in_settings = self.current_route.startswith("settings")
    if in_settings:
        children: list[Any] = [self.settings_page]
    elif self.current_route == "resume":
        children = [self.resume_page]
    else:
        children = [
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


@mutobj.impl(Conversation.change_model)
def conversation_change_model(self: Conversation, model_name: str) -> None:
    logger.info("Conversation model change requested: %s", model_name)
    ext = _cext(self)
    ext.pending_model = model_name
    self.current_model = model_name
    _refresh_shell(self)
    self.invalidate()


@mutobj.impl(Conversation.send)
async def conversation_send(self: Conversation, text: str) -> None:
    await _handle_send(self, text)


@mutobj.impl(Conversation.cancel)
async def conversation_cancel(self: Conversation) -> None:
    await _handle_cancel(self)