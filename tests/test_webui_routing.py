"""Tests for Conversation 路由层 + SettingsPage 状态机 + 双模式 render + 防循环。

对应设计文档 ``docs/specifications/feature-settings-page-routing.md`` 的「测试」清单。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mutagent.config import Disposable
from mutagent.webui import Conversation, SettingsPage
from mutagent.webui._conversation_impl import _parse_hash, _hash_for_route
from mutgui.events import Event


class _DummyAgent:
    def __init__(self) -> None:
        self.llm = type("LLM", (), {"model": "model-alpha"})()
        self.config = type(
            "Config",
            (),
            {"get": staticmethod(lambda name, default=None, **_: default)},
        )()

    def list_models(self) -> list[dict[str, str]]:
        return [{"name": "alpha", "model_id": "model-alpha"}]

    def subscribe(self, callback):
        self._callback = callback
        return Disposable()


# ── route 解析 / 构造 ─────────────────────────────────────────


@pytest.mark.parametrize(
    "hash_value,expected",
    [
        ("", ""),
        ("#", ""),
        ("#/", ""),
        ("#/settings", "settings"),
        ("#/settings/llm", "settings/llm"),
        ("#/settings/mcp", "settings/mcp"),
    ],
)
def test_parse_hash_handles_all_forms(hash_value: str, expected: str) -> None:
    assert _parse_hash(hash_value) == expected


@pytest.mark.parametrize(
    "route,expected",
    [
        ("", "#/"),
        ("settings", "#/settings"),
        ("settings/llm", "#/settings/llm"),
    ],
)
def test_hash_for_route_round_trip(route: str, expected: str) -> None:
    assert _hash_for_route(route) == expected
    # 双向映射闭合：parse(hash_for(route)) == route
    assert _parse_hash(_hash_for_route(route)) == route


# ── navigate_to / on_hash_change 状态机 ──────────────────────────


def _make_conversation() -> Conversation:
    return Conversation(agent=_DummyAgent())


def _patch_commands(conv: Conversation) -> list[tuple[str, dict[str, Any]]]:
    """劫持 send_command + broadcast_command 记录调用，绕开 ViewPort 上下文要求。"""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _record(name: str, /, **args: Any) -> None:
        calls.append((name, args))

    conv.send_command = _record  # type: ignore[method-assign]
    conv.broadcast_command = _record  # type: ignore[method-assign]
    return calls


def _patch_panel_lifecycle(page: SettingsPage) -> list[tuple[str, str]]:
    """监听全部 panel 的 on_open / on_close 调用顺序。"""
    events: list[tuple[str, str]] = []
    for panel_id, panel in page._panels.items():
        def _make_open(pid: str) -> Any:
            def _on_open() -> None:
                events.append(("open", pid))
            return _on_open

        def _make_close(pid: str) -> Any:
            def _on_close() -> None:
                events.append(("close", pid))
            return _on_close

        panel.on_open = _make_open(panel_id)  # type: ignore[method-assign]
        panel.on_close = _make_close(panel_id)  # type: ignore[method-assign]
    return events


def test_navigate_to_same_route_is_noop() -> None:
    conv = _make_conversation()
    calls = _patch_commands(conv)
    events = _patch_panel_lifecycle(conv.settings_page)

    asyncio.run(conv.navigate_to(""))

    assert conv.current_route == ""
    assert calls == []
    assert events == []


def test_navigate_to_settings_activates_default_panel() -> None:
    conv = _make_conversation()
    calls = _patch_commands(conv)
    events = _patch_panel_lifecycle(conv.settings_page)

    asyncio.run(conv.navigate_to("settings"))

    assert conv.current_route == "settings"
    assert conv.settings_page.active_panel_id == "llm"  # 首个 panel
    assert calls == [("mutgui.setHash", {"hash": "#/settings"})]
    assert events == [("open", "llm")]


def test_navigate_to_settings_panel_id_activates_specific_panel() -> None:
    conv = _make_conversation()
    _patch_commands(conv)
    events = _patch_panel_lifecycle(conv.settings_page)

    asyncio.run(conv.navigate_to("settings/mcp"))

    assert conv.current_route == "settings/mcp"
    assert conv.settings_page.active_panel_id == "mcp"
    assert events == [("open", "mcp")]


def test_navigate_between_panels_closes_old_opens_new() -> None:
    conv = _make_conversation()
    _patch_commands(conv)
    events = _patch_panel_lifecycle(conv.settings_page)

    asyncio.run(conv.navigate_to("settings/llm"))
    events.clear()
    asyncio.run(conv.navigate_to("settings/mcp"))

    assert conv.settings_page.active_panel_id == "mcp"
    # 切换严格 close 旧 → open 新
    assert events == [("close", "llm"), ("open", "mcp")]


def test_navigate_back_to_conversation_deactivates_panel() -> None:
    conv = _make_conversation()
    _patch_commands(conv)
    events = _patch_panel_lifecycle(conv.settings_page)

    asyncio.run(conv.navigate_to("settings/mcp"))
    events.clear()
    asyncio.run(conv.navigate_to(""))

    assert conv.current_route == ""
    # 离开 settings → close 当前 panel；active_panel_id 保留作为「上次激活」记忆
    assert events == [("close", "mcp")]
    assert conv.settings_page.active_panel_id == "mcp"


def test_on_hash_change_drives_route_state() -> None:
    conv = _make_conversation()
    calls = _patch_commands(conv)
    events = _patch_panel_lifecycle(conv.settings_page)

    asyncio.run(conv.on_hash_change("#/settings/mcp"))

    assert conv.current_route == "settings/mcp"
    assert conv.settings_page.active_panel_id == "mcp"
    assert events == [("open", "mcp")]
    # on_hash_change 现在广播 setHash 给所有 tab 同步 URL；
    # 防循环由 W3C（pushState 不触发 hashchange）保证，而非后端 silence
    assert calls == [("mutgui.setHash", {"hash": "#/settings/mcp"})]


def test_on_hash_change_same_route_is_noop() -> None:
    conv = _make_conversation()
    asyncio.run(conv.on_hash_change("#/"))  # 与初始 "" 等价
    calls = _patch_commands(conv)
    events = _patch_panel_lifecycle(conv.settings_page)

    asyncio.run(conv.on_hash_change("#/"))

    assert conv.current_route == ""
    assert calls == []
    assert events == []


# ── 双模式 render ────────────────────────────────────────────


def test_settings_mode_render_only_shows_settings_page() -> None:
    conv = _make_conversation()
    _patch_commands(conv)
    asyncio.run(conv.navigate_to("settings/llm"))

    root = conv.render().items[0]
    children = root["$children"]

    # 设置模式：root 只挂 settings_page；toolbar / message_list / chat_input 都不进 wire tree
    assert len(children) == 1
    assert children[0] is conv.settings_page
    assert conv.toolbar not in children
    assert conv.message_list not in children
    assert conv.chat_input not in children


def test_conversation_mode_render_excludes_settings_page() -> None:
    conv = _make_conversation()
    # 默认就是 conversation 模式
    root = conv.render().items[0]
    children = root["$children"]

    # 三个子节点：toolbar-shell / messages-shell / chat_input；不含 settings_page
    assert len(children) == 3
    assert all(child is not conv.settings_page for child in children)


# ── on_event 路由 ────────────────────────────────────────────


def test_on_event_intercepts_root_hashchange() -> None:
    conv = _make_conversation()
    _patch_commands(conv)
    events = _patch_panel_lifecycle(conv.settings_page)

    # 模拟 mutgui 系统事件：source=[], component_id="", name="$hashchange"
    event = Event("", "$hashchange", {"hash": "#/settings/mcp", "cause": "user"})

    consumed = asyncio.run(conv.on_event(event))

    assert consumed is True
    assert conv.current_route == "settings/mcp"
    assert events == [("open", "mcp")]


def test_on_event_falls_through_for_other_events() -> None:
    """非 $hashchange 走 super().on_event() 默认子组件分发（无 handler 时返回 False）。"""
    conv = _make_conversation()

    event = Event("", "click", {})  # 无注册 handler
    consumed = asyncio.run(conv.on_event(event))

    # 没有匹配 handler，走默认实现返回 False
    assert consumed is False


# ── 防循环（W3C 天然行为，但用测试 lock 住后端不会主动回发） ──────────


def test_navigate_to_emits_set_hash_command_once() -> None:
    conv = _make_conversation()
    calls = _patch_commands(conv)

    asyncio.run(conv.navigate_to("settings/llm"))

    assert calls == [("mutgui.setHash", {"hash": "#/settings/llm"})]


def test_on_hash_change_broadcasts_set_hash_to_all_tabs() -> None:
    """手动改 hash / back-forward → 后端广播 setHash 给所有 ViewPort。

    防循环由 W3C 保证（pushState 不触发 hashchange），后端负责广播。
    """
    conv = _make_conversation()
    calls = _patch_commands(conv)

    asyncio.run(conv.on_hash_change("#/settings/llm"))

    assert calls == [("mutgui.setHash", {"hash": "#/settings/llm"})]


def test_on_hash_change_same_route_still_noop() -> None:
    conv = _make_conversation()
    asyncio.run(conv.on_hash_change("#/"))  # 与初始 "" 等价
    calls = _patch_commands(conv)
    events = _patch_panel_lifecycle(conv.settings_page)

    asyncio.run(conv.on_hash_change("#/"))

    # 同 route 被 _apply_route 提前 return，不回发任何命令
    assert calls == []
    assert events == []


# ── 兼容方法：SettingsPage.close() → on_request_close ────────────


def test_settings_page_close_routes_back_to_conversation() -> None:
    """LLMSettingsPanel/_save_all_settings 仍调 ``await view.drawer.close()``，

    新结构下应转发为 Conversation.navigate_to("") 路径，URL 同步回 #/。
    """
    conv = _make_conversation()
    calls = _patch_commands(conv)
    asyncio.run(conv.navigate_to("settings/llm"))
    calls.clear()  # 清掉 navigate_to 自己产生的 setHash

    # 模拟 panel 触发 close
    asyncio.run(conv.settings_page.close())

    assert conv.current_route == ""
    # 走 navigate_to → 发出 setHash 回到 #/
    assert calls == [("mutgui.setHash", {"hash": "#/"})]
