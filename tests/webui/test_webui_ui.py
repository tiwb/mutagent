"""Tests for mutagent.webui child-view wiring."""

from __future__ import annotations

import asyncio

from mutagent.webui.messages import MessageList
from mutagent.webui._conversation_impl import _cext
from mutagent.webui._messages_impl import _aext
from mutagent.webui import Conversation
from mutagent.webui._settings_llm import LLMSettingsPanel, _ANTHROPIC_API_TYPE, _discover_remote_models
from mutagent.webui.chat_input import ChatInput
from mutagent.webui.messages import AssistantMessage, AssistantTextItem
from mutagent.webui._server_impl import _render_root_html
from mutagent.app.config import Config


class _DummyAgent:
    def __init__(self) -> None:
        self.llm = type("LLM", (), {"model": "model-alpha"})()
        self.config = Config()
        self.config.load_from_dict({})

    def list_models(self) -> list[dict[str, str]]:
        return [{"name": "alpha", "model_id": "model-alpha"}]

    def subscribe(self, callback):
        self._callback = callback
        return lambda: None


def test_conversation_child_views_have_stable_ids():
    agent = _DummyAgent()
    conversation = Conversation(agent=agent, app=agent)

    ext = _cext(conversation)
    child_ids = {
        ext.toolbar.id,
        ext.status_bar.id,
        ext.message_list.id,
        ext.chat_input.id,
        ext.chat_input.toolbar.id,
        ext.settings_page.id,
    }

    assert "" not in child_ids
    assert len(child_ids) == 6


def test_assistant_message_block_renderer_has_id():
    item = AssistantTextItem(id="msg-1", kind="assistant.text", text="hello")
    message = AssistantMessage(item=item)

    assert _aext(message).renderer.id == "block-renderer-msg-1"


def test_assistant_message_recreates_renderer_when_text_changes():
    item = AssistantTextItem(id="msg-1", kind="assistant.text", text="")
    message = AssistantMessage(item=item)
    original_renderer = _aext(message).renderer

    item.text = "updated"
    message.render()

    assert _aext(message).renderer is not original_renderer
    assert _aext(message).renderer.id == "block-renderer-msg-1"
    assert _aext(message).renderer.text == "updated"


def test_message_list_keeps_passed_empty_list_reference():
    items: list[object] = []
    message_list = MessageList(items=items)

    assert message_list.items is items


def test_message_list_shell_is_flex_scroll_container():
    message_list = MessageList(items=[])

    shell = message_list.render().items[0]

    assert shell["style"]["display"] == "flex"
    assert shell["style"]["flexDirection"] == "column"
    assert shell["style"]["minHeight"] == 0
    assert shell["style"]["overflow"] == "hidden"


def test_conversation_root_is_edge_to_edge_shell():
    agent = _DummyAgent()
    conversation = Conversation(agent=agent, app=agent)

    root = conversation.render().items[0]
    toolbar_shell = root["$children"][0]
    messages_shell = root["$children"][1]

    assert root["style"]["--mutagent-font-size-base"] == "13px"
    assert root["style"]["gap"] == 0
    assert "padding" not in root["style"]
    assert "background" not in root["style"]
    assert toolbar_shell["style"]["padding"] == "8px 12px"
    assert messages_shell["style"]["display"] == "flex"
    assert messages_shell["style"]["overflow"] == "hidden"
    assert "border" not in messages_shell["style"]


def test_chat_input_renders_unified_shell_and_press_enter_handler():
    chat_input = ChatInput(on_send=lambda text: text, on_cancel=lambda: None)

    shell = chat_input.render().items[0]

    assert shell["$component"] == "mutagent.ChatInput"
    assert shell["value"] == ""
    assert shell["sendMode"] == "enter"
    assert shell["disabled"] is False
    assert shell["placeholder"] == "Type a message… (Shift+Enter for newline)"
    assert len(shell["$children"]) == 1
    assert shell["$children"][0].id == "chat-input-toolbar"
    assert "onChange" in shell
    assert "onSubmit" in shell


def test_settings_page_excluded_from_conversation_mode_render():
    """对话模式（current_route == ""）下，root 不包含 SettingsPage。

    双模式架构下，设置页只在 ``current_route.startswith("settings")`` 时参与 wire tree。
    对话模式下三个子节点严格是 toolbar-shell / messages-shell / chat_input View。
    """
    agent = _DummyAgent()
    conversation = Conversation(agent=agent, app=agent)
    assert conversation.current_route == ""

    root = conversation.render().items[0]
    children = root["$children"]

    assert len(children) == 3
    assert children[0]["$id"] == "toolbar-shell"
    assert children[1]["$id"] == "messages-shell"
    assert children[2] is _cext(conversation).chat_input
    # settings_page 不在对话模式的 wire tree 中
    assert all(child is not _cext(conversation).settings_page for child in children)


def test_settings_panel_list_page_only_offers_anthropic_and_openai():
    agent = _DummyAgent()
    panel = LLMSettingsPanel(app=agent, agent=agent)

    root = panel.render().items[0]
    children = root["$children"]

    add_row = next(item for item in children if item.get("$id") == "provider-add-actions")
    button_ids = [item["$id"] for item in add_row["$children"]]

    assert button_ids == ["add-provider"]
    assert all(item["$id"] != "kind-item" for item in children)


def test_settings_panel_provider_list_shows_name_type_and_full_models():
    agent = _DummyAgent()
    panel = LLMSettingsPanel(app=agent, agent=agent)
    panel._drafts = {
        "volcengine": {
            "name": "volcengine",
            "type": _ANTHROPIC_API_TYPE,
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "auth_token": "token",
            "models": [
                "doubao-seed-1-6-thinking-250715",
                "doubao-seed-1-6-flash-250715",
            ],
            "discovered_models": [
                "doubao-seed-1-6-thinking-250715",
                "doubao-seed-1-6-flash-250715",
            ],
        }
    }

    root = panel.render().items[0]
    children = root["$children"]
    provider_list = next(item for item in children if item.get("$id") == "provider-list")
    button = provider_list["$children"][0]
    card = button["$children"][0]
    header, summary = card["$children"]

    assert button["style"]["justifyContent"] == "flex-start"
    assert button["style"]["alignItems"] == "stretch"
    assert card["style"]["width"] == "100%"
    assert header["$id"] == "provider-header-volcengine"
    assert header["$children"][0]["children"] == "volcengine"
    assert header["$children"][1]["children"] == "Anthropic"
    assert summary["children"] == (
        "doubao-seed-1-6-thinking-250715, "
        "doubao-seed-1-6-flash-250715"
    )


def test_settings_panel_edit_page_keeps_discover_button_inline_with_models():
    agent = _DummyAgent()
    panel = LLMSettingsPanel(app=agent, agent=agent)
    panel.current_step = "edit"
    panel.provider_name = "volcengine"
    panel.provider_type = _ANTHROPIC_API_TYPE
    panel.provider_type_label = "Anthropic"
    panel.base_url = "https://ark.cn-beijing.volces.com/api/v3"
    panel.auth_token = "token"
    panel.models = ["doubao-seed-1-6-thinking-250715"]

    root = panel.render().items[0]
    children = root["$children"]
    form = next(item for item in children if item.get("$id") == "settings-edit-form")
    name_item = next(item for item in form["$children"] if item.get("$id") == "key-item")
    type_item = next(item for item in form["$children"] if item.get("$id") == "provider-type-item")
    models_item = next(item for item in form["$children"] if item.get("$id") == "models-item")
    header = models_item["$children"][0]

    assert name_item["label"] == "Provider Name"
    assert type_item["label"] == "Provider Type"
    assert type_item["$children"][0]["value"] == _ANTHROPIC_API_TYPE
    assert header["$id"] == "models-header"
    assert [item["$id"] for item in header["$children"]] == ["models-label-wrap", "discover"]


def test_discover_remote_models_falls_back_to_v1_models():
    calls: list[str] = []

    class _Response:
        def __init__(self, status_code: int, payload: dict) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict:
            return self._payload

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, headers: dict[str, str]):
            calls.append(url)
            if url.endswith("/api/v3/models"):
                return _Response(404, {})
            return _Response(200, {
                "data": [
                    {"id": "doubao-seed-1-6-thinking-250715", "created": 2},
                    {"id": "doubao-seed-1-6-flash-250715", "created": 1},
                ]
            })

    import httpx

    original_client = httpx.AsyncClient
    httpx.AsyncClient = lambda *args, **kwargs: _Client()
    try:
        models = asyncio.run(_discover_remote_models(
            "https://ark.cn-beijing.volces.com/api/v3",
            "token",
            chat_filter=False,
        ))
    finally:
        httpx.AsyncClient = original_client

    assert calls == [
        "https://ark.cn-beijing.volces.com/api/v3/models",
        "https://ark.cn-beijing.volces.com/api/v3/v1/models",
    ]
    assert models == [
        "doubao-seed-1-6-thinking-250715",
        "doubao-seed-1-6-flash-250715",
    ]


def test_webui_root_html_uses_import_map_boot_protocol():
    html = _render_root_html()

    assert '<script type="importmap">' in html
    assert 'data-mutgui-app' in html
    assert 'data-ws-url="/ws"' in html
    assert '@mutagent/ui' in html
    assert 'boot.js' in html
