"""Tests for mutagent.webui child-view wiring."""

from __future__ import annotations

import asyncio

from mutagent.webui.messages import MessageList
from mutagent.config import Disposable
from mutagent.webui import Conversation
from mutagent.webui._settings_llm import LLMSettingsPanel, _ANTHROPIC_PROVIDER, _discover_remote_models
from mutagent.webui.chat_input import ChatInput
from mutagent.webui.messages import AssistantMessage, AssistantTextItem
from mutagent.webui._server_impl import _render_root_html


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


def test_conversation_child_views_have_stable_ids():
    conversation = Conversation(agent=_DummyAgent())

    child_ids = {
        conversation.toolbar.id,
        conversation.status_bar.id,
        conversation.message_list.id,
        conversation.chat_input.id,
        conversation.chat_input.toolbar.id,
        conversation.settings_drawer.id,
    }

    assert "" not in child_ids
    assert len(child_ids) == 6


def test_assistant_message_block_renderer_has_id():
    item = AssistantTextItem(id="msg-1", kind="assistant.text", text="hello")
    message = AssistantMessage(item=item)

    assert message._renderer.id == "block-renderer-msg-1"


def test_assistant_message_recreates_renderer_when_text_changes():
    item = AssistantTextItem(id="msg-1", kind="assistant.text", text="")
    message = AssistantMessage(item=item)
    original_renderer = message._renderer

    item.text = "updated"
    message.render()

    assert message._renderer is not original_renderer
    assert message._renderer.id == "block-renderer-msg-1"
    assert message._renderer.text == "updated"


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
    conversation = Conversation(agent=_DummyAgent())

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


def test_settings_drawer_renders_inline_in_conversation():
    """SettingsDrawer is a View child — rendered as direct child in conversation tree."""
    conversation = Conversation(agent=_DummyAgent())
    root = conversation.render().items[0]
    # settings_drawer is the 4th child (after toolbar-shell, messages-shell, chat_input)
    drawer_child = root["$children"][3]
    assert drawer_child is conversation.settings_drawer


def test_settings_panel_list_page_only_offers_anthropic_and_openai():
    panel = LLMSettingsPanel(app=object(), agent=_DummyAgent())

    root = panel.render().items[0]
    children = root["$children"]

    add_row = next(item for item in children if item.get("$id") == "provider-add-actions")
    button_ids = [item["$id"] for item in add_row["$children"]]

    assert button_ids == ["add-anthropic", "add-openai"]
    assert all(item["$id"] != "kind-item" for item in children)


def test_settings_panel_provider_list_shows_name_type_and_full_models():
    panel = LLMSettingsPanel(app=object(), agent=_DummyAgent())
    panel._drafts = {
        "volcengine": {
            "name": "volcengine",
            "provider": _ANTHROPIC_PROVIDER,
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
    panel = LLMSettingsPanel(app=object(), agent=_DummyAgent())
    panel.current_step = "edit"
    panel.provider_name = "volcengine"
    panel.provider_type = _ANTHROPIC_PROVIDER
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
    assert type_item["$children"][0]["value"] == _ANTHROPIC_PROVIDER
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
