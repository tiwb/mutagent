"""mutagent.webui -- built-in WebUI: agent widgets, actions, server, and CLI."""

from .conversation import Conversation
from .settings import SettingsPanel, SettingsDrawer
from .toolbar import AgentStatusBar
from .messages import (
    MessageList,
    ChatItem,
    UserTextItem,
    AssistantTextItem,
    AssistantErrorItem,
    TurnSeparatorItem,
    ToolCallItem,
)
from .tool_call import ToolCallCard
from .blocks import BlockRenderer, ThinkingBlock
from .chat_input import ChatInput
from .server import WebUIServer

__all__ = [
    "Conversation",
    "SettingsPanel",
    "SettingsDrawer",
    "AgentStatusBar",
    "MessageList",
    "ChatItem",
    "UserTextItem",
    "AssistantTextItem",
    "AssistantErrorItem",
    "TurnSeparatorItem",
    "ToolCallItem",
    "ToolCallCard",
    "BlockRenderer",
    "ThinkingBlock",
    "ChatInput",
    "WebUIServer",
]

from . import _actions_impl  # noqa: E402,F401
from . import _settings_llm  # noqa: E402,F401 — ensure SettingsPanel subclasses are registered before SettingsDrawer instantiates
