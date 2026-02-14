"""mutagent message models for LLM communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    """An LLM-initiated tool call.

    Attributes:
        id: Unique identifier for this tool call (assigned by the LLM).
        name: Name of the tool to invoke.
        arguments: Dictionary of arguments to pass to the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result of executing a tool call.

    Attributes:
        tool_call_id: ID of the ToolCall this result corresponds to.
        content: The string result content.
        is_error: Whether this result represents an error.
    """

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """A single message in the conversation.

    Attributes:
        role: The role of the message sender ("user", "assistant").
        content: Text content of the message.
        tool_calls: List of tool calls (for assistant messages).
        tool_results: List of tool results (for user messages containing results).
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class ToolSchema:
    """JSON Schema description of a tool for the LLM.

    Attributes:
        name: Tool name.
        description: Human-readable description of what the tool does.
        input_schema: JSON Schema object describing the tool's parameters.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """LLM response wrapper.

    Attributes:
        message: The response message from the LLM.
        stop_reason: Why the LLM stopped ("end_turn", "tool_use", etc.).
        usage: Token usage information.
    """

    message: Message
    stop_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """A single event in a streaming LLM response.

    Attributes:
        type: Event type. One of:
            "text_delta"      - Incremental text from the LLM.
            "tool_use_start"  - LLM begins constructing a tool call.
            "tool_use_delta"  - Incremental JSON for tool call arguments.
            "tool_use_end"    - LLM finished a tool call block.
            "tool_exec_start" - Agent begins executing a tool.
            "tool_exec_end"   - Agent finished executing a tool.
            "response_done"   - One LLM call completed; carries the full Response.
            "error"           - An error occurred.
        text: Text fragment (for text_delta).
        tool_call: Tool call info (for tool_use_start, tool_exec_start).
        tool_json_delta: Partial JSON string (for tool_use_delta).
        tool_result: Tool execution result (for tool_exec_end).
        response: Complete Response object (for response_done).
        error: Error description (for error).
    """

    type: str
    text: str = ""
    tool_call: Optional[ToolCall] = None
    tool_json_delta: str = ""
    tool_result: Optional[ToolResult] = None
    response: Optional[Response] = None
    error: str = ""
