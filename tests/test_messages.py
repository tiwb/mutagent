"""Tests for mutagent message models."""

from mutagent.messages import Message, ToolCall, ToolResult, Response, ToolSchema


class TestToolCall:

    def test_creation(self):
        tc = ToolCall(id="tc_1", name="view_source", arguments={"target": "mutagent"})
        assert tc.id == "tc_1"
        assert tc.name == "view_source"
        assert tc.arguments == {"target": "mutagent"}

    def test_default_arguments(self):
        tc = ToolCall(id="tc_1", name="inspect_module")
        assert tc.arguments == {}

    def test_equality(self):
        tc1 = ToolCall(id="tc_1", name="foo", arguments={"a": 1})
        tc2 = ToolCall(id="tc_1", name="foo", arguments={"a": 1})
        assert tc1 == tc2

    def test_inequality(self):
        tc1 = ToolCall(id="tc_1", name="foo")
        tc2 = ToolCall(id="tc_2", name="foo")
        assert tc1 != tc2


class TestToolResult:

    def test_creation(self):
        tr = ToolResult(tool_call_id="tc_1", content="success")
        assert tr.tool_call_id == "tc_1"
        assert tr.content == "success"
        assert tr.is_error is False

    def test_error_result(self):
        tr = ToolResult(tool_call_id="tc_1", content="failed", is_error=True)
        assert tr.is_error is True

    def test_default_is_error(self):
        tr = ToolResult(tool_call_id="tc_1", content="ok")
        assert tr.is_error is False


class TestMessage:

    def test_user_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls == []
        assert msg.tool_results == []

    def test_assistant_message_with_tool_calls(self):
        tc = ToolCall(id="tc_1", name="view_source", arguments={"target": "mutagent"})
        msg = Message(role="assistant", content="", tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "view_source"

    def test_tool_result_message(self):
        tr = ToolResult(tool_call_id="tc_1", content="source code here")
        msg = Message(role="user", tool_results=[tr])
        assert len(msg.tool_results) == 1

    def test_default_lists_are_independent(self):
        msg1 = Message(role="user")
        msg2 = Message(role="user")
        msg1.tool_calls.append(ToolCall(id="tc_1", name="foo"))
        assert len(msg2.tool_calls) == 0


class TestToolSchema:

    def test_creation(self):
        schema = ToolSchema(
            name="view_source",
            description="View source code.",
            input_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Target path"}
                },
                "required": ["target"],
            },
        )
        assert schema.name == "view_source"
        assert "properties" in schema.input_schema

    def test_default_input_schema(self):
        schema = ToolSchema(name="run_code", description="Execute Python code.")
        assert schema.input_schema == {}


class TestResponse:

    def test_creation(self):
        msg = Message(role="assistant", content="Done.")
        resp = Response(
            message=msg,
            stop_reason="end_turn",
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        assert resp.message.content == "Done."
        assert resp.stop_reason == "end_turn"
        assert resp.usage["input_tokens"] == 100

    def test_default_values(self):
        msg = Message(role="assistant", content="Hello")
        resp = Response(message=msg)
        assert resp.stop_reason == ""
        assert resp.usage == {}

    def test_tool_use_response(self):
        tc = ToolCall(id="tc_1", name="inspect_module", arguments={"module_path": "mutagent"})
        msg = Message(role="assistant", tool_calls=[tc])
        resp = Response(message=msg, stop_reason="tool_use")
        assert resp.stop_reason == "tool_use"
        assert len(resp.message.tool_calls) == 1
