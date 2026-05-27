"""Tests for mutagent message models."""

from mutagent.core.messages import (
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    Message,
    Response,
    TextBlock,
    ThinkingBlock,
    ToolSchema,
    ToolUseBlock,
)


class TestContentBlocks:

    def test_text_block(self):
        b = TextBlock(text="hello")
        assert b.type == "text"
        assert b.text == "hello"

    def test_image_block_base64(self):
        b = ImageBlock(data="abc123", media_type="image/png")
        assert b.type == "image"
        assert b.data == "abc123"
        assert b.url == ""

    def test_image_block_url(self):
        b = ImageBlock(url="https://example.com/img.png")
        assert b.url == "https://example.com/img.png"
        assert b.data == ""

    def test_document_block(self):
        b = DocumentBlock(data="base64pdf", media_type="application/pdf")
        assert b.type == "document"

    def test_thinking_block_visible(self):
        b = ThinkingBlock(thinking="Let me think...", signature="sig123")
        assert b.type == "thinking"
        assert b.thinking == "Let me think..."
        assert b.data == ""

    def test_thinking_block_redacted(self):
        b = ThinkingBlock(data="encrypted_data")
        assert b.data == "encrypted_data"
        assert b.thinking == ""

    def test_tool_use_block(self):
        b = ToolUseBlock(id="tc_1", name="search", input={"q": "test"})
        assert b.type == "tool_use"
        assert b.id == "tc_1"
        assert b.status == ""
        assert b.result == ""
        assert b.is_error is False
        assert b.duration == 0

    def test_tool_use_block_lifecycle(self):
        b = ToolUseBlock(id="tc_1", name="search", input={"q": "test"})
        b.status = "running"
        assert b.status == "running"
        b.status = "done"
        b.result = "found it"
        b.duration = 0.5
        assert b.status == "done"
        assert b.result == "found it"


class TestMessage:

    def test_user_message(self):
        msg = Message(role="user", blocks=[TextBlock(text="Hello")])
        assert msg.role == "user"
        assert len(msg.blocks) == 1
        assert msg.blocks[0].text == "Hello"

    def test_assistant_message_with_tool(self):
        msg = Message(role="assistant", blocks=[
            TextBlock(text="Let me check."),
            ToolUseBlock(id="tc_1", name="search", input={"q": "test"}),
        ])
        assert len(msg.blocks) == 2
        assert isinstance(msg.blocks[1], ToolUseBlock)

    def test_metadata_fields(self):
        msg = Message(
            role="assistant",
            blocks=[TextBlock(text="hi")],
            id="msg_1",
            sender="Agent",
            model="claude-sonnet",
            timestamp=1234.0,
            duration=0.5,
            input_tokens=100,
            output_tokens=50,
        )
        assert msg.id == "msg_1"
        assert msg.sender == "Agent"
        assert msg.model == "claude-sonnet"
        assert msg.input_tokens == 100

    def test_prompt_fields(self):
        msg = Message(
            role="system",
            blocks=[TextBlock(text="You are a helper.")],
            label="base",
            cacheable=True,
            priority=100,
        )
        assert msg.label == "base"
        assert msg.priority == 100
        assert msg.cacheable is True

    def test_default_values(self):
        msg = Message(role="user")
        assert msg.blocks == []
        assert msg.id == ""
        assert msg.timestamp == 0
        assert msg.cacheable is True
        assert msg.priority == 0

    def test_default_lists_are_independent(self):
        msg1 = Message(role="user")
        msg2 = Message(role="user")
        msg1.blocks.append(TextBlock(text="a"))
        assert len(msg2.blocks) == 0

    def test_multimodal_message(self):
        msg = Message(role="user", blocks=[
            TextBlock(text="Look at this:"),
            ImageBlock(data="b64data", media_type="image/png"),
            TextBlock(text="What is it?"),
        ])
        assert len(msg.blocks) == 3


class TestToolSchema:

    def test_creation(self):
        schema = ToolSchema(
            name="search",
            description="Search the web.",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        assert schema.name == "search"
        assert "properties" in schema.input_schema

    def test_default_input_schema(self):
        schema = ToolSchema(name="run", description="Run code.")
        assert schema.input_schema == {}


class TestResponse:

    def test_creation(self):
        msg = Message(role="assistant", blocks=[TextBlock(text="Done.")])
        resp = Response(
            message=msg,
            stop_reason="end_turn",
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        assert resp.stop_reason == "end_turn"
        assert resp.usage["input_tokens"] == 100

    def test_default_values(self):
        msg = Message(role="assistant", blocks=[TextBlock(text="Hi")])
        resp = Response(message=msg)
        assert resp.stop_reason == ""
        assert resp.usage == {}

    def test_tool_use_response(self):
        msg = Message(role="assistant", blocks=[
            ToolUseBlock(id="tc_1", name="inspect", input={"path": "mutagent"}),
        ])
        resp = Response(message=msg, stop_reason="tool_use")
        assert resp.stop_reason == "tool_use"
        tool_blocks = [b for b in resp.message.blocks if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) == 1
