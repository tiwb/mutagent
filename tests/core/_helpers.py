"""mutagent core 测试公共辅助类."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Callable

from mutagent.core.llm import LLMApiClient
from mutagent.core.messages import Message, StreamEvent, ToolSchema


class MockLLMClient(LLMApiClient):
    """测试用 LLM client，mock_send 字段可注入任意 send 实现。

    继承 LLMApiClient（而非具体 provider），所有构造参数有默认值，
    Anthropic / OpenAI 测试均可复用。

    Usage::

        client = MockLLMClient()
        client.mock_send = my_mock_send
        agent = _make_agent(llm=client)
    """

    mock_send: Callable[..., AsyncGenerator[StreamEvent, None]] | None = None

    async def send(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        prompts: list[Message] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        if self.mock_send is None:
            yield StreamEvent(type="response_done", response=None)  # type: ignore[arg-type]
            return
        async for event in self.mock_send(messages, tools, prompts, stream):
            yield event
