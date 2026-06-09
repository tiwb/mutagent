"""mutagent.core.provider -- LLM provider abstraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncGenerator, ClassVar

from mutobj import Declaration

if TYPE_CHECKING:
    from mutio.codec.json import JsonObject
    from .messages import Message, StreamEvent, ToolSchema



# ---------------------------------------------------------------------------
# LLMApiClient
# ---------------------------------------------------------------------------

class LLMApiClient(Declaration):
    """LLM Api 提供商抽象基类。

    子类通过 mutobj 子类发现机制自动注册。
    配置中指定类路径，resolve_class 自动加载。

    子类需实现 ``send`` 方法。
    """

    api_type: ClassVar[str] = ""
    model_id: str = ""
    context_window: int | None = None


    @staticmethod
    def from_spec(spec: JsonObject) -> LLMApiClient:
        """从模型 spec 创建 provider 实例。

        支持 api_type 短名（``"Anthropic"`` / ``"OpenAI"`` / ``"Copilot"``）。
        不指定时默认使用 AnthropicApiClient。
        """
        ...

    async def send(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        prompts: list[Message] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """发送请求到 LLM 后端，返回流式事件。

        Args:
            messages: 对话历史。
            tools: 可用工具 schema 列表。
            prompts: 系统指令 Message 列表。
            stream: 是否使用 SSE 流式请求。

        Yields:
            StreamEvent 实例。最后一个事件始终为 ``response_done``。
        """
        yield ...  # type: ignore[reportReturnType]


from . import _llm_impl as _llm_impl  # noqa: F401, E402
