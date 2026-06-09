"""mutagent.core.tools -- Toolkit base class and ToolSet declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import mutobj

if TYPE_CHECKING:
    from .agent import Agent
    from .messages import ToolResultBlock, ToolSchema, ToolUseBlock


class ToolSet(mutobj.Declaration):
    """Tool set manager for an Agent.

    Manages the available tools for an Agent, providing dynamic
    add/remove/query capabilities. Replaces the static ToolSelector.

    Tools can be registered from object instances (registering their
    public methods) or from individual callables.

    When ``auto_discover`` is True, ToolSet automatically scans mutobj's
    class registry for Toolkit subclasses and registers their public
    methods as tools. This enables the tool evolution workflow: Agent
    creates a Toolkit subclass via define_module, and its methods
    become callable tools immediately.

    Attributes:
        auto_discover: Enable automatic Toolkit subclass discovery.
    """

    auto_discover: bool = False
    agent: Agent | None = None

    def add(self, source: Toolkit, methods: list[str] | None = None) -> None:
        """Add tools from a Toolkit instance.

        Args:
            source: A Toolkit instance whose public methods are registered as tools.
            methods: Specify which method names to register.
                None registers all public methods defined directly on the class
                (not inherited).
        """
        ...

    def remove(self, tool_name: str) -> bool:
        """Remove a tool by name.

        Args:
            tool_name: Name of the tool to remove.

        Returns:
            True if the tool was found and removed, False otherwise.
        """
        ...

    def query(self, tool_name: str) -> ToolSchema | None:
        """Query a tool's schema by name.

        Args:
            tool_name: Name of the tool to query.

        Returns:
            The ToolSchema if found, None otherwise.
        """
        ...

    def get_tools(self) -> list[ToolSchema]:
        """Get all available tool schemas for the LLM API.

        Returns:
            List of ToolSchema objects.
        """
        ...

    async def dispatch(self, tool_call: ToolUseBlock) -> ToolResultBlock:
        """Dispatch a tool call and return a result block.

        Args:
            tool_call: The ToolUseBlock from the LLM.
        """
        ...


class Toolkit(mutobj.Declaration):
    """Base class for tool providers.

    All public methods (not starting with _) defined on subclasses
    are automatically discovered as tools by ToolSet when
    auto_discover is enabled.

    工具名格式为 ``{Prefix}-{method_name}``，前缀从类名自动生成：
    类名以 ``Toolkit`` 结尾时去掉该后缀，否则使用完整类名。
    可通过 ``_tool_prefix`` 显式指定前缀，空字符串时工具名即方法名。

    Attributes:
        owner: 拥有此 Toolkit 的 ToolSet 实例。
            由 ToolSet 在 add() 或 auto-discover 时设置。
            通过 owner 可访问绑定链：owner.agent → Session。
        discoverable: 控制是否被 auto-discover 发现。
            设为 False 则 auto-discover 跳过此类，但仍可通过 .add() 手动注册。
            子类不设置则继承默认值 True。
        tool_methods: 方法级白名单。
            设置后只暴露列表中的方法为工具。未设置则暴露所有公开方法（向后兼容）。
        tool_prefix: 工具名前缀。
            None = 从类名自动推导，空字符串 = 无前缀（工具名即方法名）。

    Example::

        class WebToolkit(mutagent.Toolkit):
            def search(self, query: str) -> str:
                '''Search the web.'''  # → 工具名 "Web-search"
                ...
    """

    owner: ToolSet | None = None
    discoverable: ClassVar[bool] = True
    tool_methods: ClassVar[list[str] | None] = None
    tool_prefix: ClassVar[str | None] = None

    def customize_schema(self, method_name: str, schema: ToolSchema) -> ToolSchema:
        """动态调整工具 schema。子类可覆盖。

        在 ToolSet 生成 schema 后调用，允许 Toolkit 实例
        根据运行时状态（如已发现的 provider）修改描述或参数。

        Args:
            method_name: 方法名称。
            schema: 自动生成的 ToolSchema。

        Returns:
            调整后的 ToolSchema（或原样返回）。
        """
        ...


from . import _tools_impl as _tools_impl  # noqa: F401, E402
