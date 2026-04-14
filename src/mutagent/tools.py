"""mutagent.tools -- Toolkit base class and ToolSet declaration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import mutagent

if TYPE_CHECKING:
    from mutagent.agent import Agent
    from mutagent.messages import ToolSchema, ToolUseBlock


@dataclass
class ToolEntry:
    """A registered tool entry.

    Attributes:
        name: Tool name (unique identifier).
        callable: The actual callable (bound method or function).
        schema: ToolSchema for LLM API.
        source: Source object reference (for batch removal by source).
    """

    name: str
    callable: Callable
    schema: ToolSchema
    source: Any


class ToolSet(mutagent.Declaration):
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

    def add(self, source: Any, methods: list[str] | None = None) -> None:
        """Add tools from a source.

        Args:
            source: Tool source, can be:
                - An object instance: register its public methods as tools.
                - A single callable: register as one tool.
            methods: When source is an object, specify which method names
                to register. None registers all public methods defined
                directly on the class (not inherited).
        """
        return tool_set_impl.add(self, source, methods=methods)

    def remove(self, tool_name: str) -> bool:
        """Remove a tool by name.

        Args:
            tool_name: Name of the tool to remove.

        Returns:
            True if the tool was found and removed, False otherwise.
        """
        return tool_set_impl.remove(self, tool_name)

    def query(self, tool_name: str) -> ToolSchema | None:
        """Query a tool's schema by name.

        Args:
            tool_name: Name of the tool to query.

        Returns:
            The ToolSchema if found, None otherwise.
        """
        return tool_set_impl.query(self, tool_name)

    def get_tools(self) -> list[ToolSchema]:
        """Get all available tool schemas for the LLM API.

        Returns:
            List of ToolSchema objects.
        """
        return tool_set_impl.get_tools(self)

    async def dispatch(self, tool_call: ToolUseBlock) -> None:
        """Dispatch a tool call, updating the ToolUseBlock in-place.

        Sets status/result/is_error/duration on the block.

        Args:
            tool_call: The ToolUseBlock from the LLM.
        """
        return await tool_set_impl.dispatch(self, tool_call)


class Toolkit(mutagent.Declaration):
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
        _discoverable: 控制是否被 auto-discover 发现。
            设为 False 则 auto-discover 跳过此类，但仍可通过 .add() 手动注册。
            子类不设置则继承默认值 True。
        _tool_methods: 方法级白名单。
            设置后只暴露列表中的方法为工具。未设置则暴露所有公开方法（向后兼容）。
        _tool_prefix: 工具名前缀。
            None = 从类名自动推导，空字符串 = 无前缀（工具名即方法名）。

    Example::

        class WebToolkit(mutagent.Toolkit):
            def search(self, query: str) -> str:
                '''Search the web.'''  # → 工具名 "Web-search"
                ...
    """

    owner: ToolSet | None = None
    _discoverable: ClassVar[bool] = True
    _tool_methods: ClassVar[list[str] | None] = None
    _tool_prefix: ClassVar[str | None] = None

    def _customize_schema(self, method_name: str, schema: ToolSchema) -> ToolSchema:
        """动态调整工具 schema。子类可覆盖。

        在 ToolSet 生成 schema 后调用，允许 Toolkit 实例
        根据运行时状态（如已发现的 provider）修改描述或参数。

        Args:
            method_name: 方法名称。
            schema: 自动生成的 ToolSchema。

        Returns:
            调整后的 ToolSchema（或原样返回）。
        """
        return schema


from .builtins import tool_set_impl
mutagent.register_module_impls(tool_set_impl)
