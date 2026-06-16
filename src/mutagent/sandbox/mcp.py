"""mutagent.sandbox.adapter_mcp -- MCP 连接管理器的公开契约。

MCPConnection 是单个 MCP source 的长生命周期代理，统一管理连接状态、
namespace 产出和 tool 元数据查询。Connector 是 Namespace 反向懒触发的
最小协议。

外部消费者（app、cli、webui）通过此契约与连接管理器交互，
不感知 transport 细节（stdio / http）、状态机实现或内部锁。
"""

from typing import Literal, Sequence

import mutobj
from mutio.codec.json import JsonObject
from .namespace import NamespaceProtocol

ConnectionState = Literal["disconnected", "connecting", "connected", "failed"]
"""连接状态枚举。

- ``disconnected``：从未连过或已主动断开
- ``connecting``：正在重建连接
- ``connected``：当前可用
- ``failed``：上次连接失败，处于冷却期或等待下次触发
"""


class MCPConnection(mutobj.Declaration):
    """单个 MCP source 的长生命周期代理。

    职责：
    - 持有 :class:`Namespace`（连接成功与否始终存在）
    - 管理连接生命周期（懒连 / 显式重连 / 关闭）
    """

    def __init__(
        self,
        ns_name: str,
        config: JsonObject,
    ) -> None:
        """构造一个 MCP connection 代理。

        Args:
            ns_name: 原始 source 名（config dict key），用作 namespace 名。
            config: MCP server 配置，含 transport 类型、对应参数及
                retry_cooldown（失败冷却秒数，默认 5.0）。
                结构见 ``make_client``。
        """
        ...

    @property
    def config(self) -> JsonObject:
        """构造时传入的 MCP server 配置（transport / command / url 等）。"""
        ...

    @property
    def name(self) -> str:
        """原始 source 名（config dict key）。构造后不可变。"""
        ...

    @property
    def state(self) -> ConnectionState:
        """当前连接状态。"""
        ...

    @property
    def last_error(self) -> str | None:
        """最近一次连接失败的原因。"""
        ...

    @property
    def namespace(self) -> NamespaceProtocol:
        ...

    @property
    def last_attempt_at(self) -> float | None:
        """最近一次重连尝试的 Unix 时间戳，未尝试过则为 None。"""
        ...

    @property
    def peer_namespaces(self) -> Sequence[NamespaceProtocol]:
        """从对端 pysandbox 融合进来的 peer namespaces。

        仅 HTTP transport + 对端支持 pysandbox capability 时非空。
        """
        ...

    async def ensure_connected(self) -> None:
        """幂等保证连接可用。

        - ``connected`` → 立即返回
        - ``failed`` 且在冷却期 → 抛 :class:`MCPTransportError`
        - 其他状态 → 创建或重建连接，刷新 namespace 函数表

        Raises:
            MCPTransportError: 冷却期内或重建失败。
        """
        ...

    async def reconnect(self) -> None:
        """显式全量重建连接。

        忽略当前状态，强制重新握手、刷新 tool 列表和 peer namespaces。
        用于 tool 调用传输错后自动恢复，或用户手动 Reload tools。

        Raises:
            MCPTransportError: 重建失败。
        """
        ...

    async def close(self) -> None:
        """彻底关闭连接。幂等，多次调用安全。

        清理底层 client 资源，摘除 peer namespaces，状态置为 ``disconnected``。
        """
        ...

    def list_tools_metadata(self) -> list[JsonObject]:
        """返回当前 source 暴露的 tool 元数据列表。"""
        ...


from . import _mcp_impl as _mcp_impl  # noqa: E402,F401
