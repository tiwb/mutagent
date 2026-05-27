"""mutagent.sandbox — 安全的 Python 代码执行环境。"""

from mutagent.sandbox._mcp_share import PYSANDBOX_CAPABILITY, register_pysandbox_methods
from mutagent.sandbox.env import SandboxEnv
from mutagent.sandbox.mcp import MCPConnection
from mutagent.sandbox.namespace import NamespaceTools, NamespaceProtocol

SandboxApp = SandboxEnv   # 兼容别名

__all__ = [
    "PYSANDBOX_CAPABILITY",
    "SandboxApp",
    "SandboxEnv",
    "MCPConnection",
    "NamespaceProtocol",
    "NamespaceTools",
    "register_pysandbox_methods",
]
