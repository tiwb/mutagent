"""mutagent.net — 兼容层，实际实现在 mutio.net + mutio.mcp。"""

# 导入 mutio.net 触发 impl 注册
import mutio.net as _net  # noqa: F401
# 导入 mutio.mcp 触发 impl 注册
import mutio.mcp as _mcp  # noqa: F401
