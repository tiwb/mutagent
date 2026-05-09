# mutagent

A Python AI Agent framework that enables LLMs to self-iterate Python code at runtime.

> **Note:** This package is in early development. Stay tuned for updates.

## Overview

**mutagent** (mutation + agent) provides a runtime environment where AI agents can view, modify, and hot-reload Python code, forming an efficient development loop.

Key concepts:
- **Agent as Developer** - LLM operates Python modules like a developer iterating code
- **Runtime Iterable** - Hot-swap implementations without restart via declaration-implementation separation
- **Self-Evolving Tools** - Agent can create, iterate, and evolve its own tools

## Pysandbox namespace sharing — 融合远端 mutbot 能力

进程内的 sandbox 除了本地 `NamespaceTools` 发现的 namespace，还可以通过
配置 `mcp_sources` 从一个远端 pysandbox peer（典型是另一个 mutbot
实例）拉取它的 namespaces 平铺融合进本地 registry，供 LLM 在
`pysandbox(code=...)` 里以 `mutbot.status()` 这种形式直接调用。

```jsonc
// .mutagent/config.json
{
  "mcp_sources": {
    "mutbot_remote": { "url": "http://127.0.0.1:8741/mcp" }
  }
}
```

启动后：

- 连接 handshake 检测到对端声明了 `capabilities.pysandbox` 后，自动拉取
  `pysandbox/namespaces.list/describe`，按远端原名合入本地 registry
- 同进程的标准 MCP tools 也照常融合（仅自动过滤对端的 `pysandbox`
  tool 自身，避免递归）
- 沙箱内 `help()` 与本地 namespace 无感知差异；namespace 详情末尾
  会标记 `(shared from <source>)`
- 连接断开时共享进来的 namespace 一起显示 `[failed: ...]`，下次
  调用会触发重连

详见 `docs/specifications/feature-pysandbox-namespace-sharing.md`。

## Installation

```bash
pip install mutagent
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Release

Tag 触发自动发布（PyPI Trusted Publishers，无需 token）：

```bash
git tag v0.2.x
git push origin v0.2.x
```

源码版本保持 `x.y.999`，CI 从 tag 提取正式版本号替换后构建发布。

## License

MIT
