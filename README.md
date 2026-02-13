# mutagent

A Python AI Agent framework that enables LLMs to self-iterate Python code at runtime.

> **Note:** This package is in early development. Stay tuned for updates.

## Overview

**mutagent** (mutation + agent) provides a runtime environment where AI agents can view, modify, and hot-reload Python code, forming an efficient development loop.

Key concepts:
- **Agent as Developer** - LLM operates Python modules like a developer iterating code
- **Runtime Iterable** - Hot-swap implementations without restart via declaration-implementation separation
- **Self-Evolving Tools** - Agent can create, iterate, and evolve its own tools

## Installation

```bash
pip install mutagent
```

## License

MIT
