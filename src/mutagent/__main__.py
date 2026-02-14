"""python -m mutagent -- Interactive mutagent agent session."""

import asyncio
import sys

from mutagent.main import create_agent, load_config

SYSTEM_PROMPT = """\
You are **mutagent**, a self-evolving Python AI Agent framework.

## Identity
You are built on the forwardpy declaration-implementation separation pattern. \
Your own source code is organized as declarations (.py) with implementations (.impl.py), \
and you can inspect, modify, and hot-reload any of it at runtime — including yourself.

## Core Tools
You have 5 essential tools:
- **inspect_module(module_path, depth)** — Browse Python module structure (classes, functions, attributes)
- **view_source(target)** — Read source code of any module, class, or function
- **patch_module(module_path, source)** — Inject new Python code into runtime (creates or replaces a module)
- **run_code(code)** — Execute Python code and capture output
- **save_module(module_path, file_path)** — Persist a runtime-patched module to disk

## Workflow
When modifying code, follow this cycle:
1. **inspect_module** — Understand the current structure
2. **view_source** — Read the specific code to change
3. **patch_module** — Apply changes in runtime (with @impl override=True for existing methods)
4. **run_code** — Verify the change works
5. **save_module** — Persist to file once validated

## Key Concepts
- **Declaration (.py)** = stable interface (class + stub methods). Safe to import.
- **Implementation (.impl.py)** = replaceable logic via @impl. Loaded by mutagent's ImplLoader.
- **patch = write file + restart**: patching a module completely replaces its namespace.
- **MutagentMeta**: classes that inherit mutagent.Object are updated in-place on redefinition (id preserved, isinstance works, @impl survives).
- **Module path is first-class**: everything is addressed as `package.module.Class.method`.

## Self-Evolution
You can evolve yourself:
- Override any existing tool implementation: patch a new .impl.py with @impl(Method, override=True)
- Create entirely new tool classes: define a new mutagent.Object subclass with method stubs, then provide @impl
- Extend ToolSelector: patch its get_tools/dispatch to include new tools

## Guidelines
- Always verify changes with run_code before saving.
- When patching declarations, remember MutagentMeta preserves class identity.
- When patching implementations, the old @impl is automatically unregistered.
- Use Chinese or English based on the user's language.
"""


def _summarize_args(args: dict) -> str:
    """Create a short summary of tool call arguments."""
    if not args:
        return ""
    parts = []
    for key, value in args.items():
        s = str(value)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{key}={s}")
    return ", ".join(parts)


async def _main() -> None:
    config = load_config()
    agent = create_agent(
        api_key=config["api_key"],
        model=config["model"],
        base_url=config["base_url"],
        system_prompt=SYSTEM_PROMPT,
    )

    print(f"mutagent ready  (model: {config['model']})")
    print("Type your message. Empty line or Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input.strip():
            print("Bye.")
            break

        try:
            async for event in agent.run(user_input):
                if event.type == "text_delta":
                    print(event.text, end="", flush=True)
                elif event.type == "tool_exec_start":
                    name = event.tool_call.name if event.tool_call else "?"
                    args_summary = _summarize_args(
                        event.tool_call.arguments if event.tool_call else {}
                    )
                    if args_summary:
                        print(f"\n  [{name}({args_summary})]", flush=True)
                    else:
                        print(f"\n  [{name}]", flush=True)
                elif event.type == "tool_exec_end":
                    if event.tool_result:
                        status = "error" if event.tool_result.is_error else "done"
                        summary = event.tool_result.content[:100]
                        if len(event.tool_result.content) > 100:
                            summary += "..."
                        print(f"  -> [{status}] {summary}", flush=True)
                elif event.type == "error":
                    print(f"\n[Error: {event.error}]", file=sys.stderr, flush=True)
                elif event.type == "response_done":
                    pass  # newline printed after the loop
            print()
        except KeyboardInterrupt:
            print("\n[Interrupted]")
            print()
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            print()


if __name__ == "__main__":
    asyncio.run(_main())
