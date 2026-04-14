"""mutagent.builtins.tool_set -- ToolSet implementation."""

import asyncio
import inspect
import logging
from typing import Any

import mutagent
from mutagent.messages import ToolSchema, ToolUseBlock
from mutagent.builtins.schema import get_declaration_method, make_schema
from mutagent.tools import ToolEntry, ToolSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal state helpers
# ---------------------------------------------------------------------------

def _get_entries(self: ToolSet) -> dict[str, ToolEntry]:
    """Get or initialize the internal entries dict (manually added tools)."""
    entries = getattr(self, '_entries', None)
    if entries is None:
        entries = {}
        object.__setattr__(self, '_entries', entries)
    return entries


def _get_added_classes(self: ToolSet) -> set[type]:
    """Get or initialize the set of classes added via add()."""
    added = getattr(self, '_added_classes', None)
    if added is None:
        added = set()
        object.__setattr__(self, '_added_classes', added)
    return added


def _get_discovered(self: ToolSet) -> dict[type, dict]:
    """Get or initialize the auto-discovered toolkit state.

    Returns dict mapping toolkit class -> {
        'instance': object,
        'entries': dict[str, ToolEntry],
        'version': int,     # module version at time of discovery
        'module': str,       # module name for version tracking
    }
    """
    discovered = getattr(self, '_discovered', None)
    if discovered is None:
        discovered = {}
        object.__setattr__(self, '_discovered', discovered)
    return discovered


# ---------------------------------------------------------------------------
# Late binding
# ---------------------------------------------------------------------------

def _make_late_bound(instance: Any, method_name: str):
    """Create a late-bound wrapper that resolves the method at call time.

    This ensures that when the class is updated via define_module, the
    next call uses the new implementation without re-registration.

    For async methods, an async wrapper is generated so that
    ``inspect.iscoroutinefunction`` returns True and dispatch awaits correctly.
    """
    actual = getattr(instance, method_name)
    if inspect.iscoroutinefunction(actual):
        async def wrapper(**kwargs):  # type: ignore[reportRedeclaration]
            return await getattr(instance, method_name)(**kwargs)
    else:
        def wrapper(**kwargs):
            return getattr(instance, method_name)(**kwargs)

    # Copy metadata for schema generation
    wrapper.__name__ = method_name
    wrapper.__doc__ = actual.__doc__
    wrapper.__annotations__ = getattr(actual, '__annotations__', {})
    return wrapper


# ---------------------------------------------------------------------------
# 工具命名
# ---------------------------------------------------------------------------

def _get_tool_prefix(cls: type) -> str:
    """从 Toolkit 类名生成工具前缀。

    优先使用 _tool_prefix 显式指定，否则从类名推导（去掉 Toolkit 后缀）。
    """
    explicit = cls.__dict__.get('_tool_prefix')
    if explicit is not None:
        return explicit
    name = cls.__name__
    if name.endswith("Toolkit") and name != "Toolkit":
        return name[:-7]  # 去掉 "Toolkit"
    return name


def _get_tool_name(cls: type, method_name: str) -> str:
    """生成工具名称。有前缀时 ``{prefix}-{method}``，无前缀时直接用方法名。"""
    prefix = _get_tool_prefix(cls)
    if not prefix:
        return method_name
    return f"{prefix}-{method_name}"


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def _discover_toolkit_classes() -> list[type]:
    """Scan mutobj registry for all Toolkit subclasses."""
    import mutobj
    from mutagent.tools import Toolkit

    return mutobj.discover_subclasses(Toolkit)


def _get_public_methods(cls: type) -> list[str]:
    """Get public method names defined directly on the class.

    If the class defines ``_tool_methods``, only those methods are returned
    (whitelist mode). Otherwise, all public (non-underscore) callables in
    ``cls.__dict__`` are returned (default behavior, backward compatible).
    """
    tool_methods = cls.__dict__.get('_tool_methods')
    if tool_methods is not None:
        return [m for m in tool_methods if m in cls.__dict__]
    return [
        name for name, val in cls.__dict__.items()
        if not name.startswith("_") and callable(val)
    ]


def _get_module_name(cls: type) -> str:
    """Get the module name for a class (for version tracking)."""
    return getattr(cls, '__module__', '')


def _make_entries_for_toolkit(cls: type, instance: Any) -> dict[str, ToolEntry]:
    """Create ToolEntry dict for a Toolkit class instance (late-bound).

    工具名根据类的 tool_prefix 决定：有前缀时使用 ``{prefix}-{method}`` 格式。
    """
    entries: dict[str, ToolEntry] = {}
    for method_name in _get_public_methods(cls):
        tool_name = _get_tool_name(cls, method_name)
        late_bound = _make_late_bound(instance, method_name)
        decl_method = get_declaration_method(cls, method_name)
        schema = make_schema(decl_method, tool_name)
        # 允许 Toolkit 实例动态调整 schema
        if hasattr(instance, '_customize_schema'):
            schema = instance._customize_schema(method_name, schema)
        entries[tool_name] = ToolEntry(
            name=tool_name,
            callable=late_bound,
            schema=schema,
            source=instance,
        )
    return entries


def _refresh_discovered(self: ToolSet) -> None:
    """Refresh auto-discovered toolkit entries.

    Scans the class registry for Toolkit subclasses, instantiates new ones,
    refreshes stale ones (version changed), and removes gone ones.
    """
    import mutobj
    from mutagent.runtime.module_manager import ModuleManager

    # 短路：注册表无变化时跳过完整扫描
    current_gen = mutobj.get_registry_generation()
    last_gen = getattr(self, '_last_registry_generation', None)
    if last_gen is not None and last_gen == current_gen:
        return
    object.__setattr__(self, '_last_registry_generation', current_gen)

    added_classes = _get_added_classes(self)
    discovered = _get_discovered(self)

    # Find the module_manager for version checking (if available)
    entries = _get_entries(self)
    mgr: ModuleManager | None = None
    for entry in entries.values():
        mgr = getattr(entry.source, 'module_manager', None)
        if mgr is not None:
            break

    current_toolkit_classes = _discover_toolkit_classes()
    current_set = set(current_toolkit_classes)

    # Remove classes that no longer exist
    gone = [cls for cls in discovered if cls not in current_set]
    for cls in gone:
        logger.info("Removing auto-discovered toolkit: %s", cls.__name__)
        del discovered[cls]

    for cls in current_toolkit_classes:
        # Skip classes that were manually added via add()
        if cls in added_classes:
            continue

        # Skip classes that opt out of auto-discovery
        if not getattr(cls, '_discoverable', True):
            continue

        module_name = _get_module_name(cls)
        current_version = mgr.get_version(module_name) if mgr else 0

        if cls in discovered:
            # Already discovered — check if version changed
            state = discovered[cls]
            if state['version'] == current_version:
                continue
            # Version changed: refresh entries
            logger.info("Refreshing toolkit %s (version %d → %d)",
                        cls.__name__, state['version'], current_version)
            instance = state['instance']
            new_entries = _make_entries_for_toolkit(cls, instance)
            state['entries'] = new_entries
            state['version'] = current_version
        else:
            # New class: try to instantiate
            # Check for tool name conflicts with manually added tools
            public_methods = _get_public_methods(cls)
            tool_name_map = {m: _get_tool_name(cls, m) for m in public_methods}
            conflict_methods = [m for m in public_methods if tool_name_map[m] in entries]
            if conflict_methods:
                conflict_tool_names = [tool_name_map[m] for m in conflict_methods]
                logger.warning(
                    "Auto-discovered toolkit %s has tools %s that conflict "
                    "with pre-registered tools; skipping conflicting tools",
                    cls.__name__, conflict_tool_names,
                )
                public_methods = [m for m in public_methods if m not in conflict_methods]
                if not public_methods:
                    continue

            try:
                instance = cls()
            except Exception:
                logger.debug("Cannot auto-instantiate %s (needs constructor args), skipping",
                             cls.__name__)
                continue

            # 设置 Toolkit.owner 绑定
            instance.owner = self

            logger.info("Auto-discovered toolkit: %s with tools %s",
                        cls.__name__, [tool_name_map[m] for m in public_methods])
            tk_entries = _make_entries_for_toolkit(cls, instance)
            # Remove conflicting entries
            for method in conflict_methods:
                tk_entries.pop(tool_name_map[method], None)
            discovered[cls] = {
                'instance': instance,
                'entries': tk_entries,
                'version': current_version,
                'module': module_name,
            }


def _all_entries(self: ToolSet) -> dict[str, ToolEntry]:
    """Get all entries: manually added + auto-discovered."""
    entries = dict(_get_entries(self))  # copy to avoid mutation
    discovered = _get_discovered(self)
    for state in discovered.values():
        for name, entry in state['entries'].items():
            if name not in entries:  # pre-registered takes priority
                entries[name] = entry
    return entries


# ---------------------------------------------------------------------------
# ToolSet method implementations
# ---------------------------------------------------------------------------

@mutagent.impl(ToolSet.add)
def add(self: ToolSet, source: Any, methods: list[str] | None = None) -> None:
    """Add tools from a source object or callable."""
    entries = _get_entries(self)

    if callable(source) and not isinstance(source, type) and methods is None:
        # Single callable (function or lambda)
        name = getattr(source, '__name__', 'unknown')
        schema = make_schema(source, name)
        entries[name] = ToolEntry(
            name=name, callable=source, schema=schema, source=source,
        )
        return

    # Object instance: register its methods
    cls = type(source)
    cls_dict = cls.__dict__

    # 设置 Toolkit.owner 绑定
    from mutagent.tools import Toolkit
    if isinstance(source, Toolkit):
        source.owner = self

    # Track this class as manually added (skip in auto-discovery)
    added_classes = _get_added_classes(self)
    added_classes.add(cls)

    if methods is not None:
        method_names = methods
    else:
        method_names = _get_public_methods(cls)

    for method_name in method_names:
        if method_name not in cls_dict:
            logger.warning("Method %s not found in %s.__dict__, skipping", method_name, cls.__name__)
            continue
        tool_name = _get_tool_name(cls, method_name)
        bound_method = getattr(source, method_name)
        # Use declaration method for schema (preserves original signature/docstring)
        decl_method = get_declaration_method(cls, method_name)
        schema = make_schema(decl_method, tool_name)
        # 允许 Toolkit 实例动态调整 schema
        if hasattr(source, '_customize_schema'):
            schema = source._customize_schema(method_name, schema)  # type: ignore[reportFunctionMemberAccess]
        entries[tool_name] = ToolEntry(
            name=tool_name,
            callable=bound_method,
            schema=schema,
            source=source,
        )


@mutagent.impl(ToolSet.remove)
def remove(self: ToolSet, tool_name: str) -> bool:
    """Remove a tool by name."""
    entries = _get_entries(self)
    if tool_name in entries:
        del entries[tool_name]
        return True
    # Also check discovered entries
    discovered = _get_discovered(self)
    for state in discovered.values():
        if tool_name in state['entries']:
            del state['entries'][tool_name]
            return True
    return False


@mutagent.impl(ToolSet.query)
def query(self: ToolSet, tool_name: str) -> ToolSchema | None:
    """Query a tool's schema by name."""
    if self.auto_discover:
        _refresh_discovered(self)
    all_entries = _all_entries(self)
    entry = all_entries.get(tool_name)
    return entry.schema if entry else None


@mutagent.impl(ToolSet.get_tools)
def get_tools(self: ToolSet) -> list[ToolSchema]:
    """Return all tool schemas (static + auto-discovered)."""
    if self.auto_discover:
        _refresh_discovered(self)
    all_entries = _all_entries(self)
    return [entry.schema for entry in all_entries.values()]


@mutagent.impl(ToolSet.dispatch)
async def dispatch(self: ToolSet, tool_call: ToolUseBlock) -> None:
    """Dispatch a tool call, updating the ToolUseBlock in-place."""
    if self.auto_discover:
        _refresh_discovered(self)
    all_entries = _all_entries(self)
    entry = all_entries.get(tool_call.name)
    if entry is None:
        tool_call.status = "done"
        tool_call.result = f"Unknown tool: {tool_call.name}"
        tool_call.is_error = True
        return

    # 跟踪当前 tool_call（供 UIToolkit 等使用）
    object.__setattr__(self, '_current_tool_call', tool_call)
    try:
        fn = entry.callable
        if inspect.iscoroutinefunction(fn):
            result = await fn(**tool_call.input)
        else:
            result = await asyncio.to_thread(fn, **tool_call.input)
        tool_call.status = "done"
        tool_call.result = str(result)
    except asyncio.CancelledError:
        tool_call.status = "done"
        tool_call.result = "(cancelled)"
        tool_call.is_error = False
        raise
    except Exception as e:
        tool_call.status = "done"
        tool_call.result = f"{type(e).__name__}: {e}"
        tool_call.is_error = True
    finally:
        # 通用清理：如果工具执行期间创建了 UIContext，关闭它
        active_ui = getattr(self, '_active_ui', None)
        if active_ui:
            active_ui.close()
            object.__setattr__(self, '_active_ui', None)
        object.__setattr__(self, '_current_tool_call', None)
