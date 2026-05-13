"""MCP Settings panel — Declaration + 完整实现。

`MCPSettingsPanel` 提供 ☰ → MCP 连接设置 入口，覆盖 list / edit 两个步骤、
显式 Connect/Disconnect/Reconnect/Reload tools 控制、`config.mcp_sources` 配置
持久化。设计参考 `_settings_llm.py`，与 LLM Settings 面板视觉对齐。

详见 `mutagent/docs/specifications/feature-mcp-source-config.md`。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import mutagent
from mutagent.sandbox._adapter_mcp import (
    MCPConnection,
    _sanitize_ns_name,
)
from mutagent.sandbox._namespace import connection_status
from mutagent.sandbox._signature import format_callable_signature
from mutagent.webui.settings import SettingsPanel
from mutgui import Bind, Callback, ViewBlock

logger = logging.getLogger(__name__)


_TRANSPORT_OPTIONS = [
    {"label": "stdio (subprocess)", "value": "stdio"},
    {"label": "http (Streamable HTTP)", "value": "http"},
]

# 哪些字段变更需要提示「需要 Disconnect → Connect」横幅
_RUNTIME_CRITICAL_FIELDS = (
    "transport", "command", "args", "shell", "env", "url", "timeout"
)

_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ═══════════════════════════════════════════════════════════════
#  Declaration
# ═══════════════════════════════════════════════════════════════


class MCPSettingsPanel(SettingsPanel):
    """MCP 连接配置面板。

    SettingsDrawer 通过 `discover_subclasses` 自动发现并实例化。
    `panel_placement="settings:10/20"` 排在 LLM 之后。
    """

    panel_id: ClassVar[str] = "mcp"
    panel_title: ClassVar[str] = "MCP 连接设置"
    panel_placement: ClassVar[str] = "settings:10/20"
    panel_width: ClassVar[int] = 640

    # ── State fields ─────────────────────────────
    current_step: str
    editing_key: str
    editing_is_new: bool
    form_name: str
    form_transport: str
    form_command: str
    form_args_text: str
    form_shell: bool
    form_env_text: str
    form_url: str
    form_timeout: float
    form_autostart: bool
    form_retry_cooldown: float
    error: str
    notice: str
    pending_button: str  # "<key>:connect" / "<key>:disconnect" / "<key>:reconnect" / "<key>:reload"
    expanded_ns: set
    expanded_fn: set

    def __init__(self, *, app: Any, agent: Any) -> None: ...

    def render(self) -> ViewBlock: ...

    def on_open(self) -> None: ...


# ═══════════════════════════════════════════════════════════════
#  env 文本解析（.env 风格）
# ═══════════════════════════════════════════════════════════════


def _parse_env_text(text: str) -> tuple[dict[str, str], list[str]]:
    """解析 .env 风格文本为 ``(env_dict, errors)``。

    规则：
    - 一行一对 ``KEY=VALUE``
    - 第一个 ``=`` 为分隔符，value 中可含 ``=``
    - value 首尾去空格；以 ``"`` / ``'`` 包裹时脱壳并保留内部空白
    - ``#`` 开头或纯空白行跳过
    - KEY 必须匹配 ``[A-Za-z_][A-Za-z0-9_]*``

    错误以 ``"line N: <reason>"`` 形式累积返回（不抛异常，便于 UI 红字展示）。
    """
    result: dict[str, str] = {}
    errors: list[str] = []
    if not text:
        return result, errors
    for lineno, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            errors.append(f"line {lineno}: missing '=' in {raw!r}")
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if not _KEY_PATTERN.match(key):
            errors.append(f"line {lineno}: invalid key {key!r}")
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        result[key] = val
    return result, errors


def _format_env_text(env: dict[str, str]) -> str:
    if not env:
        return ""
    return "\n".join(f"{k}={v}" for k, v in env.items())


def _format_args_text(args: list[str]) -> str:
    """args list 与 textarea 互转：一行一个参数，便于含空格的参数原样展示。"""
    if not args:
        return ""
    return "\n".join(args)


def _parse_args_text(text: str) -> list[str]:
    """空行跳过；每行作为一个 arg（保留行内空格）。"""
    if not text:
        return []
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s:
            out.append(s)
    return out


# ═══════════════════════════════════════════════════════════════
#  draft / config 转换
# ═══════════════════════════════════════════════════════════════


def _draft_from_config(key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    transport = str(cfg.get("transport", "stdio") or "stdio").lower()
    return {
        "name": key,
        "transport": transport if transport in ("stdio", "http") else "stdio",
        "command": str(cfg.get("command", "") or ""),
        "args": list(cfg.get("args") or []),
        "shell": bool(cfg.get("shell", False)),
        "env": dict(cfg.get("env") or {}),
        "url": str(cfg.get("url", "") or ""),
        "timeout": float(cfg.get("timeout", 30.0) or 30.0),
        "autostart": bool(cfg.get("autostart", True)),
        "retry_cooldown": float(cfg.get("retry_cooldown", 5.0) or 0.0),
    }


def _make_draft(name: str, transport: str = "stdio") -> dict[str, Any]:
    return {
        "name": name,
        "transport": transport,
        "command": "",
        "args": [],
        "shell": False,
        "env": {},
        "url": "",
        "timeout": 30.0,
        "autostart": True,
        "retry_cooldown": 5.0,
    }


def _draft_to_config(draft: dict[str, Any]) -> dict[str, Any]:
    """把 draft 落地为持久化 config（按 transport 只保留对应字段）。"""
    transport = draft["transport"]
    cfg: dict[str, Any] = {
        "transport": transport,
        "autostart": bool(draft.get("autostart", True)),
        "retry_cooldown": float(draft.get("retry_cooldown", 5.0)),
    }
    if transport == "stdio":
        cfg["command"] = str(draft.get("command", "")).strip()
        args = draft.get("args", [])
        if args:
            cfg["args"] = list(args)
        if draft.get("shell"):
            cfg["shell"] = True
        env = draft.get("env") or {}
        if env:
            cfg["env"] = dict(env)
    else:
        cfg["url"] = str(draft.get("url", "")).strip()
        cfg["timeout"] = float(draft.get("timeout", 30.0))
    return cfg


def _persist_current_form(self: MCPSettingsPanel
                          ) -> tuple[dict[str, Any], list[str]]:
    """把 form_* 字段抓回 draft，同时返回 env 解析错误列表。"""
    env_dict, env_errors = _parse_env_text(self.form_env_text)
    args_list = _parse_args_text(self.form_args_text)
    draft = {
        "name": self.form_name.strip(),
        "transport": self.form_transport,
        "command": self.form_command.strip(),
        "args": args_list,
        "shell": bool(self.form_shell),
        "env": env_dict,
        "url": self.form_url.strip(),
        "timeout": float(self.form_timeout or 30.0),
        "autostart": bool(self.form_autostart),
        "retry_cooldown": float(self.form_retry_cooldown or 0.0),
    }
    return draft, env_errors


def _apply_draft_to_form(self: MCPSettingsPanel, draft: dict[str, Any]) -> None:
    self.form_name = str(draft["name"])
    self.form_transport = str(draft["transport"])
    self.form_command = str(draft.get("command", ""))
    self.form_args_text = _format_args_text(draft.get("args", []))
    self.form_shell = bool(draft.get("shell", False))
    self.form_env_text = _format_env_text(draft.get("env") or {})
    self.form_url = str(draft.get("url", ""))
    self.form_timeout = float(draft.get("timeout", 30.0))
    self.form_autostart = bool(draft.get("autostart", True))
    self.form_retry_cooldown = float(draft.get("retry_cooldown", 5.0))


# ═══════════════════════════════════════════════════════════════
#  config 读写
# ═══════════════════════════════════════════════════════════════


def _config_path(self: MCPSettingsPanel) -> Path:
    path = getattr(self._app, "config_path", None)
    if isinstance(path, Path):
        return path
    return (Path.cwd() / ".mutagent" / "config.json").resolve()


def _write_config(self: MCPSettingsPanel,
                  mcp_sources: dict[str, dict[str, Any]]) -> None:
    config = self._agent.config
    data = getattr(config, "_data", None)
    if not isinstance(data, dict):
        raise RuntimeError("Current Config implementation cannot be saved from WebUI")
    config.set("mcp_sources", mcp_sources, source="webui")
    path = _config_path(self)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _load_from_config(self: MCPSettingsPanel) -> None:
    """从 config + sandbox 反查，刷新 self._drafts / self._conns。

    drafts: dict[原始 key, draft dict]
    conns: dict[原始 key, MCPConnection]
    两者按原始 key 对齐；conn 可能没有（rename/未启动）。
    """
    cfg_sources = self._agent.config.get("mcp_sources", default={}) or {}
    self._drafts = {
        key: _draft_from_config(key, deepcopy(cfg))
        for key, cfg in cfg_sources.items()
        if isinstance(cfg, dict)
    }
    sandbox = getattr(self._app, "sandbox", None)
    if sandbox is not None and hasattr(sandbox, "mcp_connections"):
        self._conns = sandbox.mcp_connections()
    else:
        self._conns = {}
    self.current_step = "list"
    self.editing_key = ""
    self.editing_is_new = False
    self.pending_button = ""
    self.expanded_ns = set()
    self.expanded_fn = set()
    _set_message(self)


def _set_message(self: MCPSettingsPanel, *, error: str = "",
                 notice: str = "") -> None:
    self.error = error
    self.notice = notice


# ═══════════════════════════════════════════════════════════════
#  名字校验 & 冲突
# ═══════════════════════════════════════════════════════════════


def _check_name_conflicts(self: MCPSettingsPanel,
                          new_name: str) -> str:
    """返回错误描述；空字符串表示无冲突。

    检查：
    - 名字非空
    - 不与其他 source 的原始 key 撞
    - sanitized 名不与其他 source 的 sanitized 名撞
    """
    if not new_name:
        return "Name cannot be empty."
    is_new = self.editing_is_new
    edit_key = self.editing_key
    others_keys = [
        k for k in self._drafts
        if (is_new or k != edit_key)
    ]
    if new_name in others_keys:
        return f"Name '{new_name}' already exists."
    sanitized_new = _sanitize_ns_name(new_name)
    for k in others_keys:
        if _sanitize_ns_name(k) == sanitized_new:
            return (f"Sanitized name conflict: '{new_name}' → "
                    f"'{sanitized_new}' collides with existing "
                    f"'{k}'.")
    return ""


def _unique_name(self: MCPSettingsPanel, base: str) -> str:
    candidate = base
    idx = 2
    while candidate in self._drafts:
        candidate = f"{base}-{idx}"
        idx += 1
    return candidate


# ═══════════════════════════════════════════════════════════════
#  状态/按钮 helpers
# ═══════════════════════════════════════════════════════════════


def _conn_state(self: MCPSettingsPanel, key: str) -> str:
    conn = self._conns.get(key)
    if conn is None:
        return "disconnected"
    return conn.state or "disconnected"


def _state_tag_color(state: str) -> str:
    return {
        "connected": "green",
        "connecting": "blue",
        "failed": "red",
        "disconnected": "default",
    }.get(state, "default")


def _state_tag_text(state: str, conn: MCPConnection | None) -> str:
    """UI state tag 文字。

    failed 时 reason 截断和文本端统一为 60 字符（原等于 50，调宽后
    与 :func:`mutagent.sandbox._namespace.connection_status` / ``_format_state_label``
    输出在同一 failed ns 上严格一致）。

    仍允许 conn 是 None 的史用调用形态（不需要 ns）。有 conn 时
    直接复用 ``connection_status`` 的 reason 归一化结果——对单个
    MCPConnection 会写一个最小 shim 凑出 ns.connection_state /
    ns.connection_error 两个字段供纯函数读取。
    """
    if state == "failed" and conn is not None and conn.last_error:
        # 复用纯函数 connection_status 的截断逻辑：造一个最小 ns-like shim
        class _Shim:
            _connection = conn  # 使 connection_status 不返 (None, None)
            connection_state = "failed"
            connection_error = conn.last_error
        _, reason = connection_status(_Shim())  # type: ignore[arg-type]
        return f"failed: {reason}" if reason else "failed"
    return state


def _config_changed_at_runtime(draft: dict[str, Any],
                               key: str,
                               conn: MCPConnection | None) -> bool:
    """检测 draft 与 conn.config 在运行期关键字段上是否有差异。

    两边都归一化（draft→config 管道），防止 conn.config 残留跨 transport
    字段导致误报横幅。
    """
    if conn is None:
        return False
    if conn.state not in ("connected", "connecting", "failed"):
        return False
    cfg = conn.config or {}
    # 归一化：conn.config 也走 draft→config 管道，保证两边字段结构一致
    old_cfg = _draft_to_config(_draft_from_config(key, cfg))
    new_cfg = _draft_to_config(draft)
    for field in _RUNTIME_CRITICAL_FIELDS:
        if old_cfg.get(field) != new_cfg.get(field):
            return True
    return False


# ═══════════════════════════════════════════════════════════════
#  异步操作（Connect / Disconnect / Reconnect）
# ═══════════════════════════════════════════════════════════════


def _sandbox_loop(self: MCPSettingsPanel) -> asyncio.AbstractEventLoop | None:
    sandbox = getattr(self._app, "sandbox", None)
    if sandbox is None:
        return None
    return getattr(sandbox, "_async_loop", None)


def _ensure_conn(self: MCPSettingsPanel, key: str) -> MCPConnection | None:
    """按需创建 MCPConnection。

    - 已有 → 直接返回
    - 无 → 用最新 draft 构造，挂到 sandbox._mcp_conns + conn._sandbox 回引
    """
    conn = self._conns.get(key)
    if conn is not None:
        return conn
    draft = self._drafts.get(key)
    if draft is None:
        return None
    sandbox = getattr(self._app, "sandbox", None)
    if sandbox is None:
        return None
    loop = _sandbox_loop(self)
    if loop is None:
        _set_message(self, error="Sandbox event loop not available.")
        return None
    cfg = _draft_to_config(draft)
    try:
        conn = MCPConnection(
            key, cfg, loop,
            retry_cooldown=float(draft.get("retry_cooldown", 5.0)),
        )
    except Exception as exc:
        _set_message(self, error=f"Init connection failed: {exc}")
        return None
    conn._sandbox = sandbox
    sandbox.register_mcp_connection(key, conn)
    self._conns[key] = conn
    return conn


def _submit_async(self: MCPSettingsPanel, coro: Any,
                  pending_token: str) -> None:
    """提交协程到 sandbox loop，完成后 invalidate panel 触发重渲染。"""
    loop = _sandbox_loop(self)
    if loop is None:
        _set_message(self, error="Sandbox event loop not available.")
        self.invalidate()
        return
    self.pending_button = pending_token
    self.invalidate()

    def _done(fut: Any) -> None:
        # 在 sandbox loop 完成 → 切回 panel render（mutgui 自有事件循环 + invalidate 安全）
        try:
            fut.result()
        except Exception as exc:
            self._async_error = str(exc) or exc.__class__.__name__
        else:
            self._async_error = ""
        self.pending_button = ""
        if self._async_error:
            _set_message(self, error=self._async_error)
        else:
            _set_message(self)
        # 反查 sandbox conns（autostart 后端自动新增的 conn 也能拿到）
        sandbox = getattr(self._app, "sandbox", None)
        if sandbox is not None and hasattr(sandbox, "mcp_connections"):
            self._conns = sandbox.mcp_connections()
        self.invalidate()

    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    fut.add_done_callback(_done)


async def _do_connect(self: MCPSettingsPanel, key: str) -> None:
    conn = _ensure_conn(self, key)
    if conn is None:
        raise RuntimeError(f"Connection '{key}' could not be created.")
    sandbox = getattr(self._app, "sandbox", None)
    # 若 namespace 还没注册（autostart=false 路径），先注册再连
    if sandbox is not None:
        registry = getattr(sandbox, "_registry", None)
        already_registered = (
            registry is not None and conn.namespace in
            registry._namespaces.get(conn.namespace.name, [])
        )
        if not already_registered:
            sandbox.add_namespace(conn.namespace, on_remove=conn.close)
    await conn.reconnect()


async def _do_disconnect(self: MCPSettingsPanel, key: str) -> None:
    conn = self._conns.get(key)
    if conn is None:
        return
    # 只 close（处理 peers + client + state），
    # 不再调 sandbox.remove_provider —— 它会通过 on_remove=conn.close
    # 再触发一次后台 close，双 close 并发可能抛异常覆盖 UI 消息。
    # namespace 留在 registry 中 state=disconnected，re-connect 时复用。
    await conn.close()


async def _do_reconnect(self: MCPSettingsPanel, key: str) -> None:
    conn = self._conns.get(key)
    if conn is None:
        await _do_connect(self, key)
        return
    await conn.reconnect()


# ═══════════════════════════════════════════════════════════════
#  按钮 click handlers（在 mutgui worker 线程上）
# ═══════════════════════════════════════════════════════════════


def _btn_connect(key: str, *, view: MCPSettingsPanel) -> None:
    _submit_async(view, _do_connect(view, key), f"{key}:connect")


def _btn_disconnect(key: str, *, view: MCPSettingsPanel) -> None:
    _submit_async(view, _do_disconnect(view, key), f"{key}:disconnect")


def _btn_reconnect(key: str, *, view: MCPSettingsPanel) -> None:
    _submit_async(view, _do_reconnect(view, key), f"{key}:reconnect")


def _btn_reload_tools(key: str, *, view: MCPSettingsPanel) -> None:
    _submit_async(view, _do_reconnect(view, key), f"{key}:reload")


# ═══════════════════════════════════════════════════════════════
#  导航 & CRUD handlers
# ═══════════════════════════════════════════════════════════════


def _edit_source(key: str, *, view: MCPSettingsPanel) -> None:
    draft = view._drafts.get(key)
    if draft is None:
        return
    view.current_step = "edit"
    view.editing_key = key
    view.editing_is_new = False
    _apply_draft_to_form(view, draft)
    _set_message(view)
    view.invalidate()


def _start_add(transport: str, *, view: MCPSettingsPanel) -> None:
    base = "stdio-source" if transport == "stdio" else "http-source"
    name = _unique_name(view, base)
    draft = _make_draft(name, transport=transport)
    view.current_step = "edit"
    view.editing_key = name
    view.editing_is_new = True
    _apply_draft_to_form(view, draft)
    _set_message(view)
    view.invalidate()


def _back_to_list(*, view: MCPSettingsPanel) -> None:
    view.current_step = "list"
    view.editing_is_new = False
    _set_message(view)
    view.invalidate()


def _save_edits(*, view: MCPSettingsPanel) -> None:
    """保存当前编辑的 source。

    决策 C：只持久化 config，不动运行时连接。
    若关键字段变更 → 通过横幅在编辑页提示；用户需手动 Disconnect→Connect。
    """
    draft, env_errors = _persist_current_form(view)
    if env_errors:
        _set_message(view, error="env: " + "; ".join(env_errors))
        view.invalidate()
        return
    name = draft["name"]
    err = _check_name_conflicts(view, name)
    if err:
        _set_message(view, error=err)
        view.invalidate()
        return
    # 字段必填校验
    if draft["transport"] == "stdio" and not draft["command"]:
        _set_message(view, error="stdio transport requires 'command'.")
        view.invalidate()
        return
    if draft["transport"] == "http" and not draft["url"]:
        _set_message(view, error="http transport requires 'url'.")
        view.invalidate()
        return

    old_key = view.editing_key
    is_rename = (not view.editing_is_new) and old_key != name

    # rename → 删旧 conn（摘 namespace 触发 close），新 conn 等用户点 Connect
    if is_rename:
        old_conn = view._conns.pop(old_key, None)
        sandbox = getattr(view._app, "sandbox", None)
        if old_conn is not None and sandbox is not None:
            try:
                if hasattr(sandbox, "remove_provider"):
                    sandbox.remove_provider(old_conn.namespace)
                if hasattr(sandbox, "unregister_mcp_connection"):
                    sandbox.unregister_mcp_connection(old_key)
            except Exception as exc:
                logger.warning("Cleanup old conn '%s' failed: %s", old_key, exc)
        view._drafts.pop(old_key, None)

    view._drafts[name] = draft
    view.editing_key = name
    view.editing_is_new = False

    # 落盘
    sources_payload = {k: _draft_to_config(d) for k, d in view._drafts.items()}
    try:
        _write_config(view, sources_payload)
    except Exception as exc:
        _set_message(view, error=str(exc))
        view.invalidate()
        return

    # 留在编辑页（方便看横幅 + 立即点 Disconnect/Connect 应用）
    _set_message(view, notice=f"Saved '{name}'.")
    view.invalidate()


def _delete_source(*, view: MCPSettingsPanel) -> None:
    if view.editing_is_new:
        # 新增 draft 还没落盘，直接丢弃
        _back_to_list(view=view)
        return
    key = view.editing_key
    if not key or key not in view._drafts:
        _back_to_list(view=view)
        return
    # 摘 conn + namespace
    conn = view._conns.pop(key, None)
    sandbox = getattr(view._app, "sandbox", None)
    if conn is not None and sandbox is not None:
        try:
            if hasattr(sandbox, "remove_provider"):
                sandbox.remove_provider(conn.namespace)
            if hasattr(sandbox, "unregister_mcp_connection"):
                sandbox.unregister_mcp_connection(key)
        except Exception as exc:
            logger.warning("Cleanup conn '%s' failed: %s", key, exc)
    view._drafts.pop(key, None)
    sources_payload = {k: _draft_to_config(d) for k, d in view._drafts.items()}
    try:
        _write_config(view, sources_payload)
    except Exception as exc:
        _set_message(view, error=str(exc))
        view.invalidate()
        return
    view.current_step = "list"
    view.editing_is_new = False
    view.editing_key = ""
    _set_message(view, notice=f"Removed '{key}'.")
    view.invalidate()


async def _close_panel(*, view: MCPSettingsPanel) -> None:
    await view.drawer.close()


# ═══════════════════════════════════════════════════════════════
#  渲染 — 通用 helpers
# ═══════════════════════════════════════════════════════════════


def _render_message(self: MCPSettingsPanel, *, margin_bottom: int = 12
                    ) -> list[dict[str, Any]]:
    if self.error:
        return [{
            "$component": "antd.Alert",
            "$id": "mcp-error",
            "type": "error",
            "showIcon": True,
            "message": self.error,
            "style": {"marginBottom": margin_bottom},
        }]
    if self.notice:
        return [{
            "$component": "antd.Alert",
            "$id": "mcp-notice",
            "type": "success",
            "showIcon": True,
            "message": self.notice,
            "style": {"marginBottom": margin_bottom},
        }]
    return []


# ═══════════════════════════════════════════════════════════════
#  列表页渲染
# ═══════════════════════════════════════════════════════════════


def _render_list_row(self: MCPSettingsPanel, key: str,
                     draft: dict[str, Any]) -> dict[str, Any]:
    conn = self._conns.get(key)
    state = _conn_state(self, key)
    transport = draft["transport"]
    pending = self.pending_button.startswith(f"{key}:")

    # 单按钮：disconnected/failed→Connect/Reconnect, connected→Disconnect, connecting→灰
    if state == "connected":
        btn_label, btn_handler = "Disconnect", _btn_disconnect
    elif state == "failed":
        btn_label, btn_handler = "Reconnect", _btn_reconnect
    elif state == "connecting":
        btn_label, btn_handler = "Connect", _btn_connect
    else:
        btn_label, btn_handler = "Connect", _btn_connect

    btn_disabled = pending or state == "connecting"

    if conn is not None and state == "connected":
        # 聚合同名 namespace（主 ns + peer namespaces），与 edit 页一致
        merged_funcs: dict[str, Any] = {}
        for ns in [conn.namespace] + conn.peer_namespaces:
            merged_funcs.update(ns._functions)
        tools_count = len(merged_funcs)
    else:
        tools_count = 0
    tools_label = f"{tools_count} functions" if tools_count else ""

    return {
        "$component": "div",
        "$id": f"mcp-row-{key}",
        "style": {
            "display": "flex",
            "alignItems": "center",
            "gap": "12px",
            "padding": "10px 12px",
            "border": "1px solid var(--mutgui-border)",
            "borderRadius": "6px",
            "background": "var(--mutgui-surface, transparent)",
        },
        "$children": [
            # 名字 + transport tag + state tag (clickable → edit)
            {
                "$component": "div",
                "$id": f"mcp-row-info-{key}",
                "style": {
                    "flex": "1",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "4px",
                    "cursor": "pointer",
                    "minWidth": "0",
                },
                "onClick": Callback(partial(_edit_source, key), view="@view"),
                "$children": [
                    {
                        "$component": "div",
                        "$id": f"mcp-row-title-{key}",
                        "style": {
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                            "flexWrap": "wrap",
                        },
                        "$children": [
                            {
                                "$component": "div",
                                "style": {"fontWeight": 600},
                                "children": key,
                            },
                            {
                                "$component": "antd.Tag",
                                "color": "geekblue" if transport == "stdio" else "cyan",
                                "children": transport,
                            },
                            {
                                "$component": "antd.Tag",
                                "color": _state_tag_color(state),
                                "children": _state_tag_text(state, conn),
                            },
                            *([{
                                "$component": "div",
                                "style": {
                                    "fontSize": "12px",
                                    "color": "var(--mutgui-text-dim)",
                                },
                                "children": tools_label,
                            }] if tools_label else []),
                        ],
                    },
                ],
            },
            {
                "$component": "antd.Button",
                "$id": f"mcp-row-btn-{key}",
                "size": "small",
                "loading": pending,
                "disabled": btn_disabled,
                "children": btn_label,
                "onClick": Callback(partial(btn_handler, key), view="@view"),
            },
        ],
    }


def _render_list(self: MCPSettingsPanel) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    items.extend(_render_message(self))
    items.append({
        "$component": "antd.Typography.Paragraph",
        "$id": "mcp-intro",
        "type": "secondary",
        "children": (
            "管理外部 MCP 服务连接。Connect/Disconnect 是显式控制；"
            "autostart 决定启动后是否自动连接。"
        ),
    })
    items.append({
        "$component": "antd.Space",
        "$id": "mcp-add-actions",
        "style": {"marginBottom": 16},
        "$children": [
            {
                "$component": "antd.Button",
                "$id": "mcp-add-stdio",
                "children": "+ Add stdio",
                "onClick": Callback(partial(_start_add, "stdio"), view="@view"),
            },
            {
                "$component": "antd.Button",
                "$id": "mcp-add-http",
                "children": "+ Add HTTP",
                "onClick": Callback(partial(_start_add, "http"), view="@view"),
            },
        ],
    })

    rows = [_render_list_row(self, key, draft)
            for key, draft in self._drafts.items()]
    if not rows:
        rows = [{
            "$component": "antd.Empty",
            "$id": "mcp-empty",
            "description": "No MCP sources configured yet",
        }]
    items.append({
        "$component": "div",
        "$id": "mcp-list",
        "style": {
            "display": "flex",
            "flexDirection": "column",
            "gap": "8px",
            "marginBottom": 16,
        },
        "$children": rows,
    })

    items.append({
        "$component": "div",
        "$id": "mcp-config-path",
        "style": {
            "marginTop": 4,
            "marginBottom": 16,
            "fontSize": "12px",
            "color": "var(--mutgui-text-dim)",
        },
        "children": f"Config file: {_config_path(self)}",
    })

    items.append({
        "$component": "antd.Space",
        "$id": "mcp-list-actions",
        "$children": [
            {
                "$component": "antd.Button",
                "$id": "mcp-close",
                "children": "Close",
                "onClick": Callback(_close_panel, view="@view"),
            },
        ],
    })
    return items


# ═══════════════════════════════════════════════════════════════
#  编辑页渲染
# ═══════════════════════════════════════════════════════════════


def _toggle_ns(ns_name: str, *, view: MCPSettingsPanel) -> None:
    """点击 namespace 行：展开/折叠 Level 2 函数列表。"""
    if ns_name in view.expanded_ns:
        view.expanded_ns.discard(ns_name)
    else:
        view.expanded_ns.add(ns_name)
    view.invalidate()


def _toggle_fn(full_name: str, *, view: MCPSettingsPanel) -> None:
    """点击函数行：展开/折叠 Level 3 函数详情。"""
    if full_name in view.expanded_fn:
        view.expanded_fn.discard(full_name)
    else:
        view.expanded_fn.add(full_name)
    view.invalidate()


def _fn_signature(func: Any) -> str:
    """构建函数签名字符串。

    优先走 `inspect.signature`：MCP tool / pysandbox namespace wrapper 已经
    把真签名挂在 `__signature__` 上（见
    `refactor-wrapper-faithful-signature.md`）。只在 wrapper 构造失败降级为
    `(**kwargs)` 形态时，才回落到 `_mcp_input_schema` 合成路径。
    """
    sig = format_callable_signature(func)
    return sig if sig is not None else "()"


def _fn_detail(func: Any, fn_name: str) -> str:
    """返回函数的完整签名 + docstring 文本。

    签名走 ``inspect.signature`` 统一入口（MCP tool / pysandbox wrapper 的
    ``__signature__`` 伪装先由 ``format_callable_signature`` 处理）。此处不
    再手拼 ``Parameters:`` 表 ——约束行由
    ``format_param_schema_lines`` 统一输出到 docstring
    （``feature-mcp-schema-help-display.iter3.md``）。iter3 落地前，约束信息
    本身由 MCP tool description 以 text 形式给出。
    """
    sig = _fn_signature(func)
    desc = getattr(func, '_mcp_description', None) or ''
    doc = (getattr(func, '__doc__', '') or '').strip()
    # 优先用 _mcp_description（MCP tool 的描述）
    if desc and isinstance(desc, str):
        doc = desc if not doc.startswith(desc) else doc
    elif not desc:
        desc = doc

    lines = [f"{fn_name}{sig}"]
    if desc:
        lines.append("")
        lines.append(desc)
    return '\n'.join(lines)


def _render_function_browser(self: MCPSettingsPanel,
                              conn: MCPConnection | None) -> dict[str, Any]:
    """Level 1~3 逐级展开：namespace → 函数列表 → 函数详情。

    每个 namespace 展示为可点击展开的行（Level 1），展开后显示
    函数列表（Level 2），点击函数展开完整签名 + docstring（Level 3）。
    多 namespace(MCP 主 ns + peer ns)各为独立 Level 1 块。
    """
    if conn is None or conn.state != "connected":
        return {
            "$component": "div",
            "$id": "mcp-func-empty",
            "style": {
                "padding": "12px",
                "color": "var(--mutgui-text-dim)",
                "fontSize": "13px",
                "border": "1px dashed var(--mutgui-border)",
                "borderRadius": "4px",
            },
            "children": "(not connected — use Connect to discover functions)",
        }

    # 收集所有 namespace（主 ns + peer ns），按 name 分组聚合
    ns_groups: dict[str, list[Any]] = {}
    all_ns: list[Any] = [conn.namespace]
    all_ns.extend(conn.peer_namespaces)
    for ns in all_ns:
        ns_groups.setdefault(ns.name, []).append(ns)

    # 每组合并 _functions 和 _descriptions，过滤 0-function 组，按 name 排序
    grouped: list[dict[str, Any]] = []
    for name, ns_group in ns_groups.items():
        merged_funcs: dict[str, Any] = {}
        merged_descs: dict[str, str] = {}
        for ns in ns_group:
            merged_funcs.update(ns._functions)
            merged_descs.update(ns._descriptions)
        if not merged_funcs:
            continue
        grouped.append({
            "name": name,
            "funcs": merged_funcs,
            "descs": merged_descs,
        })
    grouped.sort(key=lambda g: g["name"])

    ns_rows: list[dict[str, Any]] = []

    for g in grouped:
        ns_name = g["name"]
        funcs = g["funcs"]
        is_expanded = ns_name in self.expanded_ns
        toggle = "▾" if is_expanded else "▸"

        # 函数行（Level 2）
        func_rows: list[dict[str, Any]] = []
        if is_expanded and funcs:
            fnames = sorted(funcs.keys())
            max_len = max((len(f) for f in fnames), default=0)
            # 等宽字体 ~8.5px/char，clamp 列宽 140~240
            col_width = min(max(max_len * 9 + 16, 140), 240)
            for fname in fnames:
                func = funcs[fname]
                desc = (g["descs"].get(fname, '') or '').strip()
                first_line = desc.splitlines()[0] if desc else ''
                full_name = f"{ns_name}.{fname}"
                fn_expanded = full_name in self.expanded_fn
                fn_toggle = "▾" if fn_expanded else "▸"

                detail_block: list[dict[str, Any]] = []
                if fn_expanded:
                    detail_text = _fn_detail(func, fname)
                    detail_block = [{
                        "$component": "div",
                        "$id": f"mcp-fn-detail-{full_name}",
                        "style": {
                            "marginTop": 4,
                            "marginLeft": 16,
                            "borderLeft": "2px solid var(--mutgui-primary, #1677ff)",
                            "background": "var(--mutgui-surface, rgba(0,0,0,0.02))",
                            "padding": "10px 14px",
                            "fontSize": "13px",
                            "whiteSpace": "pre-wrap",
                            "fontFamily": "'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace",
                            "lineHeight": "1.5",
                        },
                        "children": detail_text,
                    }]

                func_rows.append({
                    "$component": "div",
                    "$id": f"mcp-fn-row-{full_name}",
                    "$children": [
                        {
                            "$component": "div",
                            "$id": f"mcp-fn-click-{full_name}",
                            "style": {
                                "display": "grid",
                                "gridTemplateColumns": f"16px {col_width}px 1fr",
                                "gap": "4px 8px",
                                "alignItems": "baseline",
                                "padding": "2px 0",
                                "cursor": "pointer",
                            },
                            "onClick": Callback(partial(_toggle_fn, full_name), view="@view"),
                            "$children": [
                                {
                                    "$component": "div",
                                    "$id": f"mcp-fn-toggle-{full_name}",
                                    "style": {
                                        "fontSize": "10px",
                                        "color": "var(--mutgui-text-dim)",
                                    },
                                    "children": fn_toggle,
                                },
                                {
                                    "$component": "div",
                                    "$id": f"mcp-fn-name-{full_name}",
                                    "style": {
                                        "fontFamily": "'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace",
                                        "fontSize": "13px",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                        "whiteSpace": "nowrap",
                                    },
                                    "children": fname,
                                },
                                {
                                    "$component": "div",
                                    "$id": f"mcp-fn-desc-{full_name}",
                                    "style": {
                                        "fontSize": "12px",
                                        "color": "var(--mutgui-text-dim)",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                        "whiteSpace": "nowrap",
                                    },
                                    "children": first_line,
                                },
                            ],
                        },
                        *detail_block,
                    ],
                })

        ns_rows.append({
            "$component": "div",
            "$id": f"mcp-ns-block-{ns_name}",
            "$children": [
                # Level 1: namespace 行
                {
                    "$component": "div",
                    "$id": f"mcp-ns-row-{ns_name}",
                    "style": {
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "6px",
                        "padding": "4px 0",
                        "cursor": "pointer",
                        "userSelect": "none",
                    },
                    "onClick": Callback(partial(_toggle_ns, ns_name), view="@view"),
                    "$children": [
                        {
                            "$component": "div",
                            "$id": f"mcp-ns-toggle-{ns_name}",
                            "style": {"fontSize": "12px", "width": "14px"},
                            "children": toggle,
                        },
                        {
                            "$component": "div",
                            "$id": f"mcp-ns-name-{ns_name}",
                            "style": {"fontWeight": 600, "fontSize": "13px"},
                            "children": ns_name,
                        },
                        {
                            "$component": "div",
                            "$id": f"mcp-ns-count-{ns_name}",
                            "style": {"fontSize": "12px", "color": "var(--mutgui-text-dim)"},
                            "children": f"({len(funcs)} functions)",
                        },
                    ],
                },
                # Level 2~3: 函数列表（仅展开时）
                *([{
                    "$component": "div",
                    "$id": f"mcp-func-list-{ns_name}",
                    "style": {
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "2px",
                        "paddingLeft": "22px",
                    },
                    "$children": func_rows,
                }] if is_expanded and func_rows else []),
            ],
        })

    if not ns_rows:
        return {
            "$component": "div",
            "$id": "mcp-func-none",
            "style": {"padding": "12px", "color": "var(--mutgui-text-dim)", "fontSize": "13px"},
            "children": "(connected, but no functions discovered)",
        }

    return {
        "$component": "div",
        "$id": "mcp-func-browser",
        "style": {
            "display": "flex",
            "flexDirection": "column",
            "gap": "4px",
        },
        "$children": ns_rows,
    }



def _sanitized_hint(name: str) -> str:
    if not name:
        return ""
    s = _sanitize_ns_name(name)
    if s == name:
        return f"运行时名: {s}"
    return f"运行时名: {s}（已 sanitize）"


def _render_edit(self: MCPSettingsPanel) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    key = self.editing_key
    conn = self._conns.get(key)
    state = _conn_state(self, key)

    items.append({
        "$component": "antd.Space",
        "$id": "mcp-edit-header",
        "style": {"marginBottom": 16, "alignItems": "center"},
        "$children": [
            {
                "$component": "antd.Button",
                "$id": "mcp-back",
                "children": "← Back",
                "onClick": Callback(_back_to_list, view="@view"),
            },
            {
                "$component": "div",
                "style": {"fontWeight": 600},
                "children": key or "(new source)",
            },
            {
                "$component": "antd.Tag",
                "color": "geekblue" if self.form_transport == "stdio" else "cyan",
                "children": self.form_transport,
            },
            {
                "$component": "antd.Tag",
                "color": _state_tag_color(state),
                "children": _state_tag_text(state, conn),
            },
        ],
    })

    # 配置变更横幅
    draft_now, _env_errs = _persist_current_form(self)
    if not self.editing_is_new and _config_changed_at_runtime(draft_now, self.editing_key, conn):
        items.append({
            "$component": "antd.Alert",
            "$id": "mcp-runtime-warn",
            "type": "warning",
            "showIcon": True,
            "message": "配置已修改，需要 Disconnect → Connect 才能生效",
            "style": {"marginBottom": 12},
        })

    # ── 表单 ──
    form_children: list[dict[str, Any]] = [
        {
            "$component": "antd.Form.Item",
            "$id": "mcp-name-item",
            "label": "Name",
            "extra": _sanitized_hint(self.form_name),
            "$children": [{
                "$component": "antd.Input",
                "$id": "mcp-name",
                "value": self.form_name,
                "onChange": Bind(self, "form_name", "$0.target.value"),
            }],
        },
        {
            "$component": "antd.Form.Item",
            "$id": "mcp-transport-item",
            "label": "Transport",
            "$children": [{
                "$component": "antd.Select",
                "$id": "mcp-transport",
                "value": self.form_transport,
                "options": _TRANSPORT_OPTIONS,
                "onChange": Bind(self, "form_transport", "$0"),
            }],
        },
    ]

    if self.form_transport == "stdio":
        form_children.extend([
            {
                "$component": "antd.Form.Item",
                "$id": "mcp-command-item",
                "label": "Command",
                "$children": [{
                    "$component": "antd.Input",
                    "$id": "mcp-command",
                    "value": self.form_command,
                    "placeholder": "npx",
                    "onChange": Bind(self, "form_command", "$0.target.value"),
                }],
            },
            {
                "$component": "antd.Form.Item",
                "$id": "mcp-args-item",
                "label": "Args",
                "extra": "一行一个参数",
                "$children": [{
                    "$component": "antd.Input.TextArea",
                    "$id": "mcp-args",
                    "value": self.form_args_text,
                    "autoSize": {"minRows": 1, "maxRows": 6},
                    "placeholder": "-y\n@anthropic/mcp-server-filesystem",
                    "onChange": Bind(self, "form_args_text", "$0.target.value"),
                }],
            },
            {
                "$component": "antd.Form.Item",
                "$id": "mcp-shell-item",
                "label": "Shell",
                "$children": [{
                    "$component": "antd.Switch",
                    "$id": "mcp-shell",
                    "checked": self.form_shell,
                    "onChange": Bind(self, "form_shell", "$0"),
                }],
            },
            {
                "$component": "antd.Form.Item",
                "$id": "mcp-env-item",
                "label": "Env",
                "extra": "一行一对 KEY=VALUE，# 开头为注释",
                "$children": [{
                    "$component": "antd.Input.TextArea",
                    "$id": "mcp-env",
                    "value": self.form_env_text,
                    "autoSize": {"minRows": 1, "maxRows": 8},
                    "placeholder": "API_KEY=xxx\nLOG_LEVEL=debug",
                    "onChange": Bind(self, "form_env_text", "$0.target.value"),
                }],
            },
        ])
    else:
        form_children.extend([
            {
                "$component": "antd.Form.Item",
                "$id": "mcp-url-item",
                "label": "URL",
                "$children": [{
                    "$component": "antd.Input",
                    "$id": "mcp-url",
                    "value": self.form_url,
                    "placeholder": "http://127.0.0.1:8800/mcp",
                    "onChange": Bind(self, "form_url", "$0.target.value"),
                }],
            },
            {
                "$component": "antd.Form.Item",
                "$id": "mcp-timeout-item",
                "label": "Timeout (s)",
                "$children": [{
                    "$component": "antd.InputNumber",
                    "$id": "mcp-timeout",
                    "value": self.form_timeout,
                    "min": 1,
                    "max": 600,
                    "step": 1,
                    "onChange": Bind(self, "form_timeout", "$0"),
                }],
            },
        ])

    form_children.extend([
        {
            "$component": "antd.Form.Item",
            "$id": "mcp-autostart-item",
            "label": "Autostart",
            "extra": "启动后自动连接",
            "$children": [{
                "$component": "antd.Switch",
                "$id": "mcp-autostart",
                "checked": self.form_autostart,
                "onChange": Bind(self, "form_autostart", "$0"),
            }],
        },
        {
            "$component": "antd.Form.Item",
            "$id": "mcp-cooldown-item",
            "label": "Retry cooldown (s)",
            "extra": "失败后冷却时间，0=禁用自动重试",
            "$children": [{
                "$component": "antd.InputNumber",
                "$id": "mcp-cooldown",
                "value": self.form_retry_cooldown,
                "min": 0,
                "max": 600,
                "step": 1,
                "onChange": Bind(self, "form_retry_cooldown", "$0"),
            }],
        },
    ])

    items.append({
        "$component": "antd.Form",
        "$id": "mcp-edit-form",
        "layout": "horizontal",
        "labelAlign": "left",
        "labelCol": {"style": {"flex": "0 0 130px"}},
        "wrapperCol": {"style": {"flex": "1"}},
        "$children": form_children,
    })

    # ── Functions 区 ──
    total_funcs = 0
    if conn is not None and conn.state == "connected":
        # 聚合同名 namespace 去重计数，与 _render_function_browser 一致
        ns_groups: dict[str, list[Any]] = {}
        all_ns: list[Any] = [conn.namespace]
        all_ns.extend(conn.peer_namespaces)
        for ns in all_ns:
            ns_groups.setdefault(ns.name, []).append(ns)
        for ns_group in ns_groups.values():
            merged: dict[str, Any] = {}
            for ns in ns_group:
                merged.update(ns._functions)
            total_funcs += len(merged)

    func_header_children: list[dict[str, Any]] = [{
        "$component": "div",
        "$id": "mcp-func-title",
        "style": {"fontWeight": 500, "fontSize": "13px"},
        "children": f"Functions ({total_funcs})" if total_funcs else "Functions",
    }]
    if conn is not None and conn.state == "connected":
        func_header_children.append({
            "$component": "antd.Button",
            "$id": "mcp-func-reload",
            "size": "small",
            "loading": self.pending_button.endswith(":reload"),
            "children": "Reload",
            "onClick": Callback(partial(_btn_reload_tools, key), view="@view"),
        })
    items.append({
        "$component": "div",
        "$id": "mcp-func-header",
        "style": {
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "marginTop": 16,
            "marginBottom": 8,
        },
        "$children": func_header_children,
    })
    items.append(_render_function_browser(self, conn))

    items.extend(_render_message(self, margin_bottom=10))

    # ── 操作按钮 ──
    action_left: list[dict[str, Any]] = [{
        "$component": "antd.Button",
        "$id": "mcp-remove",
        "danger": True,
        "disabled": self.editing_is_new,
        "children": "Remove",
        "onClick": Callback(_delete_source, view="@view"),
    }]
    action_right: list[dict[str, Any]] = []

    if not self.editing_is_new:
        pending = self.pending_button.startswith(f"{key}:")
        conn_disabled = pending or state == "connecting"
        if state == "connected":
            action_right.append({
                "$component": "antd.Button",
                "$id": "mcp-conn-disconnect",
                "danger": True,
                "loading": pending and self.pending_button.endswith(":disconnect"),
                "disabled": conn_disabled,
                "children": "Disconnect",
                "onClick": Callback(partial(_btn_disconnect, key), view="@view"),
            })
        elif state == "failed":
            action_right.append({
                "$component": "antd.Button",
                "$id": "mcp-conn-reconnect",
                "type": "primary",
                "loading": pending and self.pending_button.endswith(":reconnect"),
                "disabled": conn_disabled,
                "children": "Reconnect",
                "onClick": Callback(partial(_btn_reconnect, key), view="@view"),
            })
        elif state == "connecting":
            action_right.append({
                "$component": "antd.Button",
                "$id": "mcp-conn-connect",
                "type": "primary",
                "loading": True,
                "disabled": True,
                "children": "Connect",
            })
        else:  # disconnected
            action_right.append({
                "$component": "antd.Button",
                "$id": "mcp-conn-connect",
                "type": "primary",
                "loading": pending and self.pending_button.endswith(":connect"),
                "disabled": conn_disabled,
                "children": "Connect",
                "onClick": Callback(partial(_btn_connect, key), view="@view"),
            })

    action_right.append({
        "$component": "antd.Button",
        "$id": "mcp-save",
        "type": "primary",
        "children": "Save",
        "onClick": Callback(_save_edits, view="@view"),
    })

    items.append({
        "$component": "div",
        "$id": "mcp-edit-actions",
        "style": {
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "marginTop": 16,
            "paddingTop": 12,
            "borderTop": "1px solid var(--mutgui-border)",
        },
        "$children": [
            {"$component": "antd.Space", "$id": "mcp-actions-left", "$children": action_left},
            {"$component": "antd.Space", "$id": "mcp-actions-right", "$children": action_right},
        ],
    })

    return items


# ═══════════════════════════════════════════════════════════════
#  @impl
# ═══════════════════════════════════════════════════════════════


@mutagent.impl(MCPSettingsPanel.__init__)
def __init__(self: MCPSettingsPanel, *, app: Any, agent: Any) -> None:
    super(MCPSettingsPanel, self).__init__()
    self.id = "mcp-settings-panel"
    self._app = app
    self._agent = agent
    self._drafts: dict[str, dict[str, Any]] = {}
    self._conns: dict[str, MCPConnection] = {}
    self._async_error: str = ""
    self.current_step = "list"
    self.editing_key = ""
    self.editing_is_new = False
    self.form_name = ""
    self.form_transport = "stdio"
    self.form_command = ""
    self.form_args_text = ""
    self.form_shell = False
    self.form_env_text = ""
    self.form_url = ""
    self.form_timeout = 30.0
    self.form_autostart = True
    self.form_retry_cooldown = 5.0
    self.error = ""
    self.notice = ""
    self.pending_button = ""
    self.expanded_ns: set[str] = set()
    self.expanded_fn: set[str] = set()
    _load_from_config(self)


@mutagent.impl(MCPSettingsPanel.on_open)
def _on_open(self: MCPSettingsPanel) -> None:
    _load_from_config(self)


@mutagent.impl(MCPSettingsPanel.render)
def render(self: MCPSettingsPanel) -> ViewBlock:
    if self.current_step == "edit":
        children = _render_edit(self)
    else:
        children = _render_list(self)
    return ViewBlock([{
        "$component": "div",
        "$id": "mcp-settings-body",
        "style": {"paddingBottom": 12},
        "$children": children,
    }])
