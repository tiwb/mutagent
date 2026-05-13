"""Tests for `mutagent.webui._settings_mcp`.

覆盖：
- env 文本解析（含注释、含 ``=``、引号脱壳、KEY 校验失败）
- args 解析
- 列表/编辑 step 切换
- 列表行按钮按状态映射
- Save 流程（落盘 + 横幅判断）
- Rename / Delete 流程
- autostart=false 的 conn 反查
- sanitized 名冲突校验
"""

from __future__ import annotations

from typing import Any

import pytest

from mutagent.webui._settings_mcp import (
    MCPSettingsPanel,
    _check_name_conflicts,
    _config_changed_at_runtime,
    _draft_to_config,
    _fn_detail,
    _format_args_text,
    _format_env_text,
    _parse_args_text,
    _parse_env_text,
    _persist_current_form,
    _state_tag_color,
)


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────


class _DummyConfig:
    def __init__(self, data: dict | None = None) -> None:
        self._data = data or {}
        self._sets: list[tuple[str, Any]] = []

    def get(self, name: str, *, default: Any = None) -> Any:
        node = self._data
        for key in name.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set(self, name: str, value: Any, *, source: str = "") -> None:
        self._sets.append((name, value))
        node = self._data
        keys = name.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value


class _FakeRegistry:
    def __init__(self) -> None:
        self._namespaces: dict[str, list] = {}


class _FakeSandbox:
    def __init__(self) -> None:
        self._mcp_conns: dict[str, Any] = {}
        self._removed: list[Any] = []
        self._added: list[tuple[Any, Any]] = []
        self._registry = _FakeRegistry()
        self._async_loop = None

    def mcp_connections(self) -> dict[str, Any]:
        return dict(self._mcp_conns)

    def register_mcp_connection(self, name: str, conn: Any) -> None:
        self._mcp_conns[name] = conn

    def unregister_mcp_connection(self, name: str) -> None:
        self._mcp_conns.pop(name, None)

    def add_namespace(self, ns: Any, on_remove=None) -> None:
        self._added.append((ns, on_remove))

    def remove_provider(self, ns: Any) -> bool:
        self._removed.append(ns)
        return True


class _FakeApp:
    def __init__(self, config: _DummyConfig, sandbox: _FakeSandbox,
                 config_path=None) -> None:
        self.config = config
        self.sandbox = sandbox
        self.config_path = config_path


class _FakeAgent:
    def __init__(self, config: _DummyConfig) -> None:
        self.config = config


class _FakeConn:
    """模拟 MCPConnection — 测试只关心状态/namespace/list_tools_metadata。"""

    def __init__(self, name: str, state: str = "disconnected",
                 cfg: dict | None = None,
                 tools: list[dict] | None = None,
                 last_error: str = "") -> None:
        self.ns_name = name
        self.state = state
        self.config = cfg or {"transport": "stdio", "command": "x"}
        self.last_error = last_error
        self.peer_namespaces: list = []
        self._tools = tools or []
        # 仿造 namespace
        self.namespace = type("NS", (), {
            "_functions": {t["name"]: lambda **k: None for t in self._tools},
            "_descriptions": {},
            "name": name,
        })()

    def list_tools_metadata(self) -> list[dict]:
        return [{
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("inputSchema", {}),
            "source_namespace": self.ns_name,
        } for t in self._tools]


def _make_panel(*, mcp_sources: dict | None = None,
                conns: dict[str, _FakeConn] | None = None,
                tmp_path=None) -> tuple[MCPSettingsPanel, _FakeApp]:
    config = _DummyConfig({"mcp_sources": mcp_sources or {}})
    sandbox = _FakeSandbox()
    if conns:
        for k, c in conns.items():
            sandbox.register_mcp_connection(k, c)
    cfg_path = (tmp_path / "config.json") if tmp_path else None
    app = _FakeApp(config, sandbox, config_path=cfg_path)
    agent = _FakeAgent(config)
    panel = MCPSettingsPanel(app=app, agent=agent)
    return panel, app


# ─────────────────────────────────────────────────────────────
#  env / args 解析
# ─────────────────────────────────────────────────────────────


class TestEnvParse:

    def test_basic(self):
        d, e = _parse_env_text("A=1\nB=2")
        assert d == {"A": "1", "B": "2"}
        assert e == []

    def test_blank_and_comment(self):
        d, e = _parse_env_text("# header\n\n  # leading-space comment\nA=1")
        assert d == {"A": "1"}
        assert e == []

    def test_equal_in_value(self):
        d, e = _parse_env_text("URL=https://x.com/?a=b&c=d")
        assert d == {"URL": "https://x.com/?a=b&c=d"}
        assert e == []

    def test_quote_stripping(self):
        d, e = _parse_env_text("A=\"hello world\"\nB='foo bar'")
        assert d == {"A": "hello world", "B": "foo bar"}
        assert e == []

    def test_invalid_key_collected(self):
        d, e = _parse_env_text("1A=x\nA-B=y\nGOOD=ok")
        assert d == {"GOOD": "ok"}
        assert any("invalid key '1A'" in s for s in e)
        assert any("invalid key 'A-B'" in s for s in e)

    def test_missing_equal(self):
        d, e = _parse_env_text("NOEQUAL\nGOOD=ok")
        assert d == {"GOOD": "ok"}
        assert any("missing '='" in s for s in e)

    def test_empty(self):
        d, e = _parse_env_text("")
        assert d == {} and e == []

    def test_format_roundtrip(self):
        env = {"A": "1", "B": "two words"}
        text = _format_env_text(env)
        d, e = _parse_env_text(text)
        assert d == env and e == []


class TestArgsParse:

    def test_one_per_line(self):
        assert _parse_args_text("-y\n@some/pkg") == ["-y", "@some/pkg"]

    def test_blank_lines_skipped(self):
        assert _parse_args_text("\na\n\nb\n") == ["a", "b"]

    def test_format_roundtrip(self):
        args = ["-y", "@a/b"]
        assert _parse_args_text(_format_args_text(args)) == args


# ─────────────────────────────────────────────────────────────
#  draft → config
# ─────────────────────────────────────────────────────────────


class TestDraftToConfig:

    def test_stdio_minimal(self):
        cfg = _draft_to_config({
            "name": "x", "transport": "stdio", "command": "echo",
            "args": [], "shell": False, "env": {},
            "url": "", "timeout": 30.0,
            "autostart": True, "retry_cooldown": 5.0,
        })
        assert cfg == {
            "transport": "stdio",
            "autostart": True,
            "retry_cooldown": 5.0,
            "command": "echo",
        }

    def test_stdio_with_args_env_shell(self):
        cfg = _draft_to_config({
            "name": "x", "transport": "stdio", "command": "echo",
            "args": ["-y", "@p/m"], "shell": True,
            "env": {"K": "V"},
            "url": "", "timeout": 30.0,
            "autostart": False, "retry_cooldown": 0.0,
        })
        assert cfg == {
            "transport": "stdio",
            "autostart": False,
            "retry_cooldown": 0.0,
            "command": "echo",
            "args": ["-y", "@p/m"],
            "shell": True,
            "env": {"K": "V"},
        }

    def test_http_only_keeps_http_fields(self):
        cfg = _draft_to_config({
            "name": "x", "transport": "http",
            "command": "ignored", "args": ["ignored"], "shell": True,
            "env": {"K": "V"},
            "url": "http://x/mcp", "timeout": 60.0,
            "autostart": True, "retry_cooldown": 5.0,
        })
        assert cfg == {
            "transport": "http",
            "autostart": True,
            "retry_cooldown": 5.0,
            "url": "http://x/mcp",
            "timeout": 60.0,
        }
        assert "command" not in cfg
        assert "env" not in cfg


# ─────────────────────────────────────────────────────────────
#  panel 步骤切换 / 加载
# ─────────────────────────────────────────────────────────────


class TestPanelLoad:

    def test_loads_drafts_from_config(self):
        panel, _ = _make_panel(mcp_sources={
            "a": {"transport": "stdio", "command": "echo"},
            "b": {"transport": "http", "url": "http://x/mcp"},
        })
        assert set(panel._drafts.keys()) == {"a", "b"}
        assert panel._drafts["a"]["transport"] == "stdio"
        assert panel._drafts["b"]["url"] == "http://x/mcp"
        assert panel.current_step == "list"

    def test_picks_up_autostart_false_conn_via_sandbox(self):
        """autostart=false 的 source 没注册 namespace，仍能从 mcp_connections() 拿到。"""
        conn = _FakeConn("lazy", state="disconnected")
        panel, _ = _make_panel(
            mcp_sources={"lazy": {"transport": "stdio", "command": "x",
                                  "autostart": False}},
            conns={"lazy": conn},
        )
        assert "lazy" in panel._conns
        assert panel._conns["lazy"] is conn


# ─────────────────────────────────────────────────────────────
#  名字冲突校验
# ─────────────────────────────────────────────────────────────


class TestNameConflicts:

    def test_empty_name_rejected(self):
        panel, _ = _make_panel()
        panel.editing_is_new = True
        assert "empty" in _check_name_conflicts(panel, "").lower()

    def test_duplicate_name_rejected(self):
        panel, _ = _make_panel(mcp_sources={"a": {"command": "x"}})
        panel.editing_is_new = True
        err = _check_name_conflicts(panel, "a")
        assert "already exists" in err

    def test_sanitized_collision_detected(self):
        """`my-fs` 与 `my.fs` sanitize 后都是 `my_fs` → 拒绝。"""
        panel, _ = _make_panel(mcp_sources={"my-fs": {"command": "x"}})
        panel.editing_is_new = True
        err = _check_name_conflicts(panel, "my.fs")
        assert "Sanitized" in err and "my_fs" in err

    def test_self_rename_to_same_name_ok(self):
        """编辑现有 source、保持原名 → 通过。"""
        panel, _ = _make_panel(mcp_sources={"a": {"command": "x"}})
        panel.editing_is_new = False
        panel.editing_key = "a"
        assert _check_name_conflicts(panel, "a") == ""


# ─────────────────────────────────────────────────────────────
#  按钮状态映射
# ─────────────────────────────────────────────────────────────


class TestButtonStates:

    def _row_button_id_label(self, panel: MCPSettingsPanel, key: str
                             ) -> tuple[str, str, bool]:
        from mutagent.webui._settings_mcp import _render_list_row
        row = _render_list_row(panel, key, panel._drafts[key])
        btn = row["$children"][1]
        return btn["$id"], btn["children"], bool(btn.get("disabled"))

    def test_disconnected_shows_connect(self):
        conn = _FakeConn("a", state="disconnected")
        panel, _ = _make_panel(
            mcp_sources={"a": {"command": "x"}}, conns={"a": conn})
        _, label, disabled = self._row_button_id_label(panel, "a")
        assert label == "Connect"
        assert disabled is False

    def test_connected_shows_disconnect(self):
        conn = _FakeConn("a", state="connected", tools=[
            {"name": "f1"}, {"name": "f2"}])
        panel, _ = _make_panel(
            mcp_sources={"a": {"command": "x"}}, conns={"a": conn})
        _, label, _ = self._row_button_id_label(panel, "a")
        assert label == "Disconnect"

    def test_failed_shows_reconnect(self):
        conn = _FakeConn("a", state="failed", last_error="ECONNREFUSED")
        panel, _ = _make_panel(
            mcp_sources={"a": {"command": "x"}}, conns={"a": conn})
        _, label, _ = self._row_button_id_label(panel, "a")
        assert label == "Reconnect"

    def test_connecting_disables_button(self):
        conn = _FakeConn("a", state="connecting")
        panel, _ = _make_panel(
            mcp_sources={"a": {"command": "x"}}, conns={"a": conn})
        _, _, disabled = self._row_button_id_label(panel, "a")
        assert disabled is True

    def test_state_tag_colors(self):
        assert _state_tag_color("connected") == "green"
        assert _state_tag_color("connecting") == "blue"
        assert _state_tag_color("failed") == "red"
        assert _state_tag_color("disconnected") == "default"


# ─────────────────────────────────────────────────────────────
#  Save 横幅 — 配置变更检测
# ─────────────────────────────────────────────────────────────


class TestRuntimeChangeBanner:

    def test_no_change_no_banner(self):
        cfg = {"transport": "stdio", "command": "x"}
        conn = _FakeConn("a", state="connected", cfg=cfg)
        draft = {
            "name": "a", "transport": "stdio", "command": "x",
            "args": [], "shell": False, "env": {},
            "url": "", "timeout": 30.0,
            "autostart": True, "retry_cooldown": 5.0,
        }
        assert _config_changed_at_runtime(draft, "a", conn) is False

    def test_command_change_triggers_banner(self):
        cfg = {"transport": "stdio", "command": "x"}
        conn = _FakeConn("a", state="connected", cfg=cfg)
        draft = {
            "name": "a", "transport": "stdio", "command": "y",  # 改了
            "args": [], "shell": False, "env": {},
            "url": "", "timeout": 30.0,
            "autostart": True, "retry_cooldown": 5.0,
        }
        assert _config_changed_at_runtime(draft, "a", conn) is True

    def test_autostart_only_change_no_banner(self):
        """autostart / retry_cooldown 改动不算运行期关键字段。"""
        cfg = _draft_to_config({
            "name": "a", "transport": "stdio", "command": "x",
            "args": [], "shell": False, "env": {},
            "url": "", "timeout": 30.0,
            "autostart": True, "retry_cooldown": 5.0,
        })
        conn = _FakeConn("a", state="connected", cfg=cfg)
        draft = {
            "name": "a", "transport": "stdio", "command": "x",
            "args": [], "shell": False, "env": {},
            "url": "", "timeout": 30.0,
            "autostart": False, "retry_cooldown": 99.0,  # 改了
        }
        assert _config_changed_at_runtime(draft, "a", conn) is False

    def test_env_change_triggers_banner(self):
        cfg = _draft_to_config({
            "name": "a", "transport": "stdio", "command": "x",
            "args": [], "shell": False, "env": {"K": "v1"},
            "url": "", "timeout": 30.0,
            "autostart": True, "retry_cooldown": 5.0,
        })
        conn = _FakeConn("a", state="connected", cfg=cfg)
        draft = {
            "name": "a", "transport": "stdio", "command": "x",
            "args": [], "shell": False, "env": {"K": "v2"},  # 改了
            "url": "", "timeout": 30.0,
            "autostart": True, "retry_cooldown": 5.0,
        }
        assert _config_changed_at_runtime(draft, "a", conn) is True

    def test_http_conn_with_stdio_residual_fields_no_banner(self):
        """conn.config 含 stdio 残留字段（args/shell/command 空值），归一化后不应触发横幅。"""
        cfg = {
            "transport": "http",
            "url": "http://x/mcp",
            "command": "",      # stdio 残留
            "args": [],          # stdio 残留
            "shell": False,      # stdio 残留
            "env": {},           # stdio 残留
        }
        conn = _FakeConn("a", state="connected", cfg=cfg)
        draft = {
            "name": "a", "transport": "http",
            "command": "", "args": [], "shell": False, "env": {},
            "url": "http://x/mcp", "timeout": 30.0,
            "autostart": True, "retry_cooldown": 5.0,
        }
        assert _config_changed_at_runtime(draft, "a", conn) is False

    def test_disconnected_conn_no_banner(self):
        """conn 未连接 → 不显示横幅（无运行期可影响）。"""
        cfg = {"transport": "stdio", "command": "x"}
        conn = _FakeConn("a", state="disconnected", cfg=cfg)
        draft = {
            "name": "a", "transport": "stdio", "command": "y",
            "args": [], "shell": False, "env": {},
            "url": "", "timeout": 30.0,
            "autostart": True, "retry_cooldown": 5.0,
        }
        assert _config_changed_at_runtime(draft, "a", conn) is False


# ─────────────────────────────────────────────────────────────
#  Save / Delete / Rename 流程
# ─────────────────────────────────────────────────────────────


class TestSaveFlow:

    def test_save_writes_config_file(self, tmp_path):
        from mutagent.webui._settings_mcp import _save_edits, _start_add
        panel, app = _make_panel(tmp_path=tmp_path)
        _start_add("stdio", view=panel)
        panel.form_command = "echo"
        panel.form_name = "echo-source"
        _save_edits(view=panel)
        # 配置已写盘
        cfg_path = tmp_path / "config.json"
        assert cfg_path.exists()
        import json
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert "echo-source" in data["mcp_sources"]
        assert data["mcp_sources"]["echo-source"]["command"] == "echo"
        assert panel.notice.startswith("Saved")
        assert panel.error == ""

    def test_save_rejects_empty_command_for_stdio(self, tmp_path):
        from mutagent.webui._settings_mcp import _save_edits, _start_add
        panel, _ = _make_panel(tmp_path=tmp_path)
        _start_add("stdio", view=panel)
        panel.form_name = "noop"
        panel.form_command = ""
        _save_edits(view=panel)
        assert "command" in panel.error.lower()
        # 没写盘
        assert not (tmp_path / "config.json").exists()

    def test_save_rejects_empty_url_for_http(self, tmp_path):
        from mutagent.webui._settings_mcp import _save_edits, _start_add
        panel, _ = _make_panel(tmp_path=tmp_path)
        _start_add("http", view=panel)
        panel.form_name = "noop"
        panel.form_url = ""
        _save_edits(view=panel)
        assert "url" in panel.error.lower()

    def test_save_rejects_invalid_env(self, tmp_path):
        from mutagent.webui._settings_mcp import _save_edits, _start_add
        panel, _ = _make_panel(tmp_path=tmp_path)
        _start_add("stdio", view=panel)
        panel.form_command = "echo"
        panel.form_name = "x"
        panel.form_env_text = "1BAD=v"  # bad key
        _save_edits(view=panel)
        assert "env" in panel.error.lower()

    def test_save_rejects_sanitized_conflict(self, tmp_path):
        from mutagent.webui._settings_mcp import _save_edits, _start_add
        panel, _ = _make_panel(
            tmp_path=tmp_path,
            mcp_sources={"my-fs": {"transport": "stdio", "command": "x"}},
        )
        _start_add("stdio", view=panel)
        panel.form_command = "echo"
        panel.form_name = "my.fs"  # sanitize 与 my-fs 冲突
        _save_edits(view=panel)
        assert "Sanitized" in panel.error

    def test_delete_removes_source_and_writes_config(self, tmp_path):
        from mutagent.webui._settings_mcp import (
            _delete_source,
            _edit_source,
        )
        panel, app = _make_panel(
            tmp_path=tmp_path,
            mcp_sources={"a": {"transport": "stdio", "command": "x"}},
        )
        _edit_source("a", view=panel)
        _delete_source(view=panel)
        assert "a" not in panel._drafts
        assert panel.current_step == "list"
        assert panel.notice.startswith("Removed")
        cfg_path = tmp_path / "config.json"
        import json
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert data["mcp_sources"] == {}

    def test_rename_updates_dict_key(self, tmp_path):
        from mutagent.webui._settings_mcp import _edit_source, _save_edits
        panel, app = _make_panel(
            tmp_path=tmp_path,
            mcp_sources={"a": {"transport": "stdio", "command": "x"}},
        )
        _edit_source("a", view=panel)
        panel.form_name = "b"
        _save_edits(view=panel)
        assert "a" not in panel._drafts
        assert "b" in panel._drafts
        cfg_path = tmp_path / "config.json"
        import json
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert "b" in data["mcp_sources"]
        assert "a" not in data["mcp_sources"]

    def test_rename_with_running_conn_unregisters_old(self, tmp_path):
        from mutagent.webui._settings_mcp import _edit_source, _save_edits
        conn = _FakeConn("a", state="connected")
        panel, app = _make_panel(
            tmp_path=tmp_path,
            mcp_sources={"a": {"transport": "stdio", "command": "x"}},
            conns={"a": conn},
        )
        _edit_source("a", view=panel)
        panel.form_name = "b"
        _save_edits(view=panel)
        # 旧 conn 应从 sandbox 摘除
        assert conn.namespace in app.sandbox._removed
        assert "a" not in app.sandbox._mcp_conns


# ─────────────────────────────────────────────────────────────
#  渲染冒烟
# ─────────────────────────────────────────────────────────────


class TestRenderSmoke:

    def test_render_list_smoke(self):
        panel, _ = _make_panel(mcp_sources={
            "fs": {"transport": "stdio", "command": "echo"},
        })
        block = panel.render()
        # 顶层结构存在
        assert block.items[0]["$id"] == "mcp-settings-body"
        children = block.items[0]["$children"]
        # 至少包含 list 入口
        ids = [c.get("$id") for c in children]
        assert "mcp-add-actions" in ids
        assert "mcp-list" in ids


class TestFunctionDetailRendering:

    def test_fn_signature_unquotes_string_annotations(self):
        def tool(level: str = "info") -> str:
            return level

        assert _fn_detail(tool, "tool").startswith("tool(level: str = 'info') -> str")

    def test_fn_detail_suppresses_required_when_schema_has_default(self):
        def tool() -> None:
            return None

        tool.__signature__ = None  # type: ignore[attr-defined]
        tool._mcp_description = "desc"  # type: ignore[attr-defined]
        tool._mcp_input_schema = {  # type: ignore[attr-defined]
            "properties": {
                "level": {
                    "type": "string",
                    "default": "info",
                    "description": "Log level",
                },
                "target": {
                    "type": "string",
                    "description": "Target selector",
                },
            },
            "required": ["level", "target"],
        }

        detail = _fn_detail(tool, "browser_console_messages")
        assert "level: string (required)" not in detail
        assert "  level: string" in detail
        assert "  target: string (required)" in detail

    def test_fn_detail_uses_pysandbox_signature_fallback(self):
        def tool(**kwargs) -> None:
            return None

        tool._pysandbox_signature_str = "(level: str = 'INFO') -> str"  # type: ignore[attr-defined]
        detail = _fn_detail(tool, "peer.logs")
        assert detail.startswith("peer.logs(level: str = 'INFO') -> str")

    def test_render_edit_stdio_has_command_args_env(self):
        panel, _ = _make_panel(mcp_sources={
            "fs": {"transport": "stdio", "command": "echo"},
        })
        from mutagent.webui._settings_mcp import _edit_source
        _edit_source("fs", view=panel)
        block = panel.render()
        children = block.items[0]["$children"]
        form = next(c for c in children if c.get("$id") == "mcp-edit-form")
        item_ids = [c.get("$id") for c in form["$children"]]
        assert "mcp-command-item" in item_ids
        assert "mcp-args-item" in item_ids
        assert "mcp-env-item" in item_ids
        # http-only 字段不出现
        assert "mcp-url-item" not in item_ids
        assert "mcp-timeout-item" not in item_ids

    def test_render_edit_http_has_url_timeout(self):
        panel, _ = _make_panel(mcp_sources={
            "svc": {"transport": "http", "url": "http://x/mcp"},
        })
        from mutagent.webui._settings_mcp import _edit_source
        _edit_source("svc", view=panel)
        block = panel.render()
        children = block.items[0]["$children"]
        form = next(c for c in children if c.get("$id") == "mcp-edit-form")
        item_ids = [c.get("$id") for c in form["$children"]]
        assert "mcp-url-item" in item_ids
        assert "mcp-timeout-item" in item_ids
        assert "mcp-command-item" not in item_ids

    def test_render_edit_runtime_change_banner(self):
        """connected + 改 command → 横幅出现。"""
        conn = _FakeConn("fs", state="connected",
                         cfg=_draft_to_config({
                             "name": "fs", "transport": "stdio",
                             "command": "echo", "args": [], "shell": False,
                             "env": {}, "url": "", "timeout": 30.0,
                             "autostart": True, "retry_cooldown": 5.0,
                         }))
        panel, _ = _make_panel(
            mcp_sources={"fs": {"transport": "stdio", "command": "echo"}},
            conns={"fs": conn},
        )
        from mutagent.webui._settings_mcp import _edit_source
        _edit_source("fs", view=panel)
        # 改 command
        panel.form_command = "different"
        block = panel.render()
        children = block.items[0]["$children"]
        ids = [c.get("$id") for c in children]
        assert "mcp-runtime-warn" in ids

    def test_render_edit_functions_section_when_disconnected(self):
        panel, _ = _make_panel(mcp_sources={
            "fs": {"transport": "stdio", "command": "echo"},
        })
        from mutagent.webui._settings_mcp import _edit_source
        _edit_source("fs", view=panel)
        block = panel.render()
        children = block.items[0]["$children"]
        ids = [c.get("$id") for c in children]
        assert "mcp-func-empty" in ids
