"""Tests for mutagent.main CLI routing."""

from __future__ import annotations

import sys

import pytest

import mutagent.app.app as main_module
from mutagent.cli import main


class DummyApp:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []
        self.config = type("C", (), {"get": lambda self, name, default=None: default, "resolve_model": lambda self: None})()

    def load_config(self, path: str) -> None:
        self.calls.append(("load_config", path))

    def setup_agent(self) -> None:
        self.calls.append(("setup_agent", None))
        self.agent = type("A", (), {"list_models": lambda self: []})()


def test_main_defaults_to_headless(monkeypatch):
    app = DummyApp()
    monkeypatch.setattr(main_module, "App", lambda: app)
    monkeypatch.setattr(sys, "argv", ["mutagent", "--config", "cfg.json"])

    main.main()

    assert app.calls == []


def test_main_routes_to_webui(monkeypatch):
    app = DummyApp()
    dispatch_calls: list[tuple] = []

    def _fake_dispatch(parser, args):
        dispatch_calls.append(("dispatch_webui", args))

    monkeypatch.setattr(main_module, "App", lambda: app)
    monkeypatch.setattr("mutagent.cli.webui.dispatch_webui", _fake_dispatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mutagent", "--config", "cfg.json", "webui", "--host", "0.0.0.0", "--port", "9000", "--no-browser"],
    )

    main.main()

    assert len(dispatch_calls) == 1
    dispatched_args = dispatch_calls[0][1]
    assert dispatched_args.command == "webui"
    assert dispatched_args.host == "0.0.0.0"
    assert dispatched_args.port == 9000
    assert dispatched_args.no_browser is True


def test_main_rejects_headless_with_webui(monkeypatch):
    monkeypatch.setattr(main_module, "App", DummyApp)
    monkeypatch.setattr(sys, "argv", ["mutagent", "--headless", "webui"])

    with pytest.raises(SystemExit):
        main.main()
