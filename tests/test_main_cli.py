"""Tests for mutagent.main CLI routing."""

from __future__ import annotations

import sys

import pytest

import mutagent.main as main_module


class DummyApp:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def load_config(self, path: str) -> None:
        self.calls.append(("load_config", path))

    def run(self) -> None:
        self.calls.append(("run", None))

    def run_webui(self, *, host: str, port: int, open_browser: bool) -> None:
        self.calls.append(("run_webui", (host, port, open_browser)))


def test_main_defaults_to_headless(monkeypatch):
    app = DummyApp()
    monkeypatch.setattr(main_module, "App", lambda: app)
    monkeypatch.setattr(sys, "argv", ["mutagent", "--config", "cfg.json"])

    main_module.main()

    assert app.calls == [
        ("load_config", "cfg.json"),
        ("run", None),
    ]


def test_main_routes_to_webui(monkeypatch):
    app = DummyApp()
    monkeypatch.setattr(main_module, "App", lambda: app)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mutagent", "--config", "cfg.json", "webui", "--host", "0.0.0.0", "--port", "9000", "--no-browser"],
    )

    main_module.main()

    assert app.calls == [
        ("load_config", "cfg.json"),
        ("run_webui", ("0.0.0.0", 9000, False)),
    ]


def test_main_rejects_headless_with_webui(monkeypatch):
    monkeypatch.setattr(main_module, "App", DummyApp)
    monkeypatch.setattr(sys, "argv", ["mutagent", "--headless", "webui"])

    with pytest.raises(SystemExit):
        main_module.main()
