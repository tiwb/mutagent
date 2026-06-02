"""CLI terminal session lifecycle tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from mutagent.app.app import App
from mutagent.cli.terminal import _build_agent_session, add_terminal_subcommand
from mutagent.core.context import AgentContext
from mutagent.core._session_impl import SessionData, SessionMeta, _resolve_resume_path, _save
from mutagent.core.messages import Message, TextBlock
from mutagent.core.session import AgentSession


def test_terminal_subcommand_accepts_resume():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_terminal_subcommand(subparsers)

    args = parser.parse_args(["terminal", "--resume", "abc123"])

    assert args.command == "terminal"
    assert args.resume == "abc123"


def test_terminal_subcommand_accepts_resume_without_value():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_terminal_subcommand(subparsers)

    args = parser.parse_args(["terminal", "--resume"])

    assert args.command == "terminal"
    assert args.resume == ""


def test_resume_lookup_matches_session_suffix(tmp_path: Path):
    context = AgentContext()
    context.messages.append(Message(role="user", blocks=[TextBlock(text="hi")], id="u1"))
    path = _save(
        tmp_path / "seed.jsonl",
        SessionData(
            context=context,
            meta=SessionMeta(
                session_id="abc123",
                model="gpt-5",
                cwd="/home/user/project",
                created_at=1.0,
            ),
        ),
    )
    target = tmp_path / "2026-05-28T00-00-00Z_abc123.jsonl"
    path.rename(target)

    session = AgentSession()
    session.start_new(session_dir=tmp_path, cwd="/home/user/project", model="gpt-5")

    assert _resolve_resume_path(session, "abc123") == target


def test_build_agent_session_resumes_latest_for_empty_resume_value(tmp_path: Path, monkeypatch):
    context = AgentContext()
    context.messages.append(Message(role="user", blocks=[TextBlock(text="hi")], id="u1"))
    _save(
        tmp_path / "2026-05-28T00-00-00Z_latest123.jsonl",
        SessionData(
            context=context,
            meta=SessionMeta(
                session_id="latest123",
                model="gpt-5",
                cwd="/home/user/project",
                created_at=1.0,
            ),
        ),
    )
    app = App()
    object.__setattr__(
        app,
        "agent",
        SimpleNamespace(
            model="gpt-5",
            context=AgentContext(),
        ),
    )
    monkeypatch.setattr("mutagent.cli.terminal._default_session_dir", lambda: tmp_path)

    session = _build_agent_session(app, "")

    assert session.id == "latest123"
    assert [msg.id for msg in app.agent.context.messages] == ["u1"]
