"""Tests for session persistence internals and AgentSession."""

from __future__ import annotations

import json
from pathlib import Path

from mutagent.core.context import AgentContext
from mutagent.core._session_impl import (
    SessionData,
    SessionMeta,
    SessionRuntime,
    _append_message,
    _append_model_change,
    _append_prompt,
    _load,
    _resolve_resume_path,
    _write_header,
)
from mutagent.core.messages import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    TurnEndBlock,
    TurnStartBlock,
)
from mutagent.core.session import AgentSession


def test_session_store_roundtrip(tmp_path: Path):
    context = AgentContext()
    context.prompts.append(
        Message(
            role="system",
            blocks=[TextBlock(text="You are helpful.")],
            id="p1",
            label="base",
            priority=100,
        )
    )
    context.messages.extend([
        Message(
            role="user",
            blocks=[TurnStartBlock(turn_id="t1"), TextBlock(text="Inspect x.py")],
            id="m1",
            timestamp=1.0,
        ),
        Message(
            role="assistant",
            blocks=[ToolUseBlock(id="tool_1", name="read_file", input={"path": "x.py"})],
            id="m2",
            model="claude-sonnet",
            timestamp=2.0,
        ),
        Message(
            role="user",
            blocks=[
                ToolResultBlock(
                    tool_use_id="tool_1",
                    tool_name="read_file",
                    content="file content",
                    duration=0.2,
                )
            ],
            id="m3",
            timestamp=3.0,
        ),
        Message(
            role="assistant",
            blocks=[TextBlock(text="Done."), TurnEndBlock(turn_id="t1", duration=1.5)],
            id="m4",
            timestamp=4.0,
        ),
    ])
    data = SessionData(
        context=context,
        meta=SessionMeta(
            session_id="sess_1",
            title="Test session",
            model="claude-sonnet",
            cwd="/home/user/project",
            created_at=123.0,
            extra={"workspace": "mutagent"},
        ),
    )

    path = tmp_path / "session.jsonl"
    _write_header(path, data.meta)
    prev_id: str | None = None
    for prompt in data.context.prompts:
        prev_id = _append_prompt(path, prompt, parent_id=prev_id)
    for message in data.context.messages:
        prev_id = _append_message(path, message, parent_id=prev_id)
    loaded = _load(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    prompt_entry = json.loads(lines[1])
    first_message = json.loads(lines[3])

    assert header["type"] == "session"
    assert prompt_entry["parentId"] is None
    assert first_message["parentId"] == json.loads(lines[2])["id"]

    assert loaded.meta.session_id == "sess_1"
    assert loaded.meta.title == "Test session"
    assert loaded.meta.cwd == "/home/user/project"
    assert loaded.meta.extra["workspace"] == "mutagent"
    assert len(loaded.context.prompts) == 1
    assert len(loaded.context.messages) == 4
    assert loaded.context.messages[0].blocks[0].type == "text"
    assert isinstance(loaded.context.messages[1].blocks[0], ToolUseBlock)
    assert isinstance(loaded.context.messages[2].blocks[0], ToolResultBlock)
    assert loaded.context.messages[2].blocks[0].content == "file content"
    assert len(loaded.context.messages[3].blocks) == 1
    assert loaded.meta.head_entry_id == "m4"


def test_session_store_appends_entries(tmp_path: Path):
    path = tmp_path / "append.jsonl"
    _write_header(
        path,
        SessionMeta(
            session_id="sess_2",
            title="Append test",
            model="gpt",
            created_at=9.0,
        ),
    )
    prompt_id = _append_prompt(
        path,
        Message(role="system", blocks=[TextBlock(text="sys")], id="p1"),
    )
    message_id = _append_message(
        path,
        Message(role="assistant", blocks=[TextBlock(text="hello")], id="m1", timestamp=2.0),
        parent_id=prompt_id,
    )
    change_id = _append_model_change(
        path,
        model="gpt-5",
        parent_id=message_id,
        timestamp=3.0,
        source="manual",
    )

    loaded = _load(path)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert lines[2]["parentId"] == prompt_id
    assert lines[3]["parentId"] == message_id
    assert lines[3]["id"] == change_id
    assert loaded.meta.model == "gpt-5"
    assert loaded.context.prompts[0].id == "p1"
    assert loaded.context.messages[0].id == "m1"
    assert loaded.meta.extra["model_changes"][0]["model"] == "gpt-5"
    assert loaded.meta.extra["model_changes"][0]["source"] == "manual"


def test_agent_session_creates_file_on_first_message(tmp_path: Path):
    context = AgentContext()
    context.prompts.append(Message(role="system", blocks=[TextBlock(text="sys")], id="p1"))
    session = AgentSession()
    session.start_new(session_dir=tmp_path, cwd="/home/user/project", model="gpt-5")

    session.sync(context)
    assert list(tmp_path.glob("*.jsonl")) == []

    context.messages.append(
        Message(
            role="user",
            blocks=[TurnStartBlock(turn_id="t1"), TextBlock(text="hello")],
            id="u1",
            timestamp=10.0,
        )
    )
    session.sync(context)

    runtime = SessionRuntime.get(session)
    assert runtime is not None
    assert runtime.is_persisted is True
    assert runtime.path is not None
    assert runtime.path.name.endswith(f"_{session.id}.jsonl")
    loaded = _load(runtime.path)
    assert [msg.id for msg in loaded.context.messages] == ["u1"]
    assert loaded.meta.model == "gpt-5"
    assert loaded.meta.cwd == "/home/user/project"


def test_agent_session_resume_by_path_and_id(tmp_path: Path):
    context = AgentContext()
    context.prompts.append(Message(role="system", blocks=[TextBlock(text="sys")], id="p1"))
    context.messages.extend([
        Message(role="user", blocks=[TextBlock(text="hi")], id="u1", timestamp=1.0),
        Message(role="assistant", blocks=[TextBlock(text="hello")], id="a1", timestamp=2.0),
    ])
    path = tmp_path / "resume.jsonl"
    data = SessionData(
        context=context,
        meta=SessionMeta(
            session_id="resume123",
            model="gpt-4.1",
            cwd="/home/user/project",
            created_at=1.0,
        ),
    )
    _write_header(path, data.meta)
    prev_id: str | None = None
    for prompt in data.context.prompts:
        prev_id = _append_prompt(path, prompt, parent_id=prev_id)
    for message in data.context.messages:
        prev_id = _append_message(path, message, parent_id=prev_id)
    renamed = path.with_name(f"2026-05-28T00-00-00Z_resume123.jsonl")
    path.rename(renamed)

    resumed_context = AgentContext()
    session = AgentSession()
    session.start_new(session_dir=tmp_path, cwd="/home/user/project", model="gpt-5")
    assert _resolve_resume_path(session, "resume123") == renamed
    assert session.resume(renamed, resumed_context) == renamed
    assert [msg.id for msg in resumed_context.messages] == ["u1", "a1"]
    assert session.id == "resume123"
    runtime = SessionRuntime.get(session)
    assert runtime is not None
    assert runtime.persisted_message_count == 2
    assert runtime.is_persisted is True

    resumed_context.messages.append(
        Message(role="user", blocks=[TextBlock(text="next")], id="u2", timestamp=3.0)
    )
    session.sync(resumed_context)
    reloaded = _load(renamed)
    assert [msg.id for msg in reloaded.context.messages] == ["u1", "a1", "u2"]
