"""mutagent.core._session_impl -- JSONL session runtime implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import mutobj
from mutio.codec.json import JsonObject, get_field

from .context import AgentContext
from .messages import (
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    TurnEndBlock,
    TurnStartBlock,
)
from .session import AgentSession


@dataclass
class SessionMeta:
    """Session metadata for persisted transcripts."""

    session_id: str = ""
    title: str = ""
    model: str = ""
    cwd: str = ""
    created_at: float = 0.0
    head_entry_id: str = ""
    extra: JsonObject = field(default_factory=JsonObject)


@dataclass
class SessionData:
    """Loaded session payload: transcript context + session metadata."""

    context: AgentContext
    meta: SessionMeta = field(default_factory=SessionMeta)


class SessionRuntime(mutobj.Extension[AgentSession]):
    """AgentSession internal runtime bookkeeping."""

    path: Path | None = None
    persisted_model: str = ""
    created_at: float = 0.0
    head_entry_id: str = ""
    persisted_message_count: int = 0
    is_persisted: bool = False


def _normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _create_session_id() -> str:
    return uuid4().hex[:12]


def _entry_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


def _now_ts() -> float:
    return time.time()


def _to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _from_iso(value: str | None) -> float:
    if not value:
        return 0.0
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return 0.0


def _file_timestamp(ts: float) -> str:
    return _to_iso(ts).replace(":", "-").replace(".", "-")


def _session_filename(ts: float, session_id: str) -> str:
    return f"{_file_timestamp(ts)}_{session_id}.jsonl"


def _append_jsonl(path: Path, entry: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _filter_persisted_blocks(blocks: list[ContentBlock]) -> list[ContentBlock]:
    return [
        block for block in blocks
        if not isinstance(block, (TurnStartBlock, TurnEndBlock))
    ]


def _block_to_dict(block: ContentBlock) -> JsonObject:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {
            "type": "image",
            "data": block.data,
            "media_type": block.media_type,
            "url": block.url,
        }
    if isinstance(block, DocumentBlock):
        return {
            "type": "document",
            "data": block.data,
            "media_type": block.media_type,
        }
    if isinstance(block, ThinkingBlock):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
            "data": block.data,
        }
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "tool_name": block.tool_name,
            "content": block.content,
            "is_error": block.is_error,
            "duration": block.duration,
        }
    raise TypeError(f"Unsupported block type: {type(block)!r}")


def _block_from_dict(data: dict[str, Any]) -> ContentBlock:
    block_type = data.get("type")
    if block_type == "text":
        return TextBlock(text=data.get("text", ""))
    if block_type == "image":
        return ImageBlock(
            data=data.get("data", ""),
            media_type=data.get("media_type", ""),
            url=data.get("url", ""),
        )
    if block_type == "document":
        return DocumentBlock(
            data=data.get("data", ""),
            media_type=data.get("media_type", ""),
        )
    if block_type == "thinking":
        return ThinkingBlock(
            thinking=data.get("thinking", ""),
            signature=data.get("signature", ""),
            data=data.get("data", ""),
        )
    if block_type == "tool_use":
        return ToolUseBlock(
            id=data.get("id", ""),
            name=data.get("name", ""),
            input=data.get("input", {}),
        )
    if block_type == "tool_result":
        return ToolResultBlock(
            tool_use_id=data.get("tool_use_id", ""),
            tool_name=data.get("tool_name", ""),
            content=data.get("content", ""),
            is_error=bool(data.get("is_error", False)),
            duration=float(data.get("duration", 0.0)),
        )
    raise ValueError(f"Unknown block type: {block_type!r}")


def _message_to_dict(message: Message) -> JsonObject:
    blocks = [_block_to_dict(block) for block in _filter_persisted_blocks(message.blocks)]
    return {
        "role": message.role,
        "content": blocks,
        "id": message.id,
        "label": message.label,
        "sender": message.sender,
        "model": message.model,
        "timestamp": message.timestamp,
        "duration": message.duration,
        "input_tokens": message.input_tokens,
        "output_tokens": message.output_tokens,
        "cacheable": message.cacheable,
        "priority": message.priority,
    }


def _message_from_dict(data: JsonObject) -> Message:
    blocks = [_block_from_dict(block) for block in get_field(data, "content", list[dict[str, Any]], default=[])]
    timestamp_raw = data.get("timestamp", 0.0)
    if isinstance(timestamp_raw, str):
        timestamp = _from_iso(timestamp_raw)
    elif isinstance(timestamp_raw, (int, float)):
        timestamp = float(timestamp_raw)
    else:
        timestamp = 0.0
    return Message(
        role=get_field(data, "role", str, default="user"),
        blocks=blocks,
        id=get_field(data, "id", str, default=""),
        label=get_field(data, "label", str, default=""),
        sender=get_field(data, "sender", str, default=""),
        model=get_field(data, "model", str, default=""),
        timestamp=float(timestamp or 0.0),
        duration=get_field(data, "duration", float, default=0.0),
        input_tokens=get_field(data, "input_tokens", int, default=0),
        output_tokens=get_field(data, "output_tokens", int, default=0),
        cacheable=get_field(data, "cacheable", bool, default=True),
        priority=get_field(data, "priority", int, default=0),
    )


def _header_entry(meta: SessionMeta) -> JsonObject:
    session_id = meta.session_id or _create_session_id()
    created_at = meta.created_at or _now_ts()
    return {
        "type": "session",
        "version": 2,
        "id": session_id,
        "timestamp": _to_iso(created_at),
        "cwd": meta.cwd,
        "title": meta.title,
        "model": meta.model,
        "meta": meta.extra,
    }


def _append_parented_entry(
    path: Path,
    *,
    entry_type: str,
    entry_id: str,
    parent_id: str | None,
    timestamp: float,
    payload: JsonObject,
) -> str:
    entry: JsonObject = {
        "type": entry_type,
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": _to_iso(timestamp),
        **payload,
    }
    _append_jsonl(path, entry)
    return entry_id


def _runtime(session: AgentSession) -> SessionRuntime:
    return SessionRuntime.get_or_create(session)


def _write_header(path: str | Path, meta: SessionMeta) -> None:
    file_path = _normalize_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(_header_entry(meta), ensure_ascii=False) + "\n")


def _append_prompt(
    path: str | Path,
    prompt: Message,
    *,
    entry_id: str = "",
    parent_id: str | None = None,
) -> str:
    file_path = _normalize_path(path)
    return _append_parented_entry(
        file_path,
        entry_type="system_prompt",
        entry_id=entry_id or prompt.id or _entry_id("prompt"),
        parent_id=parent_id,
        timestamp=prompt.timestamp or _now_ts(),
        payload={"message": _message_to_dict(prompt)},
    )


def _append_message(
    path: str | Path,
    message: Message,
    *,
    entry_id: str = "",
    parent_id: str | None = None,
) -> str:
    file_path = _normalize_path(path)
    return _append_parented_entry(
        file_path,
        entry_type="message",
        entry_id=entry_id or message.id or _entry_id("message"),
        parent_id=parent_id,
        timestamp=message.timestamp or _now_ts(),
        payload={"message": _message_to_dict(message)},
    )


def _append_model_change(
    path: str | Path,
    *,
    model: str,
    entry_id: str = "",
    parent_id: str | None = None,
    **meta: Any,
) -> str:
    file_path = _normalize_path(path)
    timestamp = meta.pop("timestamp", _now_ts())
    if isinstance(timestamp, str):
        timestamp = _from_iso(timestamp)
    return _append_parented_entry(
        file_path,
        entry_type="model_change",
        entry_id=entry_id or _entry_id("model"),
        parent_id=parent_id,
        timestamp=float(timestamp or _now_ts()),
        payload={"model": model, "meta": meta},
    )



def _load(path: str | Path) -> SessionData:
    file_path = _normalize_path(path)
    context = AgentContext()
    meta = SessionMeta()
    model_changes: list[JsonObject] = []
    head_entry_id = ""

    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry_type = entry.get("type")
            if entry_type == "session":
                meta = SessionMeta(
                    session_id=entry.get("id") or entry.get("session_id", ""),
                    title=entry.get("title", ""),
                    model=entry.get("model", ""),
                    cwd=entry.get("cwd", ""),
                    created_at=_from_iso(entry.get("timestamp")),
                    extra=dict(entry.get("meta", {})),
                )
                continue

            head_entry_id = str(entry.get("id", "") or head_entry_id)
            if entry_type == "system_prompt":
                context.prompts.append(_message_from_dict(entry.get("message", {})))
            elif entry_type == "message":
                context.messages.append(_message_from_dict(entry.get("message", {})))
            elif entry_type == "model_change":
                model = entry.get("model", "")
                meta.model = model or meta.model
                model_changes.append({
                    "id": entry.get("id", ""),
                    "parentId": entry.get("parentId"),
                    "model": model,
                    "timestamp": entry.get("timestamp", ""),
                    **dict(entry.get("meta", {})),
                })

    if model_changes:
        meta.extra = dict(meta.extra)
        meta.extra["model_changes"] = model_changes
    meta.head_entry_id = head_entry_id
    return SessionData(context=context, meta=meta)


@mutobj.impl(AgentSession.start_new)
def agent_session_start_new(
    self: AgentSession,
    *,
    session_dir: str | Path,
    cwd: str,
    model: str,
    session_id: str = "",
) -> None:
    rt = _runtime(self)
    self.dir = _normalize_path(session_dir)
    self.id = session_id or _create_session_id()
    self.cwd = cwd
    self.model = model
    rt.path = None
    rt.persisted_model = model
    rt.created_at = 0.0
    rt.head_entry_id = ""
    rt.persisted_message_count = 0
    rt.is_persisted = False


def _resolve_resume_path(session: AgentSession, value: str | Path) -> Path:
    value_str = str(value)

    # 空字符串 = 取最新 session
    if value_str == "":
        session_dir = session.dir or _normalize_path(Path.home() / ".mutagent" / "sessions")
        matches = sorted(session_dir.glob("*.jsonl"))
        if not matches:
            raise FileNotFoundError(f"No sessions found in {session_dir}")
        return matches[-1]

    candidate = _normalize_path(value_str)
    if ("\\" in value_str or "/" in value_str or value_str.endswith(".jsonl")):
        if not candidate.exists():
            raise FileNotFoundError(f"Session file not found: {candidate}")
        return candidate

    session_dir = session.dir or _normalize_path(Path.home() / ".mutagent" / "sessions")
    matches = sorted(session_dir.glob(f"*_{value_str}.jsonl"))
    if not matches:
        raise FileNotFoundError(f"Session not found for id: {value_str}")
    return matches[-1]


@mutobj.impl(AgentSession.resume)
def agent_session_resume(self: AgentSession, value: str | Path, context: AgentContext) -> Path:
    rt = _runtime(self)
    path = _resolve_resume_path(self, value)
    data = _load(path)
    context.prompts = list(data.context.prompts)
    context.messages = list(data.context.messages)

    rt.path = path
    self.dir = path.parent
    self.id = data.meta.session_id or path.stem.split("_")[-1]
    self.cwd = data.meta.cwd or self.cwd
    rt.persisted_model = data.meta.model or self.model
    if not self.model:
        self.model = data.meta.model
    rt.created_at = data.meta.created_at
    rt.head_entry_id = data.meta.head_entry_id
    rt.persisted_message_count = len(context.messages)
    rt.is_persisted = True
    return path


def _ensure_created(session: AgentSession, context: AgentContext) -> Path:
    rt = _runtime(session)
    if rt.is_persisted and rt.path is not None:
        return rt.path

    created_at = rt.created_at or next(
        (msg.timestamp for msg in context.messages if msg.timestamp),
        _now_ts(),
    )
    session_dir = session.dir or _normalize_path(Path.home() / ".mutagent" / "sessions")
    session_dir.mkdir(parents=True, exist_ok=True)
    session_id = session.id or _create_session_id()
    path = session_dir / _session_filename(created_at, session_id)
    meta = SessionMeta(
        session_id=session_id,
        model=session.model,
        cwd=session.cwd,
        created_at=created_at,
    )
    _write_header(path, meta)
    head_id: str | None = None
    for prompt in context.prompts:
        head_id = _append_prompt(path, prompt, parent_id=head_id)
    if session.model:
        head_id = _append_model_change(path, model=session.model, parent_id=head_id)

    rt.path = path
    session.dir = session_dir
    session.id = meta.session_id
    rt.created_at = created_at
    rt.head_entry_id = head_id or ""
    rt.persisted_model = session.model
    rt.is_persisted = True
    return path


@mutobj.impl(AgentSession.sync)
def agent_session_sync(self: AgentSession, context: AgentContext) -> None:
    rt = _runtime(self)
    new_messages = context.messages[rt.persisted_message_count:]
    if not new_messages:
        return
    path = rt.path
    if path is None or not rt.is_persisted:
        path = _ensure_created(self, context)
        rt = _runtime(self)

    head_id = rt.head_entry_id or None
    if self.model and rt.persisted_model and self.model != rt.persisted_model:
        head_id = _append_model_change(
            path,
            model=self.model,
            parent_id=head_id,
        )
        rt.persisted_model = self.model

    for message in new_messages:
        head_id = _append_message(path, message, parent_id=head_id)

    rt.head_entry_id = head_id or ""
    rt.persisted_message_count = len(context.messages)
