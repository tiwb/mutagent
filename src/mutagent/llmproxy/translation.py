"""Anthropic/OpenAI proxy translation helpers."""

from __future__ import annotations

import re
from typing import Iterator

from mutio.codec import json
from mutio.codec.json import JsonObject, JsonValue

_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")

_input_empty: JsonObject = {}


def normalize_model_name(model: str) -> str:
    return _DATE_SUFFIX_RE.sub("", model)


# ---- request translation ----

def anthropic_request_to_openai(body: JsonObject) -> JsonObject:
    openai_messages: list[JsonObject] = []
    system = body.get("system")
    if isinstance(system, str) and system:
        openai_messages.append({"role": "system", "content": system})
    else:
        blocks = json.narrow_value(system, list[JsonObject], fallback=None)
        if blocks is not None:
            parts: list[str] = []
            for block in blocks:
                block_type = json.get_field(block, "type", str, default="")
                if block_type == "text":
                    tv = json.get_field(block, "text", str, default="")
                    if tv:
                        parts.append(tv)
            text = "\n".join(parts).strip()
            if text:
                openai_messages.append({"role": "system", "content": text})

    for msg in json.get_field(body, "messages", list[JsonObject], default=[]):
        role = json.get_field(msg, "role", str, default="")
        content = msg.get("content")
        if role == "user":
            user_entry = _anthropic_user_to_openai(content)
            if user_entry is not None:
                openai_messages.append(user_entry)
            openai_messages.extend(_anthropic_tool_results_to_openai(content))
        elif role == "assistant":
            assistant_entry = _anthropic_assistant_to_openai(content)
            if assistant_entry is not None:
                openai_messages.append(assistant_entry)

    result: JsonObject = {
        "model": normalize_model_name(str(body.get("model", ""))),
        "messages": openai_messages,
    }
    if "max_tokens" in body:
        result["max_tokens"] = body["max_tokens"]
    tools: list[JsonObject] = json.get_field(body, "tools", list[JsonObject], default=[])
    if tools:
        result["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": json.get_field(tool, "name", str, default=""),
                    "description": json.get_field(tool, "description", str, default=""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for tool in tools
        ]
    tool_choice = body.get("tool_choice")
    if tool_choice:
        result["tool_choice"] = _anthropic_tool_choice_to_openai(tool_choice)
    if body.get("stream"):
        result["stream"] = True
    return result


def openai_request_to_anthropic(body: JsonObject) -> JsonObject:
    messages: list[JsonObject] = []
    system_blocks: list[JsonObject] = []

    for msg in json.get_field(body, "messages", list[JsonObject], default=[]):
        role = json.get_field(msg, "role", str, default="")
        if role == "system":
            text = _openai_content_to_text(msg.get("content"))
            if text:
                system_blocks.append({"type": "text", "text": text})
        elif role == "user":
            blocks = _openai_content_to_anthropic_blocks(msg.get("content"))
            if blocks:
                messages.append({"role": "user", "content": _compact_anthropic_content(blocks)})
        elif role == "assistant":
            blocks: list[JsonObject] = []
            blocks.extend(_openai_content_to_anthropic_blocks(msg.get("content")))
            for tc in json.get_field(msg, "tool_calls", list[JsonObject], default=[]):
                func_raw = tc.get("function")
                if isinstance(func_raw, dict):
                    try:
                        tool_input = json.loads(str(func_raw.get("arguments", "{}")))
                    except json.JSONDecodeError:
                        tool_input = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": json.get_field(tc, "id", str, default=""),
                            "name": json.get_field(func_raw, "name", str, default=""),
                            "input": tool_input,
                        }
                    )
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            tool_result: JsonObject = {
                "type": "tool_result",
                "tool_use_id": json.get_field(msg, "tool_call_id", str, default=""),
                "content": _tool_result_content(msg.get("content")),
            }
            if messages and messages[-1]["role"] == "user":
                existing = _expand_anthropic_content(messages[-1]["content"])
                existing.append(tool_result)
                messages[-1]["content"] = _compact_anthropic_content(existing)
            else:
                messages.append({"role": "user", "content": [tool_result]})

    merged = _merge_consecutive_anthropic_messages(messages)
    result: JsonObject = {
        "model": normalize_model_name(str(body.get("model", ""))),
        "messages": merged,
        "max_tokens": json.get_field(body, "max_completion_tokens", int, default=0)
        or json.get_field(body, "max_tokens", int, default=0)
        or 4096,
    }
    if system_blocks:
        result["system"] = system_blocks
    tools: list[JsonObject] = json.get_field(body, "tools", list[JsonObject], default=[])
    if tools:
        tools_result: list[JsonObject] = []
        for tool in tools:
            func_raw = tool.get("function")
            if isinstance(func_raw, dict):
                tools_result.append({
                    "name": json.get_field(func_raw, "name", str, default=""),
                    "description": json.get_field(func_raw, "description", str, default=""),
                    "input_schema": func_raw.get("parameters", {"type": "object", "properties": {}}),
                })
        result["tools"] = tools_result
    tool_choice = body.get("tool_choice")
    if tool_choice:
        result["tool_choice"] = _openai_tool_choice_to_anthropic(tool_choice)
    if body.get("stream"):
        result["stream"] = True
    return result


# ---- response translation ----

def openai_response_to_anthropic(data: JsonObject, *, model: str = "") -> JsonObject:
    usage_raw = data.get("usage")
    usage: dict[str, int] = _openai_usage_to_anthropic(usage_raw) if isinstance(usage_raw, dict) else {}

    choices: list[JsonObject] = json.get_field(data, "choices", list[JsonObject], default=[])
    if not choices:
        return {
            "id": json.get_field(data, "id", str, default=""),
            "type": "message",
            "role": "assistant",
            "model": model or json.get_field(data, "model", str, default=""),
            "content": [],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": usage,
        }

    choice = choices[0]
    finish_reason = json.get_field(choice, "finish_reason", str, default="")

    stop_reason_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "content_filter",
    }
    content: list[JsonObject] = []

    message_data: JsonObject | None = json.get_field(choice, "message", JsonObject, default=None, fallback=None)
    if message_data is not None:
        text = message_data.get("content") or ""
        if text:
            content.append({"type": "text", "text": text})
        for tool_call in json.get_field(message_data, "tool_calls", list[JsonObject], default=[]):
            func_raw = tool_call.get("function")
            tool_id = json.get_field(tool_call, "id", str, default="")
            if isinstance(func_raw, dict):
                try:
                    tool_input = json.loads(str(func_raw.get("arguments", "{}")))
                except json.JSONDecodeError:
                    tool_input = {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": json.get_field(func_raw, "name", str, default=""),
                        "input": tool_input,
                    }
                )

    return {
        "id": data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": model or data.get("model", ""),
        "content": content,
        "stop_reason": stop_reason_map.get(finish_reason, finish_reason),
        "stop_sequence": None,
        "usage": usage,
    }


def anthropic_response_to_openai(data: JsonObject, *, model: str = "") -> JsonObject:
    text_parts: list[str] = []
    tool_calls: list[JsonObject] = []
    for block in json.get_field(data, "content", list[JsonObject], default=[]):
        block_type = json.get_field(block, "type", str, default="")
        if block_type == "text":
            text = json.get_field(block, "text", str, default="")
            if text:
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": json.get_field(block, "id", str, default=""),
                    "type": "function",
                    "function": {
                        "name": json.get_field(block, "name", str, default=""),
                        "arguments": json.dumps(block.get("input") or _input_empty),
                    },
                }
            )

    finish_reason_map = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
    }
    message: JsonObject = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage_raw = data.get("usage")
    usage = _anthropic_usage_to_openai(usage_raw) if isinstance(usage_raw, dict) else JsonObject()
    stop_reason = json.get_field(data, "stop_reason", str, default="")
    return {
        "id": data.get("id", ""),
        "object": "chat.completion",
        "created": 0,
        "model": model or data.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason_map.get(stop_reason, stop_reason),
            }
        ],
        "usage": usage,
    }


# ---- SSE streaming ----

def openai_sse_to_anthropic_events(
    openai_lines: Iterator[str],
    *,
    model: str = "",
) -> Iterator[tuple[str, str]]:
    state = _OpenAIToAnthropicState(model=model)
    for line in openai_lines:
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            if state.current_block_type:
                yield ("content_block_stop", json.dumps({"type": "content_block_stop", "index": state.block_index}))
            yield (
                "message_delta",
                json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": state.stop_reason or "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": state.output_tokens},
                    }
                ),
            )
            yield ("message_stop", json.dumps({"type": "message_stop"}))
            return
        try:
            raw = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        data: JsonObject = raw

        # usage
        usage_raw = data.get("usage")
        has_usage = isinstance(usage_raw, dict)
        if has_usage:
            state.input_tokens = json.get_field(usage_raw, "prompt_tokens", int, default=state.input_tokens)
            state.output_tokens = json.get_field(usage_raw, "completion_tokens", int, default=state.output_tokens)

        # choices
        choices: list[JsonObject] = json.get_field(data, "choices", list[JsonObject], default=[])
        if not choices:
            if not state.message_started and has_usage:
                state.message_started = True
                yield ("message_start", json.dumps(state.message_start()))
            continue

        choice = choices[0]

        # delta
        delta = json.narrow_value(choice.get("delta"), JsonObject, fallback=None)
        if delta is not None:
            finish_reason = json.get_field(choice, "finish_reason", str, default="")
            if finish_reason:
                state.stop_reason = {
                    "stop": "end_turn",
                    "tool_calls": "tool_use",
                    "length": "max_tokens",
                    "content_filter": "content_filter",
                }.get(finish_reason, finish_reason) or ""
            if not state.message_started:
                state.message_started = True
                yield ("message_start", json.dumps(state.message_start()))
            content = delta.get("content")
            if content:
                if state.current_block_type != "text":
                    if state.current_block_type:
                        yield ("content_block_stop", json.dumps({"type": "content_block_stop", "index": state.block_index}))
                    yield ("content_block_start", json.dumps(state.start_text_block()))
                yield (
                    "content_block_delta",
                    json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": state.block_index,
                            "delta": {"type": "text_delta", "text": content},
                        }
                    ),
                )
            for tc_delta in json.get_field(delta, "tool_calls", list[JsonObject], default=[]):
                if state.current_block_type != "tool_use":
                    if state.current_block_type:
                        yield ("content_block_stop", json.dumps({"type": "content_block_stop", "index": state.block_index}))
                    func_raw2 = tc_delta.get("function")
                    tool_name = ""
                    if isinstance(func_raw2, dict):
                        tool_name = json.get_field(func_raw2, "name", str, default="")
                    yield (
                        "content_block_start",
                        json.dumps(
                            state.start_tool_block(
                                json.get_field(tc_delta, "id", str, default=""),
                                tool_name,
                            )
                        ),
                    )
                func_raw3 = tc_delta.get("function")
                args_chunk = ""
                if isinstance(func_raw3, dict):
                    args_chunk = json.get_field(func_raw3, "arguments", str, default="")
                if args_chunk:
                    yield (
                        "content_block_delta",
                        json.dumps(
                            {
                                "type": "content_block_delta",
                                "index": state.block_index,
                                "delta": {"type": "input_json_delta", "partial_json": args_chunk},
                            }
                        ),
                    )


def anthropic_sse_to_openai_chunks(
    anthropic_lines: Iterator[str],
    *,
    model: str = "",
) -> Iterator[str]:
    state = _AnthropicToOpenAIState(model=model)
    event_type = ""
    for line in anthropic_lines:
        if line.startswith("event: "):
            event_type = line[7:]
            continue
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        try:
            raw = json.loads(data_str)
        except json.JSONDecodeError:
            event_type = ""
            continue
        if not isinstance(raw, dict):
            event_type = ""
            continue
        data: JsonObject = raw

        if event_type == "message_start":
            message: JsonObject | None = json.narrow_value(data.get("message"), JsonObject, fallback=None)
            if message is not None:
                state.message_id = json.get_field(message, "id", str, default="")
                state.model = state.model or json.get_field(message, "model", str, default="")
                usage_data = json.narrow_value(message.get("usage"), JsonObject, fallback=None)
                if usage_data is not None:
                    state.prompt_tokens = json.get_field(usage_data, "input_tokens", int, default=0)
        elif event_type == "content_block_start":
            block_raw = json.narrow_value(data.get("content_block"), JsonObject, fallback=None)
            if block_raw is not None:
                state.current_index = json.get_field(data, "index", int, default=0)
                state.current_block_type = json.get_field(block_raw, "type", str, default="")
                if state.current_block_type == "tool_use":
                    state.tool_names[state.current_index] = json.get_field(block_raw, "name", str, default="")
                    payload = state.chunk_payload(
                        {
                            "tool_calls": [
                                {
                                    "index": state.current_index,
                                    "id": block_raw.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": block_raw.get("name", ""),
                                        "arguments": "",
                                    },
                                }
                            ]
                        }
                    )
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        elif event_type == "content_block_delta":
            block_delta = json.narrow_value(data.get("delta"), JsonObject, fallback=None)
            if block_delta is not None:
                delta_type = json.get_field(block_delta, "type", str, default="")
                if delta_type == "text_delta":
                    payload = state.chunk_payload({"content": json.get_field(block_delta, "text", str, default="")})
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                elif delta_type == "input_json_delta":
                    payload = state.chunk_payload(
                        {
                            "tool_calls": [
                                {
                                    "index": state.current_index,
                                    "function": {
                                        "arguments": json.get_field(block_delta, "partial_json", str, default=""),
                                    },
                                }
                            ]
                        }
                    )
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        elif event_type == "message_delta":
            msg_delta = json.narrow_value(data.get("delta"), JsonObject, fallback=None)
            stop_reason = ""
            if msg_delta is not None:
                stop_reason = json.get_field(msg_delta, "stop_reason", str, default="")
            state.finish_reason = {
                "end_turn": "stop",
                "tool_use": "tool_calls",
                "max_tokens": "length",
                "content_filter": "content_filter",
            }.get(stop_reason, stop_reason) or ""
            delta_usage = json.narrow_value(data.get("usage"), JsonObject, fallback=None)
            if delta_usage is not None:
                state.completion_tokens = json.get_field(delta_usage, "output_tokens", int, default=state.completion_tokens)
                state.prompt_tokens = json.get_field(delta_usage, "input_tokens", int, default=state.prompt_tokens)
            payload = state.finish_payload()
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        elif event_type == "message_stop":
            usage_payload = state.usage_payload()
            if usage_payload is not None:
                yield f"data: {json.dumps(usage_payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        event_type = ""


# ---- SSE summarization ----

def summarize_openai_sse(lines: list[str]) -> JsonObject:
    finish_reason = ""
    usage: dict[str, int] = {}
    for line in lines:
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            continue
        try:
            raw = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        data: JsonObject = raw

        current_usage: JsonObject | None = json.narrow_value(data.get("usage"), JsonObject, fallback=None)
        if current_usage is not None:
            usage["input_tokens"] = json.get_field(current_usage, "prompt_tokens", int, default=usage.get("input_tokens", 0))
            usage["output_tokens"] = json.get_field(current_usage, "completion_tokens", int, default=usage.get("output_tokens", 0))
            prompt_details = json.narrow_value(current_usage.get("prompt_tokens_details"), JsonObject, fallback=None)
            if prompt_details is not None and "cached_tokens" in prompt_details:
                usage["cache_read_input_tokens"] = json.get_field(prompt_details, "cached_tokens", int, default=0)

        for choice in json.get_field(data, "choices", list[JsonObject], default=[]):
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
    return {"finish_reason": finish_reason, "usage": usage}


def summarize_anthropic_sse(lines: list[str]) -> JsonObject:
    event_type = ""
    stop_reason = ""
    usage: dict[str, int] = {}
    for line in lines:
        if line.startswith("event: "):
            event_type = line[7:]
            continue
        if not line.startswith("data: "):
            continue
        try:
            raw = json.loads(line[6:])
        except json.JSONDecodeError:
            event_type = ""
            continue
        if not isinstance(raw, dict):
            event_type = ""
            continue
        data: JsonObject = raw

        if event_type == "message_start":
            start_message = json.narrow_value(data.get("message"), JsonObject, fallback=None)
            if start_message is not None:
                start_usage = json.narrow_value(start_message.get("usage"), JsonObject, fallback=None)
                if start_usage is not None and "input_tokens" in start_usage:
                    usage["input_tokens"] = json.get_field(start_usage, "input_tokens", int, default=0)
        elif event_type == "message_delta":
            md_delta = json.narrow_value(data.get("delta"), JsonObject, fallback=None)
            if md_delta is not None:
                sr = md_delta.get("stop_reason")
                if sr:
                    stop_reason = str(sr)
            md_usage = json.narrow_value(data.get("usage"), JsonObject, fallback=None)
            if md_usage is not None:
                usage["input_tokens"] = max(
                    usage.get("input_tokens", 0),
                    json.get_field(md_usage, "input_tokens", int, default=0),
                )
                usage["output_tokens"] = json.get_field(md_usage, "output_tokens", int, default=0)
        event_type = ""
    return {"stop_reason": stop_reason, "usage": usage}


# ---- internal helpers ----

def _anthropic_user_to_openai(content: JsonValue) -> JsonObject | None:
    blocks = _expand_anthropic_content(content)
    if not blocks:
        return None
    text_parts: list[str] = []
    image_parts: list[JsonObject] = []
    for block in blocks:
        block_type = json.get_field(block, "type", str, default="")
        if block_type == "text":
            text = json.get_field(block, "text", str, default="")
            if text:
                text_parts.append(text)
        elif block_type == "image":
            image_part = _anthropic_image_to_openai(block)
            if image_part is not None:
                image_parts.append(image_part)
    if image_parts:
        payload: list[JsonObject] = []
        if text_parts:
            payload.append({"type": "text", "text": "\n".join(text_parts)})
        payload.extend(image_parts)
        return {"role": "user", "content": payload}
    if text_parts:
        return {"role": "user", "content": "\n".join(text_parts)}
    return None


def _anthropic_tool_results_to_openai(content: JsonValue) -> list[JsonObject]:
    results: list[JsonObject] = []
    for block in _expand_anthropic_content(content):
        if block.get("type") != "tool_result":
            continue
        results.append(
            {
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": _tool_result_content(block.get("content")),
            }
        )
    return results


def _anthropic_assistant_to_openai(content: JsonValue) -> JsonObject | None:
    blocks = _expand_anthropic_content(content)
    if not blocks:
        return None
    text_parts: list[str] = []
    tool_calls: list[JsonObject] = []
    for block in blocks:
        block_type = json.get_field(block, "type", str, default="")
        if block_type == "text":
            text = json.get_field(block, "text", str, default="")
            if text:
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or _input_empty),
                    },
                }
            )
    entry: JsonObject = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        entry["tool_calls"] = tool_calls
    return entry


def _anthropic_image_to_openai(block: JsonObject) -> JsonObject | None:
    source_raw = block.get("source")
    if not isinstance(source_raw, dict):
        return None
    source_type = json.get_field(source_raw, "type", str, default="")
    if source_type == "url":
        url = json.get_field(source_raw, "url", str, default="")
        if url:
            return {"type": "image_url", "image_url": {"url": url}}
    if source_type == "base64":
        data = json.get_field(source_raw, "data", str, default="")
        if data:
            media_type = json.get_field(source_raw, "media_type", str, default="image/png")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            }
    return None


def _openai_content_to_text(content: JsonValue) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                tv = part.get("text", "")
                if isinstance(tv, str):
                    parts.append(tv)
        return "\n".join(parts).strip()
    return ""


def _openai_content_to_anthropic_blocks(content: JsonValue) -> list[JsonObject]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return []
    blocks: list[JsonObject] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type", "")
        if part_type == "text":
            tv = part.get("text", "")
            if isinstance(tv, str) and tv:
                blocks.append({"type": "text", "text": tv})
        elif part_type == "image_url":
            image_raw = part.get("image_url", {})
            if isinstance(image_raw, dict):
                blocks.append(_openai_image_to_anthropic(image_raw))
    return [block for block in blocks if block]


def _openai_image_to_anthropic(image: JsonObject) -> JsonObject:
    url = json.get_field(image, "url", str, default="")
    if url.startswith("data:") and ";base64," in url:
        prefix, data = url.split(";base64,", 1)
        media_type = prefix[5:] or "image/png"
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }
    return {
        "type": "image",
        "source": {"type": "url", "url": url},
    }


def _tool_result_content(content: JsonValue) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                tv = block.get("text", "")
                if isinstance(tv, str):
                    parts.append(tv)
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def _expand_anthropic_content(content: JsonValue) -> list[JsonObject]:
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    if isinstance(content, str) and content:
        return [{"type": "text", "text": content}]
    return []


def _compact_anthropic_content(content: list[JsonObject]) -> str | list[JsonObject]:
    if len(content) == 1 and content[0].get("type") == "text":
        return str(content[0].get("text", ""))
    return content


def _merge_consecutive_anthropic_messages(messages: list[JsonObject]) -> list[JsonObject]:
    if not messages:
        return []
    merged: list[JsonObject] = [messages[0]]
    for message in messages[1:]:
        prev = merged[-1]
        if prev["role"] != message["role"]:
            merged.append(message)
            continue
        prev_blocks = _expand_anthropic_content(prev["content"])
        cur_blocks = _expand_anthropic_content(message["content"])
        prev["content"] = _compact_anthropic_content(prev_blocks + cur_blocks)
    return merged


def _anthropic_tool_choice_to_openai(tool_choice: JsonValue) -> JsonValue:
    if isinstance(tool_choice, str):
        return "required" if tool_choice == "any" else tool_choice
    if not isinstance(tool_choice, dict):
        return tool_choice
    choice_type = tool_choice.get("type", "")
    if choice_type == "any":
        return "required"
    if choice_type == "tool":
        return {
            "type": "function",
            "function": {"name": tool_choice.get("name", "")},
        }
    return choice_type or tool_choice


def _openai_tool_choice_to_anthropic(tool_choice: JsonValue) -> JsonValue:
    if isinstance(tool_choice, str):
        if tool_choice == "required":
            return {"type": "any"}
        return {"type": tool_choice}
    if not isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice.get("type") == "function":
        func_raw = tool_choice.get("function")
        name = ""
        if isinstance(func_raw, dict):
            name = json.get_field(func_raw, "name", str, default="")
        return {
            "type": "tool",
            "name": name,
        }
    return tool_choice


def _openai_usage_to_anthropic(usage: JsonObject) -> dict[str, int]:
    result: dict[str, int] = {}
    if "prompt_tokens" in usage:
        result["input_tokens"] = json.get_field(usage, "prompt_tokens", int, default=0)
    if "completion_tokens" in usage:
        result["output_tokens"] = json.get_field(usage, "completion_tokens", int, default=0)
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict) and "cached_tokens" in prompt_details:
        result["cache_read_input_tokens"] = json.get_field(prompt_details, "cached_tokens", int, default=0)
    return result


def _anthropic_usage_to_openai(usage: JsonObject) -> JsonObject:
    result: JsonObject = {}
    if "input_tokens" in usage:
        result["prompt_tokens"] = json.get_field(usage, "input_tokens", int, default=0)
    if "output_tokens" in usage:
        result["completion_tokens"] = json.get_field(usage, "output_tokens", int, default=0)
    if "cache_read_input_tokens" in usage:
        result["prompt_tokens_details"] = {
            "cached_tokens": json.get_field(usage, "cache_read_input_tokens", int, default=0)
        }
    return result


# ---- SSE state classes ----

class _OpenAIToAnthropicState:
    def __init__(self, *, model: str) -> None:
        self.model = model
        self.block_index = -1
        self.current_block_type = ""
        self.input_tokens = 0
        self.output_tokens = 0
        self.message_started = False
        self.stop_reason = ""

    def message_start(self) -> JsonObject:
        return {
            "type": "message_start",
            "message": {
                "id": "",
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": 0,
                },
            },
        }

    def start_text_block(self) -> JsonObject:
        self.block_index += 1
        self.current_block_type = "text"
        return {
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": {"type": "text", "text": ""},
        }

    def start_tool_block(self, tool_id: str, name: str) -> JsonObject:
        self.block_index += 1
        self.current_block_type = "tool_use"
        return {
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": {},
            },
        }


class _AnthropicToOpenAIState:
    def __init__(self, *, model: str) -> None:
        self.model = model
        self.message_id = ""
        self.role_emitted = False
        self.current_index = 0
        self.current_block_type = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.finish_reason = ""
        self.tool_names: dict[int, str] = {}

    def chunk_payload(self, delta: JsonObject) -> JsonObject:
        choice_delta = dict(delta)
        if not self.role_emitted:
            choice_delta["role"] = "assistant"
            self.role_emitted = True
        return {
            "id": self.message_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": choice_delta,
                    "finish_reason": None,
                }
            ],
        }

    def finish_payload(self) -> JsonObject:
        return {
            "id": self.message_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": self.finish_reason or "stop",
                }
            ],
        }

    def usage_payload(self) -> JsonObject | None:
        if not self.prompt_tokens and not self.completion_tokens:
            return None
        return {
            "id": self.message_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self.model,
            "choices": [],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            },
        }
