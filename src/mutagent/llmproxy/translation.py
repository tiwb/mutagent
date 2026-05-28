"""Anthropic/OpenAI proxy translation helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Iterator

_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


def normalize_model_name(model: str) -> str:
    return _DATE_SUFFIX_RE.sub("", model)


def anthropic_request_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    openai_messages: list[dict[str, Any]] = []
    system = body.get("system")
    if isinstance(system, str) and system:
        openai_messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = "\n".join(
            block.get("text", "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if text:
            openai_messages.append({"role": "system", "content": text})

    for msg in body.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content")
        if role == "user":
            user_entry = _anthropic_user_to_openai(content)
            if user_entry is not None:
                openai_messages.append(user_entry)
            openai_messages.extend(_anthropic_tool_results_to_openai(content))
            continue
        if role != "assistant":
            continue
        assistant_entry = _anthropic_assistant_to_openai(content)
        if assistant_entry is not None:
            openai_messages.append(assistant_entry)

    result: dict[str, Any] = {
        "model": normalize_model_name(str(body.get("model", ""))),
        "messages": openai_messages,
    }
    if "max_tokens" in body:
        result["max_tokens"] = body["max_tokens"]
    if "tools" in body:
        result["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for tool in body.get("tools", [])
        ]
    tool_choice = body.get("tool_choice")
    if tool_choice:
        result["tool_choice"] = _anthropic_tool_choice_to_openai(tool_choice)
    if body.get("stream"):
        result["stream"] = True
    return result


def openai_request_to_anthropic(body: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system_blocks: list[dict[str, Any]] = []

    for msg in body.get("messages", []):
        role = msg.get("role", "")
        if role == "system":
            text = _openai_content_to_text(msg.get("content"))
            if text:
                system_blocks.append({"type": "text", "text": text})
            continue
        if role == "user":
            blocks = _openai_content_to_anthropic_blocks(msg.get("content"))
            if blocks:
                messages.append({"role": "user", "content": _compact_anthropic_content(blocks)})
            continue
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            blocks.extend(_openai_content_to_anthropic_blocks(msg.get("content")))
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                try:
                    tool_input = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_input = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": tool_input,
                    }
                )
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            tool_result = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": _tool_result_content(msg.get("content")),
            }
            if messages and messages[-1]["role"] == "user":
                existing = _expand_anthropic_content(messages[-1]["content"])
                existing.append(tool_result)
                messages[-1]["content"] = _compact_anthropic_content(existing)
            else:
                messages.append({"role": "user", "content": [tool_result]})

    merged = _merge_consecutive_anthropic_messages(messages)
    result: dict[str, Any] = {
        "model": normalize_model_name(str(body.get("model", ""))),
        "messages": merged,
        "max_tokens": int(
            body.get("max_completion_tokens")
            or body.get("max_tokens")
            or 4096
        ),
    }
    if system_blocks:
        result["system"] = system_blocks
    if "tools" in body:
        result["tools"] = [
            {
                "name": tool.get("function", {}).get("name", ""),
                "description": tool.get("function", {}).get("description", ""),
                "input_schema": tool.get("function", {}).get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            for tool in body.get("tools", [])
        ]
    tool_choice = body.get("tool_choice")
    if tool_choice:
        result["tool_choice"] = _openai_tool_choice_to_anthropic(tool_choice)
    if body.get("stream"):
        result["stream"] = True
    return result


def openai_response_to_anthropic(data: dict[str, Any], *, model: str = "") -> dict[str, Any]:
    choice = data.get("choices", [{}])[0]
    message_data = choice.get("message", {})
    finish_reason = choice.get("finish_reason") or ""

    stop_reason_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "content_filter",
    }
    content: list[dict[str, Any]] = []
    text = message_data.get("content") or ""
    if text:
        content.append({"type": "text", "text": text})
    for tool_call in message_data.get("tool_calls", []):
        func = tool_call.get("function", {})
        try:
            tool_input = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            tool_input = {}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id", ""),
                "name": func.get("name", ""),
                "input": tool_input,
            }
        )

    usage = _openai_usage_to_anthropic(data.get("usage", {}))
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


def anthropic_response_to_openai(data: dict[str, Any], *, model: str = "") -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in data.get("content", []):
        block_type = block.get("type", "")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )

    finish_reason_map = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
    }
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = _anthropic_usage_to_openai(data.get("usage", {}))
    return {
        "id": data.get("id", ""),
        "object": "chat.completion",
        "created": 0,
        "model": model or data.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason_map.get(
                    data.get("stop_reason", ""), data.get("stop_reason")
                ),
            }
        ],
        "usage": usage,
    }


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
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if data.get("usage"):
            usage = data["usage"]
            state.input_tokens = usage.get("prompt_tokens", state.input_tokens)
            state.output_tokens = usage.get("completion_tokens", state.output_tokens)
        choices = data.get("choices", [])
        if not choices:
            if not state.message_started and data.get("usage"):
                state.message_started = True
                yield ("message_start", json.dumps(state.message_start()))
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
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
        for tc_delta in delta.get("tool_calls", []):
            if state.current_block_type != "tool_use":
                if state.current_block_type:
                    yield ("content_block_stop", json.dumps({"type": "content_block_stop", "index": state.block_index}))
                yield (
                    "content_block_start",
                    json.dumps(
                        state.start_tool_block(
                            tc_delta.get("id", ""),
                            tc_delta.get("function", {}).get("name", ""),
                        )
                    ),
                )
            args_chunk = tc_delta.get("function", {}).get("arguments", "")
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
            data = json.loads(data_str)
        except json.JSONDecodeError:
            event_type = ""
            continue
        if event_type == "message_start":
            message = data.get("message", {})
            state.message_id = message.get("id", "")
            state.model = state.model or message.get("model", "")
            usage = message.get("usage", {})
            state.prompt_tokens = int(usage.get("input_tokens", 0))
        elif event_type == "content_block_start":
            block = data.get("content_block", {})
            state.current_index = int(data.get("index", 0))
            state.current_block_type = block.get("type", "")
            if state.current_block_type == "tool_use":
                state.tool_names[state.current_index] = block.get("name", "")
                payload = state.chunk_payload(
                    {
                        "tool_calls": [
                            {
                                "index": state.current_index,
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": "",
                                },
                            }
                        ]
                    }
                )
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        elif event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "text_delta":
                payload = state.chunk_payload({"content": delta.get("text", "")})
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            elif delta_type == "input_json_delta":
                payload = state.chunk_payload(
                    {
                        "tool_calls": [
                            {
                                "index": state.current_index,
                                "function": {
                                    "arguments": delta.get("partial_json", ""),
                                },
                            }
                        ]
                    }
                )
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        elif event_type == "message_delta":
            stop_reason = data.get("delta", {}).get("stop_reason", "")
            state.finish_reason = {
                "end_turn": "stop",
                "tool_use": "tool_calls",
                "max_tokens": "length",
                "content_filter": "content_filter",
            }.get(stop_reason, stop_reason) or ""
            usage = data.get("usage", {})
            state.completion_tokens = int(usage.get("output_tokens", state.completion_tokens))
            if usage.get("input_tokens"):
                state.prompt_tokens = int(usage["input_tokens"])
            payload = state.finish_payload()
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        elif event_type == "message_stop":
            usage_payload = state.usage_payload()
            if usage_payload is not None:
                yield f"data: {json.dumps(usage_payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        event_type = ""


def summarize_openai_sse(lines: list[str]) -> dict[str, Any]:
    finish_reason = ""
    usage: dict[str, int] = {}
    for line in lines:
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        current_usage = data.get("usage", {})
        if "prompt_tokens" in current_usage:
            usage["input_tokens"] = int(current_usage["prompt_tokens"])
        if "completion_tokens" in current_usage:
            usage["output_tokens"] = int(current_usage["completion_tokens"])
        prompt_details = current_usage.get("prompt_tokens_details", {})
        if isinstance(prompt_details, dict) and "cached_tokens" in prompt_details:
            usage["cache_read_input_tokens"] = int(prompt_details["cached_tokens"])
        for choice in data.get("choices", []):
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
    return {"finish_reason": finish_reason, "usage": usage}


def summarize_anthropic_sse(lines: list[str]) -> dict[str, Any]:
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
            data = json.loads(line[6:])
        except json.JSONDecodeError:
            event_type = ""
            continue
        if event_type == "message_start":
            start_usage = data.get("message", {}).get("usage", {})
            if "input_tokens" in start_usage:
                usage["input_tokens"] = int(start_usage["input_tokens"])
        elif event_type == "message_delta":
            delta = data.get("delta", {})
            if delta.get("stop_reason"):
                stop_reason = str(delta["stop_reason"])
            update = data.get("usage", {})
            if "input_tokens" in update:
                usage["input_tokens"] = max(
                    usage.get("input_tokens", 0),
                    int(update["input_tokens"]),
                )
            if "output_tokens" in update:
                usage["output_tokens"] = int(update["output_tokens"])
        event_type = ""
    return {"stop_reason": stop_reason, "usage": usage}


def _anthropic_user_to_openai(content: Any) -> dict[str, Any] | None:
    blocks = _expand_anthropic_content(content)
    if not blocks:
        return None
    text_parts: list[str] = []
    image_parts: list[dict[str, Any]] = []
    for block in blocks:
        block_type = block.get("type", "")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
        elif block_type == "image":
            image_part = _anthropic_image_to_openai(block)
            if image_part is not None:
                image_parts.append(image_part)
    if image_parts:
        payload: list[dict[str, Any]] = []
        if text_parts:
            payload.append({"type": "text", "text": "\n".join(text_parts)})
        payload.extend(image_parts)
        return {"role": "user", "content": payload}
    if text_parts:
        return {"role": "user", "content": "\n".join(text_parts)}
    return None


def _anthropic_tool_results_to_openai(content: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
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


def _anthropic_assistant_to_openai(content: Any) -> dict[str, Any] | None:
    blocks = _expand_anthropic_content(content)
    if not blocks:
        return None
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in blocks:
        block_type = block.get("type", "")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
    entry: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        entry["tool_calls"] = tool_calls
    return entry


def _anthropic_image_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source", {})
    if source.get("type") == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    if source.get("type") == "base64" and source.get("data"):
        media_type = source.get("media_type", "image/png")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{source['data']}"},
        }
    return None


def _openai_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return ""


def _openai_content_to_anthropic_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return []
    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type", "")
        if part_type == "text":
            text = part.get("text", "")
            if text:
                blocks.append({"type": "text", "text": text})
        elif part_type == "image_url":
            blocks.append(_openai_image_to_anthropic(part.get("image_url", {})))
    return [block for block in blocks if block]


def _openai_image_to_anthropic(image: dict[str, Any]) -> dict[str, Any]:
    url = image.get("url", "")
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


def _tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return json.dumps(content, ensure_ascii=False)


def _expand_anthropic_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    if isinstance(content, str) and content:
        return [{"type": "text", "text": content}]
    return []


def _compact_anthropic_content(content: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if len(content) == 1 and content[0].get("type") == "text":
        return str(content[0].get("text", ""))
    return content


def _merge_consecutive_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        return []
    merged: list[dict[str, Any]] = [messages[0]]
    for message in messages[1:]:
        prev = merged[-1]
        if prev["role"] != message["role"]:
            merged.append(message)
            continue
        prev_blocks = _expand_anthropic_content(prev["content"])
        cur_blocks = _expand_anthropic_content(message["content"])
        prev["content"] = _compact_anthropic_content(prev_blocks + cur_blocks)
    return merged


def _anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
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


def _openai_tool_choice_to_anthropic(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        if tool_choice == "required":
            return {"type": "any"}
        return {"type": tool_choice}
    if not isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice.get("type") == "function":
        return {
            "type": "tool",
            "name": tool_choice.get("function", {}).get("name", ""),
        }
    return tool_choice


def _openai_usage_to_anthropic(usage: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    if "prompt_tokens" in usage:
        result["input_tokens"] = int(usage["prompt_tokens"])
    if "completion_tokens" in usage:
        result["output_tokens"] = int(usage["completion_tokens"])
    prompt_details = usage.get("prompt_tokens_details", {})
    if isinstance(prompt_details, dict) and "cached_tokens" in prompt_details:
        result["cache_read_input_tokens"] = int(prompt_details["cached_tokens"])
    return result


def _anthropic_usage_to_openai(usage: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "input_tokens" in usage:
        result["prompt_tokens"] = int(usage["input_tokens"])
    if "output_tokens" in usage:
        result["completion_tokens"] = int(usage["output_tokens"])
    if "cache_read_input_tokens" in usage:
        result["prompt_tokens_details"] = {
            "cached_tokens": int(usage["cache_read_input_tokens"])
        }
    return result


class _OpenAIToAnthropicState:
    def __init__(self, *, model: str) -> None:
        self.model = model
        self.block_index = -1
        self.current_block_type = ""
        self.input_tokens = 0
        self.output_tokens = 0
        self.message_started = False
        self.stop_reason = ""

    def message_start(self) -> dict[str, Any]:
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

    def start_text_block(self) -> dict[str, Any]:
        self.block_index += 1
        self.current_block_type = "text"
        return {
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": {"type": "text", "text": ""},
        }

    def start_tool_block(self, tool_id: str, name: str) -> dict[str, Any]:
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

    def chunk_payload(self, delta: dict[str, Any]) -> dict[str, Any]:
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

    def finish_payload(self) -> dict[str, Any]:
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

    def usage_payload(self) -> dict[str, Any] | None:
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
