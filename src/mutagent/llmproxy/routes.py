"""Optional HTTP Views for the mutagent LLM proxy."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from mutio.codec import json

from mutio.codec.json import JsonArray, JsonObject

from mutagent.app.config import Config
from mutagent.core._llm_impl_anthropic import AnthropicApiClient
from mutagent.core._llm_impl_copilot import CopilotApiClient
from mutagent.core._llm_impl_openai import OpenAIApiClient
from mutagent.core.llm import LLMApiClient
from mutio.net.client import HttpClient
from mutio.net.server import HTMLResponse, JSONResponse, Request, Response, StreamingResponse, View

from .logging import ProxyLogger
from .translation import (
    anthropic_request_to_openai,
    anthropic_response_to_openai,
    anthropic_sse_to_openai_chunks,
    normalize_model_name,
    openai_request_to_anthropic,
    openai_response_to_anthropic,
    openai_sse_to_anthropic_events,
    summarize_anthropic_sse,
    summarize_openai_sse,
)

logger = logging.getLogger(__name__)


@dataclass
class ResolvedModel:
    request_model: str
    backend_model: str
    provider_name: str
    provider_type: str
    provider: LLMApiClient


class LLMProxyRuntime:
    """Shared config + provider cache for proxy Views."""

    def __init__(self, config: Config, *, log_dir: Path | str | None = None) -> None:
        self.config = config
        self.log_dir = Path(log_dir) if log_dir is not None else _default_log_dir(config)
        self.logger = ProxyLogger(self.log_dir)
        self._providers_signature = ""
        self._provider_cache: dict[str, LLMApiClient] = {}

    def list_models(self) -> list[JsonObject]:
        self._refresh_cache_if_needed()
        return self.config.list_models()

    def resolve_model(self, model_name: str) -> ResolvedModel | None:
        self._refresh_cache_if_needed()
        normalized = normalize_model_name(model_name)
        providers = self.config.root.get_field("providers", JsonObject, default=JsonObject())
        for provider_name in providers:
            provider_conf = json.get_field(providers, provider_name, JsonObject, default={})
            match = self._match_provider_model(model_name, normalized, provider_conf)
            if match is None:
                continue
            provider = self._get_provider(provider_name, provider_conf, match)
            return ResolvedModel(
                request_model=model_name,
                backend_model=match,
                provider_name=provider_name,
                provider_type=str(provider_conf.get("type", "Anthropic")),
                provider=provider,
            )
        return None

    def _refresh_cache_if_needed(self) -> None:
        providers = self.config.root.get_field("providers", JsonObject, default=JsonObject())
        signature = json.dumps(providers, ensure_ascii=False, sort_keys=True)
        if signature == self._providers_signature:
            return
        self._providers_signature = signature
        self._provider_cache.clear()

    def _match_provider_model(
        self,
        requested_model: str,
        normalized_model: str,
        provider_conf: JsonObject,
    ) -> str | None:
        # Models can be list[str] (direct IDs) or list[dict[str, str]] (alias map)
        models_list = json.get_field(provider_conf, "models", JsonArray, default=None, fallback=None)
        if models_list is not None:
            if not models_list:
                return None
            for item in models_list:
                if isinstance(item, str):
                    if item == requested_model or normalize_model_name(item) == normalized_model:
                        return item
                elif isinstance(item, dict):
                    for alias_raw, model_id_raw in item.items():
                        if isinstance(model_id_raw, str):
                            if alias_raw == requested_model or normalize_model_name(alias_raw) == normalized_model:
                                return model_id_raw
                            if model_id_raw == requested_model or normalize_model_name(model_id_raw) == normalized_model:
                                return model_id_raw
            return None
        # Try dict format (alias → model_id mapping)
        models_dict_raw = provider_conf.get("models")
        if isinstance(models_dict_raw, dict):
            for alias_raw, model_id_raw in models_dict_raw.items():
                if isinstance(model_id_raw, str):
                    if alias_raw == requested_model or normalize_model_name(alias_raw) == normalized_model:
                        return model_id_raw
                    if model_id_raw == requested_model or normalize_model_name(model_id_raw) == normalized_model:
                        return model_id_raw
        return None

    def _get_provider(
        self,
        provider_name: str,
        provider_conf: JsonObject,
        model_id: str,
    ) -> LLMApiClient:
        cache_key = json.dumps(
            {
                "provider_name": provider_name,
                "provider_conf": provider_conf,
                "model_id": model_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cached = self._provider_cache.get(cache_key)
        if cached is not None:
            return cached
        spec = {k: v for k, v in provider_conf.items() if k != "models"}
        spec["provider_name"] = provider_name
        spec["model_id"] = model_id
        provider = LLMApiClient.from_spec(spec)
        self._provider_cache[cache_key] = provider
        return provider


_runtime: LLMProxyRuntime | None = None


def configure_llm_proxy(config: Config, *, log_dir: Path | str | None = None) -> LLMProxyRuntime:
    global _runtime
    _runtime = LLMProxyRuntime(config, log_dir=log_dir)
    return _runtime


def get_llm_proxy_runtime() -> LLMProxyRuntime | None:
    return _runtime


def reset_llm_proxy_runtime() -> None:
    global _runtime
    if _runtime is not None:
        _runtime.logger.close()
    _runtime = None


class LlmInfoView(View):
    path = ("/llm", "/llm/")

    async def get(self, request: Request) -> Response:
        return HTMLResponse(_render_info_page())


class LlmModelsView(View):
    path = "/llm/v1/models"

    async def get(self, request: Request) -> Response:
        runtime = get_llm_proxy_runtime()
        if runtime is None:
            return _error_response(
                "openai",
                "LLM proxy is not configured. Call configure_llm_proxy(config) first.",
                status_code=503,
                error_code="proxy_not_configured",
            )
        data = [
            {
                "id": model["name"],
                "object": "model",
                "created": 0,
                "owned_by": model["provider_name"],
                "model_id": model["model_id"],
                "type": model["type"],
            }
            for model in runtime.list_models()
        ]
        return JSONResponse({"object": "list", "data": data})


class LlmMessagesView(View):
    path = "/llm/v1/messages"

    async def post(self, request: Request) -> Response | StreamingResponse:
        body = json.narrow_value(await request.json(), JsonObject)
        return await _proxy_request(body, client_format="anthropic")


class LlmCompletionsView(View):
    path = "/llm/v1/chat/completions"

    async def post(self, request: Request) -> Response | StreamingResponse:
        body = json.narrow_value(await request.json(), JsonObject)
        return await _proxy_request(body, client_format="openai")


async def _proxy_request(
    body: JsonObject,
    *,
    client_format: str,
) -> Response | StreamingResponse:
    runtime = get_llm_proxy_runtime()
    if runtime is None:
        return _error_response(
            client_format,
            "LLM proxy is not configured. Call configure_llm_proxy(config) first.",
            status_code=503,
            error_code="proxy_not_configured",
        )
    request_model = str(body.get("model", "")).strip()
    if not request_model:
        return _error_response(
            client_format,
            "Missing required field: model",
            status_code=400,
            error_code="missing_model",
        )
    resolved = runtime.resolve_model(request_model)
    if resolved is None:
        return _error_response(
            client_format,
            f"Model not found: {request_model}",
            status_code=404,
            error_code="model_not_found",
        )

    backend_format, base_url, headers = _backend_request_info(resolved.provider)
    try:
        endpoint, backend_body, translated = _build_backend_request(
            resolved=resolved,
            body=body,
            client_format=client_format,
            backend_format=backend_format,
            base_url=base_url,
        )
    except ValueError as exc:
        return _error_response(
            client_format,
            str(exc),
            status_code=400,
            error_code="invalid_request",
        )

    t0 = time.monotonic()
    stream = bool(body.get("stream"))
    tools_raw = json.get_field(body, "tools", JsonArray, default=[])
    request_meta: JsonObject = {}
    request_meta["stream"] = stream
    request_meta["translated"] = translated
    request_meta["tool_count"] = len(tools_raw)
    if stream:
        return await _proxy_stream(
            runtime=runtime,
            endpoint=endpoint,
            headers=headers,
            body=backend_body,
            client_format=client_format,
            backend_format=backend_format,
            resolved=resolved,
            translated=translated,
            request_meta=request_meta,
            started_at=t0,
        )
    return await _proxy_no_stream(
        runtime=runtime,
        endpoint=endpoint,
        headers=headers,
        body=backend_body,
        client_format=client_format,
        backend_format=backend_format,
        resolved=resolved,
        translated=translated,
        request_meta=request_meta,
        started_at=t0,
    )


def _build_backend_request(
    *,
    resolved: ResolvedModel,
    body: JsonObject,
    client_format: str,
    backend_format: str,
    base_url: str,
) -> tuple[str, JsonObject, bool]:
    if client_format == backend_format:
        payload = dict(body)
        payload["model"] = resolved.backend_model
        if backend_format == "openai" and payload.get("stream"):
            payload["stream_options"] = {"include_usage": True}
        endpoint = _endpoint_for_format(base_url, backend_format)
        return endpoint, payload, False
    if client_format == "anthropic" and backend_format == "openai":
        payload = anthropic_request_to_openai(body)
        payload["model"] = resolved.backend_model
        if payload.get("stream"):
            payload["stream_options"] = {"include_usage": True}
        return _endpoint_for_format(base_url, backend_format), payload, True
    if client_format == "openai" and backend_format == "anthropic":
        payload = openai_request_to_anthropic(body)
        payload["model"] = resolved.backend_model
        return _endpoint_for_format(base_url, backend_format), payload, True
    raise ValueError(f"Unsupported proxy path: {client_format} -> {backend_format}")


async def _proxy_no_stream(
    *,
    runtime: LLMProxyRuntime,
    endpoint: str,
    headers: dict[str, str],
    body: JsonObject,
    client_format: str,
    backend_format: str,
    resolved: ResolvedModel,
    translated: bool,
    request_meta: JsonObject,
    started_at: float,
) -> Response:
    body = dict(body)
    body.pop("stream", None)
    async with HttpClient.create(timeout=120.0) as client:
        resp = await client.post(endpoint, headers=headers, json=body)
    duration_ms = int((time.monotonic() - started_at) * 1000)
    try:
        data: JsonObject = resp.json()  # type: ignore[assignment]
    except Exception:
        data: JsonObject = {"error": {"message": str(resp.text)}}
    if resp.status_code != 200:
        message = _extract_error_message(data)
        return _error_response(
            client_format,
            message,
            status_code=resp.status_code,
            error_code="backend_error",
        )
    response_data = data
    if client_format != backend_format:
        if client_format == "anthropic":
            response_data = openai_response_to_anthropic(data, model=resolved.request_model)
        else:
            response_data = anthropic_response_to_openai(data, model=resolved.request_model)
    usage = _usage_from_payload(response_data, client_format)
    runtime.logger.log_call(
        client_format=client_format,
        backend_format=backend_format,
        model=resolved.request_model,
        backend_model=resolved.backend_model,
        provider=resolved.provider_name,
        translated=translated,
        request_meta=request_meta,
        response_meta={"status_code": resp.status_code, "stream": False},
        usage=usage,
        duration_ms=duration_ms,
    )
    return JSONResponse(response_data)


async def _proxy_stream(
    *,
    runtime: LLMProxyRuntime,
    endpoint: str,
    headers: dict[str, str],
    body: JsonObject,
    client_format: str,
    backend_format: str,
    resolved: ResolvedModel,
    translated: bool,
    request_meta: JsonObject,
    started_at: float,
) -> StreamingResponse:
    body = dict(body)
    body["stream"] = True
    if backend_format == "openai":
        body["stream_options"] = {"include_usage": True}

    async def event_generator():
        captured_lines: list[str] = []
        async with HttpClient.create(timeout=120.0) as client:
            async with client.stream("POST", endpoint, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    try:
                        payload: JsonObject = json.loads(error_text)  # type: ignore[assignment]
                    except json.JSONDecodeError:
                        payload: JsonObject = {"error": {"message": error_text.decode("utf-8", errors="replace")}}
                    message = _extract_error_message(payload)
                    for chunk in _error_stream_chunks(client_format, message):
                        yield chunk
                    return
                if client_format == backend_format:
                    async for raw_line in resp.aiter_lines():
                        captured_lines.append(raw_line)
                        yield f"{raw_line}\n".encode()
                        if raw_line == "":
                            yield b"\n"
                elif client_format == "anthropic":
                    async for raw_line in resp.aiter_lines():
                        captured_lines.append(raw_line)
                    for event_type, event_data in openai_sse_to_anthropic_events(
                        iter(captured_lines),
                        model=resolved.request_model,
                    ):
                        yield f"event: {event_type}\ndata: {event_data}\n\n".encode()
                else:
                    async for raw_line in resp.aiter_lines():
                        captured_lines.append(raw_line)
                    for chunk in anthropic_sse_to_openai_chunks(
                        iter(captured_lines),
                        model=resolved.request_model,
                    ):
                        yield chunk.encode()
        duration_ms = int((time.monotonic() - started_at) * 1000)
        usage = _usage_from_stream(captured_lines, backend_format)
        runtime.logger.log_call(
            client_format=client_format,
            backend_format=backend_format,
            model=resolved.request_model,
            backend_model=resolved.backend_model,
            provider=resolved.provider_name,
            translated=translated,
            request_meta=request_meta,
            response_meta={"status_code": 200, "stream": True},
            usage=usage,
            duration_ms=duration_ms,
        )

    return StreamingResponse(
        body_iterator=event_generator(),
        media_type="text/event-stream",
        headers={
            "cache-control": "no-cache",
            "connection": "keep-alive",
        },
    )


def _backend_request_info(provider: LLMApiClient) -> tuple[str, str, dict[str, str]]:
    if isinstance(provider, CopilotApiClient):
        return provider.base_url, "openai", provider.auth.get_headers()
    if isinstance(provider, OpenAIApiClient):
        return provider.base_url, "openai", {
            "authorization": f"Bearer {provider.api_key}",
            "content-type": "application/json",
        }
    if isinstance(provider, AnthropicApiClient):
        return provider.base_url, "anthropic", {
            "authorization": f"Bearer {provider.api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    raise TypeError(f"Unsupported proxy provider instance: {type(provider)!r}")


def _endpoint_for_format(base_url: str, backend_format: str) -> str:
    return f"{base_url}/chat/completions" if backend_format == "openai" else f"{base_url}/v1/messages"


def _extract_error_message(payload: JsonObject) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        msg = error.get("message")
        if isinstance(msg, str) and msg:
            return msg
    if isinstance(error, str):
        return error
    return json.dumps(payload, ensure_ascii=False)


def _error_response(
    client_format: str,
    message: str,
    *,
    status_code: int,
    error_code: str,
) -> Response:
    if client_format == "anthropic":
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": error_code,
                    "message": message,
                },
            },
            status_code=status_code,
        )
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": error_code,
                "code": error_code,
            }
        },
        status_code=status_code,
    )


def _error_stream_chunks(client_format: str, message: str) -> list[bytes]:
    if client_format == "anthropic":
        payload = {
            "type": "error",
            "error": {
                "type": "backend_error",
                "message": message,
            },
        }
        return [f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()]
    payload = {
        "error": {
            "message": message,
            "type": "backend_error",
            "code": "backend_error",
        }
    }
    return [f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()]


def _usage_from_payload(payload: JsonObject, response_format: str) -> dict[str, int]:
    if response_format == "anthropic":
        usage_raw = json.get_field(payload, "usage", JsonObject, default={})
        result: dict[str, int] = {}
        for k, v in usage_raw.items():
            if isinstance(v, (int, float)):
                result[k] = int(v)
        return result
    usage_raw = json.get_field(payload, "usage", JsonObject, default={})
    result: dict[str, int] = {}
    result["input_tokens"] = json.get_field(usage_raw, "prompt_tokens", int, default=0)
    result["output_tokens"] = json.get_field(usage_raw, "completion_tokens", int, default=0)
    prompt_details = usage_raw.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        result["cache_read_input_tokens"] = json.get_field(prompt_details, "cached_tokens", int, default=0)
    return result


def _usage_from_stream(lines: list[str], backend_format: str) -> dict[str, int]:
    summary = summarize_openai_sse(lines) if backend_format == "openai" else summarize_anthropic_sse(lines)
    usage_raw = json.get_field(summary, "usage", JsonObject, default={})
    result: dict[str, int] = {}
    for k, v in usage_raw.items():
        if isinstance(v, (int, float)):
            result[k] = int(v)
    return result


def _default_log_dir(config: Config) -> Path:
    base_dir = Path(str(config.root.get_field("logging.log_dir", str, default=".mutagent/logs")))
    return base_dir / "proxy"


def _render_info_page() -> str:
    runtime = get_llm_proxy_runtime()
    models = runtime.list_models() if runtime is not None else []
    if models:
        rows = "".join(
            "<tr>"
            f"<td><code>{model['name']}</code></td>"
            f"<td><code>{model['model_id']}</code></td>"
            f"<td>{model['provider_name']}</td>"
            f"<td><code>{model['type']}</code></td>"
            "</tr>"
            for model in models
        )
        models_table = (
            "<table><thead><tr>"
            "<th>Name</th><th>Backend Model</th><th>Provider</th><th>Type</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
        )
    else:
        models_table = "<p><em>No models configured.</em></p>"

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>mutagent llm proxy</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #333; line-height: 1.6; }}
    code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }}
    pre {{ background: #f8f8f8; padding: 16px; border: 1px solid #ddd; border-radius: 8px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    .endpoint {{ margin: 16px 0; padding: 12px 16px; background: #f8f9fa; border-left: 4px solid #4a9eff; border-radius: 4px; }}
    .post {{ color: #2f855a; font-weight: 600; }}
    .get {{ color: #2b6cb0; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>mutagent LLM Proxy</h1>
  <p>Optional OpenAI / Anthropic compatible endpoints backed by <code>Config.providers</code>.</p>

  <div class="endpoint"><span class="get">GET</span> <code>/llm/v1/models</code> — list configured models</div>
  <div class="endpoint"><span class="post">POST</span> <code>/llm/v1/messages</code> — Anthropic Messages API</div>
  <div class="endpoint"><span class="post">POST</span> <code>/llm/v1/chat/completions</code> — OpenAI Chat Completions API</div>

  <p>Import <code>mutagent.llmproxy</code> to register the routes, then call
  <code>configure_llm_proxy(app.config)</code> before the server starts serving requests.</p>

  <h2>Configured models</h2>
  {models_table}

  <h2>Configuration example</h2>
  <pre><code>{{
  "default_model": "claude-sonnet-4",
  "providers": {{
    "anthropic": {{
      "type": "Anthropic",
      "auth_token": "$ANTHROPIC_API_KEY",
      "models": ["claude-sonnet-4-20250514"]
    }},
    "copilot": {{
      "type": "Copilot",
      "github_token": "$GITHUB_TOKEN",
      "models": {{
        "copilot-gpt": "gpt-4.1",
        "copilot-claude": "claude-sonnet-4"
      }}
    }}
  }}
}}</code></pre>
</body>
</html>"""
