"""Tests for the GitHub Copilot API provider."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from mutagent.core._llm_impl import _resolve_provider
from mutagent.core._llm_impl_copilot import (
    COPILOT_BASE_URLS,
    CopilotApiClient,
    CopilotAuth,
)
from mutagent.core.llm import LLMApiClient
from mutagent.core.messages import Message, Response, StreamEvent, TextBlock, Usage


async def _collect_events(stream) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    async for event in stream:
        events.append(event)
    return events


class TestCopilotProviderAliases:

    @pytest.mark.parametrize(
        "provider_type",
        [
            "Copilot",
            "CopilotProvider",
            "mutbot.copilot.provider.CopilotProvider",
        ],
    )
    def test_from_spec_accepts_compat_aliases(self, provider_type: str):
        provider = LLMApiClient.from_spec(
            {
                "type": provider_type,
                "model_id": "claude-sonnet-4",
                "github_token": "ghu_test",
            }
        )
        assert isinstance(provider, CopilotApiClient)
        assert provider.base_url == COPILOT_BASE_URLS["individual"]

    def test_resolve_provider_accepts_short_and_legacy_names(self):
        assert _resolve_provider("Copilot") is CopilotApiClient
        assert _resolve_provider("CopilotProvider") is CopilotApiClient
        assert (
            _resolve_provider("mutbot.copilot.provider.CopilotProvider")
            is CopilotApiClient
        )


class TestCopilotAuth:

    def test_missing_github_token_runs_device_flow_once(self):
        auth = CopilotAuth()

        def fake_refresh(self: CopilotAuth) -> None:
            self.copilot_token = "jwt_test"
            self.expires_at = time.time() + 3600

        with (
            patch.object(CopilotAuth, "_device_flow", return_value="ghu_new"),
            patch.object(CopilotAuth, "_refresh_copilot_token", new=fake_refresh),
        ):
            assert auth.get_token() == "jwt_test"

        assert auth.github_token == "ghu_new"


class TestCopilotApiClient:

    @pytest.mark.asyncio
    async def test_send_uses_copilot_base_url_and_openai_helpers(self):
        provider = CopilotApiClient(
            {
                "type": "Copilot",
                "model_id": "gpt-4.1",
                "github_token": "ghu_test",
                "account_type": "enterprise",
            }
        )
        response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="OK")]),
            stop_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1),
        )
        expected_events = [
            StreamEvent(type="text_delta", text="OK"),
            StreamEvent(type="response_done", response=response),
        ]
        captured: dict[str, object] = {}

        async def fake_send_no_stream(base_url: str, payload: dict, headers: dict):
            captured["base_url"] = base_url
            captured["payload"] = payload
            captured["headers"] = headers
            for event in expected_events:
                yield event

        with (
            patch.object(provider.auth, "get_headers", return_value={"authorization": "Bearer jwt"}),
            patch(
                "mutagent.core._llm_impl_copilot._send_no_stream",
                side_effect=fake_send_no_stream,
            ),
        ):
            events = await _collect_events(
                provider.send(
                    [Message(role="user", blocks=[TextBlock(text="Hello")])],
                    [],
                    prompts=[Message(role="system", blocks=[TextBlock(text="Be concise")])],
                    stream=False,
                )
            )

        assert events == expected_events
        assert captured["base_url"] == COPILOT_BASE_URLS["enterprise"]
        assert captured["headers"] == {"authorization": "Bearer jwt"}
        assert captured["payload"] == {
            "model": "gpt-4.1",
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Hello"},
            ],
        }
