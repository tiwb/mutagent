"""mutagent.core._llm_impl_copilot -- GitHub Copilot API provider."""

from __future__ import annotations

import logging
import time
from typing import AsyncGenerator, ClassVar
from uuid import uuid4

import httpx

from mutio.codec.json import JsonObject, get_field
from ._llm_impl import get_default_context_window
from ._llm_impl_openai import (
    messages_to_openai,
    send_no_stream,
    send_stream,
    tools_to_openai,
)
from .llm import LLMApiClient
from .messages import Message, StreamEvent, TextBlock, ToolSchema

logger = logging.getLogger(__name__)

GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"
VSCODE_VERSION = "1.99.0"
COPILOT_PLUGIN_VERSION = "copilot-chat/0.26.7"
COPILOT_USER_AGENT = "GitHubCopilotChat/0.26.7"
COPILOT_BASE_URLS = {
    "individual": "https://api.githubcopilot.com",
    "business": "https://api.business.githubcopilot.com",
    "enterprise": "https://api.enterprise.githubcopilot.com",
}

def _resolve_base_url(account_type: str, override: str = "") -> str:
    if override:
        return override
    return COPILOT_BASE_URLS.get(account_type, COPILOT_BASE_URLS["individual"])


class CopilotAuth:
    """GitHub Copilot 认证管理。"""

    github_token: str | None
    copilot_token: str | None
    expires_at: float

    def __init__(self, github_token: str | None = None) -> None:
        self.github_token = github_token
        self.copilot_token = None
        self.expires_at = 0.0

    def get_token(self) -> str:
        """获取有效的 Copilot JWT。"""
        self.ensure_github_token()
        if self._is_expired():
            self._refresh_copilot_token()
        assert self.copilot_token is not None
        return self.copilot_token

    def get_headers(self) -> dict[str, str]:
        """获取 Copilot API 请求头。"""
        return {
            "authorization": f"Bearer {self.get_token()}",
            "content-type": "application/json",
            "copilot-integration-id": "vscode-chat",
            "editor-version": f"vscode/{VSCODE_VERSION}",
            "editor-plugin-version": COPILOT_PLUGIN_VERSION,
            "openai-intent": "conversation-panel",
            "user-agent": COPILOT_USER_AGENT,
            "x-github-api-version": "2025-04-01",
            "x-request-id": str(uuid4()),
        }

    def ensure_github_token(self) -> str:
        """确保已有 GitHub access token。"""
        if self.github_token:
            return self.github_token
        token = self._device_flow()
        self.github_token = token
        return token

    def _is_expired(self) -> bool:
        return self.copilot_token is None or time.time() >= (self.expires_at - 300)

    def _device_flow(self) -> str:
        """执行 GitHub OAuth 设备流，返回 access token。"""
        logger.info("Starting GitHub OAuth device flow for Copilot")
        resp = httpx.post(
            "https://github.com/login/device/code",
            headers={"accept": "application/json"},
            data={"client_id": GITHUB_CLIENT_ID, "scope": "read:user"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        device_code = data["device_code"]
        user_code = data["user_code"]
        verification_uri = data.get("verification_uri_complete") or data["verification_uri"]
        interval = int(data.get("interval", 5))

        print(f"\nOpen: {verification_uri}")
        print(f"Enter code: {user_code}\n")

        while True:
            time.sleep(interval)
            poll = httpx.post(
                "https://github.com/login/oauth/access_token",
                headers={"accept": "application/json"},
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                timeout=30,
            )
            poll.raise_for_status()
            result = poll.json()

            error = result.get("error")
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval += 5
                continue
            if error == "expired_token":
                raise RuntimeError("GitHub device code expired. Please try again.")
            if error == "access_denied":
                raise RuntimeError("GitHub authorization denied by user.")
            if error:
                raise RuntimeError(f"GitHub OAuth error: {error}")
            return str(result["access_token"])

    def _refresh_copilot_token(self) -> None:
        """用 GitHub access token 换取 Copilot JWT。"""
        github_token = self.ensure_github_token()
        resp = httpx.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers={
                "authorization": f"token {github_token}",
                "accept": "application/json",
                "user-agent": COPILOT_USER_AGENT,
            },
            timeout=30,
        )
        if resp.status_code == 401:
            self.github_token = None
            self.copilot_token = None
            raise RuntimeError(
                "GitHub token expired. Update github_token in config or re-run device auth."
            )
        resp.raise_for_status()
        data = resp.json()
        self.copilot_token = str(data["token"])
        self.expires_at = float(data["expires_at"])
        logger.info("Copilot JWT refreshed (expires_at=%.0f)", self.expires_at)


class CopilotApiClient(LLMApiClient):
    """GitHub Copilot Chat Completions provider."""

    api_type: ClassVar[str] = "Copilot"

    base_url: str
    account_type: str
    auth: CopilotAuth

    def __init__(self, spec: JsonObject):
        model_id = get_field(spec, "model_id", str, default="")
        account_type = get_field(spec, "account_type", str, default="individual")
        base_url_override = get_field(spec, "base_url", str, default="")
        github_token_raw = get_field(spec, "github_token", str, default="")
        super().__init__(
            model_id=model_id,
            context_window=get_default_context_window(model_id),
            base_url=_resolve_base_url(account_type, base_url_override),
            account_type=account_type,
            auth=CopilotAuth(
                github_token=github_token_raw.strip() or None,
            ),
        )

    async def send(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        prompts: list[Message] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Send messages to Copilot API and yield streaming events."""
        openai_messages = messages_to_openai(messages)
        if prompts:
            for msg in reversed(prompts):
                for block in msg.blocks:
                    if isinstance(block, TextBlock) and block.text:
                        openai_messages.insert(
                            0, {"role": "system", "content": block.text}
                        )

        payload: JsonObject = {
            "model": self.model_id,
            "messages": openai_messages,
        }
        if tools:
            payload["tools"] = tools_to_openai(tools)

        headers = self.auth.get_headers()
        if stream:
            async for event in send_stream(self.base_url, payload, headers):
                yield event
            return

        async for event in send_no_stream(self.base_url, payload, headers):
            yield event
