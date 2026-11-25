"""Provider-independent LLM adapter (OpenAI-compatible HTTP JSON).

No API key is hardcoded. When no provider is configured, the deterministic extractor
remains the production path. LLM output is always schema-validated and grounded against
the source message; it can never decide a procurement action.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from src.errors import LLMProviderError


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str | None
    api_key: str | None
    model: str | None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    temperature: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.provider and self.provider.lower() not in {"none", "disabled", "deterministic"})

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            provider=(os.getenv("LLM_PROVIDER") or "none").strip(),
            base_url=(os.getenv("LLM_BASE_URL") or "").strip() or None,
            api_key=(os.getenv("LLM_API_KEY") or "").strip() or None,
            model=(os.getenv("LLM_MODEL") or "").strip() or None,
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        )


class LLMProvider(Protocol):
    name: str

    def complete_json(self, *, system_prompt: str, user_prompt: str, schema: dict) -> dict: ...


class NullProvider:
    name = "none"

    def complete_json(self, *, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        raise LLMProviderError("no LLM provider is configured; using deterministic extraction")


class OpenAICompatibleProvider:
    """Minimal HTTP client for /chat/completions compatible endpoints."""

    name = "openai-compatible"

    def __init__(self, config: LLMConfig) -> None:
        if not config.base_url or not config.api_key or not config.model:
            raise LLMProviderError(
                "LLM_BASE_URL, LLM_API_KEY and LLM_MODEL are required for a live LLM provider"
            )
        self.config = config

    def complete_json(self, *, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                content = parsed["choices"][0]["message"]["content"]
                if isinstance(content, dict):
                    return content
                return json.loads(content)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.config.max_retries:
                    time.sleep(min(2**attempt, 5))
                    continue
                raise LLMProviderError(f"LLM HTTP error {exc.code}: {exc.reason}") from exc
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(min(2**attempt, 5))
                    continue
                raise LLMProviderError(f"LLM request failed: {exc}") from exc
        raise LLMProviderError(f"LLM request failed after retries: {last_error}")


def get_provider(config: LLMConfig | None = None) -> LLMProvider:
    config = config or LLMConfig.from_env()
    if not config.configured:
        return NullProvider()
    return OpenAICompatibleProvider(config)
