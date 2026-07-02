"""Unified LLM API client with retry logic, provider presets, and jailbreak support."""

import asyncio
from typing import Any

import httpx
from openai import AsyncOpenAI

from .rate_limiter import RateLimiter

# Provider presets (base_url)
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "grok": {
        "base_url": "https://ld.uzumakinoharu.top/v1",
    },
}


class LLMClient:
    """Async LLM client wrapping OpenAI-compatible APIs.

    Uses OpenAI SDK when possible, falls back to raw httpx for
    providers that block the SDK's headers (e.g. some Grok proxies).

    Jailbreak support: when jailbreak config is provided, injects a fake
    assistant reply + system prompt override to bypass safety filters.
    """

    def __init__(self, config: dict, rate_limiter: RateLimiter):
        llm_cfg = config["llm"]
        provider = llm_cfg.get("provider", "custom")
        preset = PROVIDER_PRESETS.get(provider, {})

        self._base_url = llm_cfg.get("base_url") or preset.get("base_url", "")
        self._api_key = llm_cfg["api_key"]
        self._model = llm_cfg["model"]
        self._temperature = llm_cfg.get("temperature", 0.8)
        self._max_tokens = llm_cfg.get("max_tokens", 2048)
        self._extra = llm_cfg.get("extra", {})
        self._rate_limiter = rate_limiter
        self._provider = provider
        self._use_httpx = provider in ("grok",)  # providers that block OpenAI SDK

        # Jailbreak config
        jb = config.get("jailbreak", {})
        self._jailbreak_enabled: bool = jb.get("enabled", False)
        self._jb_system_prompt: str = jb.get("system_prompt", "").strip()
        self._jb_fake_reply: str = jb.get("fake_assistant_reply", "")

        self._client: AsyncOpenAI | None = None
        self._httpx_client: httpx.AsyncClient | None = None

        # Stats
        self.total_calls: int = 0
        self.total_tokens: int = 0

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-init the OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key or "sk-placeholder",
                base_url=self._base_url,
            )
        return self._client

    def _get_httpx(self) -> httpx.AsyncClient:
        """Lazy-init a plain httpx client (for providers that block SDK)."""
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=60.0)
        return self._httpx_client

    def _build_messages(
        self,
        user_content: str,
        system_prompt: str = "",
    ) -> list[dict[str, str]]:
        """Build the messages array, optionally applying jailbreak injection."""
        messages: list[dict[str, str]] = []

        if self._jailbreak_enabled and self._jb_system_prompt:
            messages.append({"role": "system", "content": self._jb_system_prompt})
            if self._jb_fake_reply:
                messages.append({"role": "assistant", "content": self._jb_fake_reply})
        elif system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_content})
        return messages

    async def _generate_via_sdk(self, messages: list[dict], **overrides) -> str | None:
        """Generate using OpenAI SDK."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": overrides.get("temperature", self._temperature),
            "max_tokens": overrides.get("max_tokens", self._max_tokens),
        }
        for k, v in self._extra.items():
            kwargs.setdefault(k, v)

        client = self._get_client()
        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if response.usage:
            self.total_tokens += response.usage.total_tokens
        return content

    async def _generate_via_httpx(self, messages: list[dict], **overrides) -> str | None:
        """Generate using raw httpx (bypasses SDK blocks)."""
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": overrides.get("temperature", self._temperature),
            "max_tokens": overrides.get("max_tokens", self._max_tokens),
            "stream": False,
        }
        for k, v in self._extra.items():
            body.setdefault(k, v)

        http = self._get_httpx()
        resp = await http.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self.total_tokens += usage.get("total_tokens", 0)
        return content

    async def generate(
        self,
        user_content: str,
        system_prompt: str = "",
        **overrides: Any,
    ) -> str | None:
        """Generate a single completion. Returns None on failure after retries."""
        messages = self._build_messages(user_content, system_prompt)

        retry_delays = self._rate_limiter.retry_delays()

        for attempt in range(len(retry_delays) + 1):
            try:
                await self._rate_limiter.acquire()
                try:
                    if self._use_httpx:
                        content = await self._generate_via_httpx(messages, **overrides)
                    else:
                        content = await self._generate_via_sdk(messages, **overrides)
                finally:
                    self._rate_limiter.release()

                self.total_calls += 1
                return content

            except Exception as e:
                if attempt < len(retry_delays):
                    delay = retry_delays[attempt]
                    print(f"  [RETRY {attempt + 1}] {e} — waiting {delay:.0f}s")
                    await asyncio.sleep(delay)
                else:
                    print(f"  [FAIL] All retries exhausted: {e}")
                    return None

    @property
    def model_name(self) -> str:
        return self._model
