from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .settings import AgentSettings


@dataclass
class LLMMessage:
    role: str
    content: str


class SiliconFlowClient:
    """通过 OpenAI-compatible Chat Completions 接口调用硅基流动。"""

    def __init__(
        self,
        settings: AgentSettings | None = None,
        timeout: int = 120,
        *,
        min_request_interval: float = 0.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 5.0,
        max_retry_wait_seconds: float = 60.0,
    ) -> None:
        self.settings = settings or AgentSettings.load()
        self.timeout = timeout
        self.min_request_interval = max(0.0, float(min_request_interval))
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.max_retry_wait_seconds = max(0.0, float(max_retry_wait_seconds))
        self._effective_request_interval = self.min_request_interval
        self._last_request_at: float | None = None
        self._logical_request_count = 0
        self._request_attempt_count = 0
        self._successful_response_count = 0
        self._rate_limit_response_count = 0
        self._retry_count = 0
        self._throttle_wait_seconds = 0.0
        self._retry_wait_seconds = 0.0

    def _wait_for_request_slot(self) -> None:
        if self._last_request_at is None or self._effective_request_interval <= 0.0:
            return
        elapsed = time.monotonic() - self._last_request_at
        wait_seconds = self._effective_request_interval - elapsed
        if wait_seconds > 0.0:
            time.sleep(wait_seconds)
            self._throttle_wait_seconds += wait_seconds

    def _retry_delay(self, exc: urllib.error.HTTPError, retry_index: int) -> float:
        retry_after = None
        if exc.headers is not None:
            raw_retry_after = exc.headers.get("Retry-After")
            try:
                retry_after = float(raw_retry_after) if raw_retry_after is not None else None
            except (TypeError, ValueError):
                retry_after = None
        exponential_delay = self.retry_backoff_seconds * (2 ** retry_index)
        delay = max(retry_after or 0.0, exponential_delay)
        return min(delay, self.max_retry_wait_seconds)

    def metrics(self) -> dict[str, int | float]:
        """返回请求节流与重试统计，供实验日志判断是否仍触发限流。"""
        return {
            "logical_request_count": self._logical_request_count,
            "request_attempt_count": self._request_attempt_count,
            "successful_response_count": self._successful_response_count,
            "rate_limit_response_count": self._rate_limit_response_count,
            "retry_count": self._retry_count,
            "throttle_wait_seconds": round(self._throttle_wait_seconds, 3),
            "retry_wait_seconds": round(self._retry_wait_seconds, 3),
            "min_request_interval": self.min_request_interval,
            "effective_request_interval": round(self._effective_request_interval, 3),
            "max_retries": self.max_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
        }

    def chat(
        self,
        messages: Iterable[LLMMessage | dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        enable_thinking: bool | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> str:
        if not self.settings.has_llm_credentials:
            raise RuntimeError("缺少 SILICONFLOW_API_KEY，无法调用硅基流动模型。")

        self._logical_request_count += 1

        payload_messages = [
            message if isinstance(message, dict) else {"role": message.role, "content": message.content}
            for message in messages
        ]
        payload = {
            "model": self.settings.agent_chat_model,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if enable_thinking is not None:
            # 硅基流动的 Qwen 推理模型会把内容放进 reasoning_content；
            # 结构化 JSON 解析场景需要关闭 thinking，避免 message.content 为空。
            payload["enable_thinking"] = enable_thinking
        if extra_body:
            payload.update(extra_body)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = f"{self.settings.siliconflow_base_url.rstrip('/')}/chat/completions"
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.settings.siliconflow_api_key}",
                "Content-Type": "application/json",
            },
        )

        retry_index = 0
        while True:
            self._wait_for_request_slot()
            self._last_request_at = time.monotonic()
            self._request_attempt_count += 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                self._successful_response_count += 1
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429:
                    self._rate_limit_response_count += 1
                    # 发生 TPM 限流后提高后续请求间隔，避免整组实验持续撞限流窗口。
                    increased_interval = max(1.0, self._effective_request_interval + 1.0)
                    self._effective_request_interval = min(increased_interval, self.max_retry_wait_seconds)
                if exc.code != 429 or retry_index >= self.max_retries:
                    raise RuntimeError(f"硅基流动请求失败: HTTP {exc.code} {detail}") from exc

                wait_seconds = self._retry_delay(exc, retry_index)
                self._retry_count += 1
                retry_index += 1
                if wait_seconds > 0.0:
                    time.sleep(wait_seconds)
                    self._retry_wait_seconds += wait_seconds
            except urllib.error.URLError as exc:
                raise RuntimeError(f"硅基流动请求失败: {exc}") from exc

        parsed = json.loads(body)
        try:
            return str(parsed["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"硅基流动响应格式异常: {body}") from exc
