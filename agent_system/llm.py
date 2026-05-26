from __future__ import annotations

import json
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

    def __init__(self, settings: AgentSettings | None = None, timeout: int = 120) -> None:
        self.settings = settings or AgentSettings.load()
        self.timeout = timeout

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

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"硅基流动请求失败: HTTP {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"硅基流动请求失败: {exc}") from exc

        parsed = json.loads(body)
        try:
            return str(parsed["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"硅基流动响应格式异常: {body}") from exc
