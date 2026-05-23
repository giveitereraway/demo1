from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .llm import LLMMessage, SiliconFlowClient
from .tactical_actions import (
    ACTION_BY_ID,
    TACTICAL_ACTIONS,
    action_name,
    action_name_matches,
    coerce_action_id,
)


@dataclass(frozen=True)
class TacticalDecision:
    action_id: int | None
    tactical_action_name: str = ""
    reason: str = ""
    scene: str = "1v1"
    agent_id: str = "A0100"
    source: str = "invalid"
    raw_text: str = ""
    valid: bool = False

    @classmethod
    def invalid(cls, reason: str, *, source: str = "invalid", raw_text: str = "", agent_id: str = "A0100") -> "TacticalDecision":
        return cls(None, reason=reason, source=source, raw_text=raw_text, agent_id=agent_id, valid=False)


TACTICAL_SYSTEM_PROMPT = """你是 1v1 空战战术调度 Agent。
你只能把用户一句中文战术指令映射为 TacticalHierarchySelfplay 的 12 个战术动作之一。
不要输出舵面、油门或连续飞控动作。
只返回 JSON，不要返回 Markdown。JSON 字段固定为 scene、agent_id、tactical_action_id、tactical_action_name、reason。
scene 必须是 1v1，agent_id 默认 A0100，tactical_action_id 必须是 0 到 11。
"""


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        cleaned = match.group(0)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM 输出不是 JSON 对象。")
    return parsed


def parse_tactical_json(text: str, *, agent_id: str = "A0100") -> TacticalDecision:
    try:
        parsed = _json_object(text)
    except Exception as exc:
        return TacticalDecision.invalid(f"无法解析 LLM JSON: {exc}", source="llm", raw_text=text, agent_id=agent_id)

    scene = str(parsed.get("scene", "")).strip()
    if scene != "1v1":
        return TacticalDecision.invalid(f"scene 必须是 1v1，当前为 {scene!r}。", source="llm", raw_text=text, agent_id=agent_id)

    action_id = coerce_action_id(parsed.get("tactical_action_id"))
    if action_id is None:
        return TacticalDecision.invalid("tactical_action_id 不在 0-11 范围内。", source="llm", raw_text=text, agent_id=agent_id)

    tactical_action_name = str(parsed.get("tactical_action_name", "")).strip()
    if tactical_action_name and not action_name_matches(action_id, tactical_action_name):
        return TacticalDecision.invalid(
            f"动作名 {tactical_action_name!r} 与编号 {action_id} 不一致。",
            source="llm",
            raw_text=text,
            agent_id=agent_id,
        )

    return TacticalDecision(
        action_id=action_id,
        tactical_action_name=action_name(action_id),
        reason=str(parsed.get("reason", "")).strip(),
        scene=scene,
        agent_id=str(parsed.get("agent_id", agent_id)).strip() or agent_id,
        source="llm",
        raw_text=text,
        valid=True,
    )


def keyword_parse_tactical_instruction(user_input: str, *, agent_id: str = "A0100") -> TacticalDecision:
    text = user_input.strip()
    if not text:
        return TacticalDecision.invalid("输入为空。", source="keyword", raw_text=user_input, agent_id=agent_id)

    normalized = text.lower()
    scored: list[tuple[int, TacticalDecision]] = []
    for action in TACTICAL_ACTIONS:
        for alias in action.aliases:
            if alias.lower() in normalized:
                scored.append(
                    (
                        len(alias),
                        TacticalDecision(
                            action_id=action.action_id,
                            tactical_action_name=action.code,
                            reason=f"关键词匹配到“{alias}”。",
                            agent_id=agent_id,
                            source="keyword",
                            raw_text=user_input,
                            valid=True,
                        ),
                    )
                )
    if scored:
        # 优先使用最长关键词，避免“加速俯冲”被普通“加速”抢先匹配。
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    return TacticalDecision.invalid("未识别到支持的战术意图。", source="keyword", raw_text=user_input, agent_id=agent_id)


def parse_tactical_instruction(
    user_input: str,
    *,
    client: SiliconFlowClient | None = None,
    agent_id: str = "A0100",
) -> TacticalDecision:
    if not user_input.strip():
        return TacticalDecision.invalid("输入为空。", source="manual", raw_text=user_input, agent_id=agent_id)

    if client is not None:
        action_table = "\n".join(
            f"{item.action_id}: {item.code}（{item.chinese_name}）- {item.description}"
            for item in TACTICAL_ACTIONS
        )
        try:
            content = client.chat(
                [
                    LLMMessage("system", TACTICAL_SYSTEM_PROMPT),
                    LLMMessage("user", f"可选战术动作如下：\n{action_table}\n\n用户指令：{user_input}"),
                ],
                temperature=0.0,
                max_tokens=512,
            )
            decision = parse_tactical_json(content, agent_id=agent_id)
            if decision.valid:
                return decision
        except Exception as exc:
            # LLM 不可用时继续走关键词，保证演示脚本离线可跑。
            fallback = keyword_parse_tactical_instruction(user_input, agent_id=agent_id)
            if fallback.valid:
                return TacticalDecision(
                    action_id=fallback.action_id,
                    tactical_action_name=fallback.tactical_action_name,
                    reason=f"{fallback.reason} LLM 解析失败，已使用关键词兜底: {exc}",
                    agent_id=agent_id,
                    source=fallback.source,
                    raw_text=user_input,
                    valid=True,
                )
            return TacticalDecision.invalid(f"LLM 解析失败且关键词未识别: {exc}", source="llm", raw_text=user_input, agent_id=agent_id)

    return keyword_parse_tactical_instruction(user_input, agent_id=agent_id)


def decision_to_log(decision: TacticalDecision) -> dict[str, object]:
    action = ACTION_BY_ID.get(decision.action_id) if decision.action_id is not None else None
    return {
        "valid": decision.valid,
        "source": decision.source,
        "scene": decision.scene,
        "agent_id": decision.agent_id,
        "tactical_action_id": decision.action_id,
        "tactical_action_name": decision.tactical_action_name,
        "tactical_action_cn": action.chinese_name if action else "",
        "reason": decision.reason,
        "raw_text": decision.raw_text,
    }
