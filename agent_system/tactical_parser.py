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
from .tactical_state import TacticalSituation


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
你的任务是把用户的一句自然语言战术指令映射为 TacticalHierarchySelfplay 的 12 个高层战术动作之一；如果用户输入不是当前系统支持的 1v1 空战战术指令，则必须拒识为 INVALID。
你不能输出舵面、油门、俯仰角、滚转角、航向角、速度指令等连续飞控动作，也不能创造第 13 类动作。

可选战术动作如下：
0: PURE_PURSUIT（纯追击）- 直接指向敌机当前位置，快速压向目标，适合用户要求“直接追击、咬住敌机、纯追”。
1: LEAD_PURSUIT（提前量追击）- 指向敌机短时预测位置，抢占敌机前方，适合用户要求“提前量、抢占前方、切入前方”。
2: LAG_PURSUIT（滞后追击）- 瞄准敌机后方位置，降低过冲风险，适合用户要求“滞后、尾随、跟在后面、别抢太前”。
3: DISENGAGE（脱离）- 背离敌机并拉开距离，适合用户要求“脱离、先撤、退出交战、重建态势”。
4: CLIMB_POSITION（爬升占位）- 通过爬升换取高度优势，适合用户要求“爬升、占高度、获得高度优势”。
5: DIVE_ACCELERATE（俯冲加速）- 俯冲并恢复速度/能量，适合用户要求“俯冲、下压、用高度换速度”。
6: LEVEL_ACCELERATE（平飞加速）- 保持平飞并提升速度，适合用户要求“平飞加速、保持高度加速、提高空速”。
7: LEVEL_DECELERATE（平飞减速）- 保持平飞并降低速度，适合用户要求“减速、避免冲过头、控制接近率”。
8: DEFENSIVE_TURN_LEFT（左防御转弯）- 向左转入防御机动，适合用户要求“左防御、左转防御、向左规避”。
9: DEFENSIVE_TURN_RIGHT（右防御转弯）- 向右转入防御机动，适合用户要求“右防御、右转防御、向右规避”。
10: HIGH_YOYO（高悠悠）- 偏向敌机并爬升减速，用高度和角度控制过冲，适合用户要求“高悠悠、高 yo、高 yo-yo”。
11: LOW_YOYO（低悠悠）- 偏向敌机并俯冲加速，先换速度再转入进攻，适合用户要求“低悠悠、低 yo、低 yo-yo”。

无效指令处理规则：
- 如果用户输入与 1v1 空战战术动作无关，例如闲聊、查询天气、解释代码、文件操作等，输出 tactical_action_id=-1，tactical_action_name="INVALID"。
- 如果用户输入超出系统能力边界，例如复杂任务链规划、多机协同、导弹发射/规避、切换模型、直接控制舵面/油门/速度数值等，输出 tactical_action_id=-1，tactical_action_name="INVALID"。
- 如果用户同时要求多个连续战术动作，例如“先提前量追击超过他再爬升占位”“先俯冲加速再高悠悠”，这是复杂任务链，不要拆解或选择其中一个动作，输出 tactical_action_id=-1，tactical_action_name="INVALID"。
- 如果语义不确定且无法明确对应 0-11 中任一动作，输出 tactical_action_id=-1，tactical_action_name="INVALID"。

只返回 JSON，不要返回 Markdown、解释性段落或代码块。JSON 字段固定为：
scene、agent_id、tactical_action_id、tactical_action_name、reason。
scene 必须是 "1v1"；agent_id 默认 "A0100"；tactical_action_id 必须是 0 到 11，或在无效指令时为 -1；
tactical_action_name 必须与编号完全对应，例如 4 对应 "CLIMB_POSITION"，-1 对应 "INVALID"。

有效输出示例：
用户指令：爬升占位，先拿高度优势。
{
  "scene": "1v1",
  "agent_id": "A0100",
  "tactical_action_id": 4,
  "tactical_action_name": "CLIMB_POSITION",
  "reason": "用户要求爬升并获取高度优势，因此选择爬升占位。"
}

无效输出示例：
用户指令：帮我写一段论文摘要。
{
  "scene": "1v1",
  "agent_id": "A0100",
  "tactical_action_id": -1,
  "tactical_action_name": "INVALID",
  "reason": "用户指令与当前 1v1 空战战术动作无关，无法映射到支持的 12 类高层战术动作，因此拒绝执行。"
}
"""


SEQUENCE_MARKERS = ("先", "再", "然后", "接着", "随后", "最后", "then", "after that", "next")


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

    raw_action_id = parsed.get("tactical_action_id")
    try:
        parsed_action_id = int(raw_action_id)
    except (TypeError, ValueError):
        parsed_action_id = None

    tactical_action_name = str(parsed.get("tactical_action_name", "")).strip()
    parsed_agent_id = str(parsed.get("agent_id", agent_id)).strip() or agent_id
    if parsed_action_id == -1:
        if tactical_action_name and tactical_action_name.upper() not in {"INVALID", "无效", "无效指令"}:
            return TacticalDecision.invalid(
                "无效指令必须使用 tactical_action_name=INVALID。",
                source="llm",
                raw_text=text,
                agent_id=parsed_agent_id,
            )
        return TacticalDecision(
            action_id=None,
            tactical_action_name="INVALID",
            reason=str(parsed.get("reason", "")).strip() or "LLM 判断该输入不是当前支持的 1v1 空战战术指令。",
            scene=scene,
            agent_id=parsed_agent_id,
            source="llm",
            raw_text=text,
            valid=False,
        )

    action_id = coerce_action_id(raw_action_id)
    if action_id is None:
        return TacticalDecision.invalid("tactical_action_id 不在 0-11 范围内。", source="llm", raw_text=text, agent_id=agent_id)

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
        agent_id=parsed_agent_id,
        source="llm",
        raw_text=text,
        valid=True,
    )


def _matched_action_ids(normalized_text: str) -> set[int]:
    matched_ids: set[int] = set()
    for action in TACTICAL_ACTIONS:
        if any(alias.lower() in normalized_text for alias in action.aliases):
            matched_ids.add(action.action_id)
    return matched_ids


def _looks_like_complex_task_chain(normalized_text: str) -> bool:
    if not any(marker in normalized_text for marker in SEQUENCE_MARKERS):
        return False
    return len(_matched_action_ids(normalized_text)) >= 2


def _state_hint_decision(user_input: str, *, agent_id: str, situation: TacticalSituation | None) -> TacticalDecision | None:
    normalized = user_input.lower()
    situation_reason = f" {situation.to_prompt_text()}" if situation is not None else ""
    if any(term in normalized for term in ("低空", "太低", "高度低", "高度太低", "拉起来", "别撞地")):
        return TacticalDecision(
            action_id=4,
            tactical_action_name=action_name(4),
            reason=f"状态相关指令指向低空改出/爬升需求，选择爬升占位。{situation_reason}",
            agent_id=agent_id,
            source="keyword_state",
            raw_text=user_input,
            valid=True,
        )
    if any(term in normalized for term in ("距离太近", "贴太近", "快冲过头", "快过冲", "接近率太大", "过冲风险")):
        if situation is None or situation.close_fast_risk:
            return TacticalDecision(
                action_id=7,
                tactical_action_name=action_name(7),
                reason=f"状态相关指令指向近距高速过冲风险，选择平飞减速。{situation_reason}",
                agent_id=agent_id,
                source="keyword_state",
                raw_text=user_input,
                valid=True,
            )
    if any(term in normalized for term in ("姿态不利", "被动", "先保命", "危险姿态", "角度不利", "态势不利")):
        if situation is None or situation.bad_posture_risk:
            return TacticalDecision(
                action_id=3,
                tactical_action_name=action_name(3),
                reason=f"状态相关指令指向不利姿态下先重整态势，选择脱离。{situation_reason}",
                agent_id=agent_id,
                source="keyword_state",
                raw_text=user_input,
                valid=True,
            )
    if any(term in normalized for term in ("速度不够", "能量不足", "空速不够", "速度太低", "补能量")):
        return TacticalDecision(
            action_id=6,
            tactical_action_name=action_name(6),
            reason=f"状态相关指令指向速度/能量不足，选择平飞加速。{situation_reason}",
            agent_id=agent_id,
            source="keyword_state",
            raw_text=user_input,
            valid=True,
        )
    return None


def keyword_parse_tactical_instruction(
    user_input: str,
    *,
    agent_id: str = "A0100",
    situation: TacticalSituation | None = None,
    allow_complex: bool = True,
) -> TacticalDecision:
    text = user_input.strip()
    if not text:
        return TacticalDecision.invalid("输入为空。", source="keyword", raw_text=user_input, agent_id=agent_id)

    normalized = text.lower()
    if allow_complex and _looks_like_complex_task_chain(normalized):
        return TacticalDecision.invalid("检测到多步复杂任务链，当前版本只支持单个高层战术动作。", source="keyword", raw_text=user_input, agent_id=agent_id)

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
        matched_ids = {item[1].action_id for item in scored}
        if (
            situation is not None
            and situation.close_fast_risk
            and matched_ids
            and matched_ids.issubset({2, 7})
            and any(term in normalized for term in ("别冲过头", "避免冲过头", "冲过头", "过冲"))
        ):
            return TacticalDecision(
                action_id=7,
                tactical_action_name=action_name(7),
                reason=f"当前存在近距高速过冲风险，过冲相关指令优先解释为平飞减速。 {situation.to_prompt_text()}",
                agent_id=agent_id,
                source="keyword_state",
                raw_text=user_input,
                valid=True,
            )
        # 优先使用最长关键词，避免“加速俯冲”被普通“加速”抢先匹配。
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    state_hint = _state_hint_decision(user_input, agent_id=agent_id, situation=situation)
    if state_hint is not None:
        return state_hint

    return TacticalDecision.invalid("未识别到支持的战术意图。", source="keyword", raw_text=user_input, agent_id=agent_id)


def parse_tactical_instruction(
    user_input: str,
    *,
    client: SiliconFlowClient | None = None,
    agent_id: str = "A0100",
    situation: TacticalSituation | None = None,
) -> TacticalDecision:
    if not user_input.strip():
        return TacticalDecision.invalid("输入为空。", source="manual", raw_text=user_input, agent_id=agent_id)

    if client is not None:
        situation_text = situation.to_prompt_text() if situation is not None else "当前态势摘要：未提供。"
        try:
            content = client.chat(
                [
                    LLMMessage("system", TACTICAL_SYSTEM_PROMPT),
                    LLMMessage("user", f"{situation_text}\n\n用户指令：{user_input}"),
                ],
                temperature=0.0,
                max_tokens=256,
                enable_thinking=False,
            )
            decision = parse_tactical_json(content, agent_id=agent_id)
            if decision.valid:
                return decision
            if decision.tactical_action_name == "INVALID":
                return decision
            fallback = keyword_parse_tactical_instruction(user_input, agent_id=agent_id, situation=situation)
            if fallback.valid:
                return TacticalDecision(
                    action_id=fallback.action_id,
                    tactical_action_name=fallback.tactical_action_name,
                    reason=f"{fallback.reason} LLM 输出 JSON 无效，已使用关键词兜底：{decision.reason}",
                    agent_id=agent_id,
                    source=fallback.source,
                    raw_text=user_input,
                    valid=True,
                )
            return decision
        except Exception as exc:
            # LLM 不可用时继续走关键词，保证演示脚本离线可跑。
            fallback = keyword_parse_tactical_instruction(user_input, agent_id=agent_id, situation=situation)
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

    return keyword_parse_tactical_instruction(user_input, agent_id=agent_id, situation=situation)


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
