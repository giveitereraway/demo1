from __future__ import annotations

import re

from .llm import LLMMessage, SiliconFlowClient
from .tactical_actions import TACTICAL_ACTIONS, action_name, action_name_matches, coerce_action_id
from .tactical_parser import (
    SEQUENCE_MARKERS,
    _json_object,
    _looks_like_complex_task_chain,
    keyword_parse_tactical_instruction,
    parse_tactical_instruction,
)
from .tactical_plan import TacticalCommand, TacticalPlan, TacticalPlanStep
from .tactical_state import TacticalSituation


TACTICAL_PLAN_SYSTEM_PROMPT = """你是 1v1 空战战术计划 Agent。
你的任务是把用户的复杂自然语言战术指令拆解成 2 到 4 个有限高层战术步骤。每个步骤必须是 TacticalHierarchySelfplay 已支持的 12 类 tactical action id 之一。

硬性边界：
- 只能输出 0-11 的高层战术动作，不能输出舵面、油门、俯仰角、滚转角、航向角或连续速度数值。
- 不处理多机协同、导弹发射/规避、模型切换、RAG 查询、文件操作或论文写作等无关任务。
- 如果用户指令不包含至少两个清晰战术动作，或超出上述边界，输出 valid=false 且 steps=[]。
- 当前态势只能用于选择合理的切换条件和解释，不允许创造新动作。

可选 until 条件：
- fixed_steps：按 max_steps 执行固定步数。
- range_close：敌我距离进入近距。
- range_opened：敌我距离已拉开。
- altitude_advantage：己方获得高度优势。
- overshoot_risk_reduced：近距高速过冲风险缓解。
- bad_posture_recovered：不利姿态风险恢复。

只返回 JSON，不要返回 Markdown。JSON 字段固定为：
valid、reason、steps。
steps 中每个对象字段固定为：
tactical_action_id、tactical_action_name、min_steps、max_steps、until、reason。

有效输出示例：
用户指令：先提前量追击超过他再爬升占位
{
  "valid": true,
  "reason": "用户给出两个连续战术动作，先抢占前方，再爬升获取高度优势。",
  "steps": [
    {
      "tactical_action_id": 1,
      "tactical_action_name": "LEAD_PURSUIT",
      "min_steps": 3,
      "max_steps": 10,
      "until": "range_close",
      "reason": "先用提前量追击压向敌机前方。"
    },
    {
      "tactical_action_id": 4,
      "tactical_action_name": "CLIMB_POSITION",
      "min_steps": 3,
      "max_steps": 10,
      "until": "altitude_advantage",
      "reason": "接着爬升占位以获得高度优势。"
    }
  ]
}
"""


UNSUPPORTED_COMPLEX_TERMS = (
    "导弹",
    "发射",
    "规避导弹",
    "多机",
    "编队",
    "协同",
    "切换模型",
    "油门",
    "舵面",
    "俯仰角",
    "滚转角",
    "航向角",
    "速度数值",
    "论文",
    "代码",
    "文件",
)


def _default_until_for_action(action_id: int) -> str:
    if action_id == 1:
        return "range_close"
    if action_id == 3:
        return "range_opened"
    if action_id == 4:
        return "altitude_advantage"
    if action_id == 7:
        return "overshoot_risk_reduced"
    return "fixed_steps"


def _split_complex_instruction(text: str) -> list[str]:
    marker_pattern = "|".join(re.escape(marker) for marker in sorted(SEQUENCE_MARKERS, key=len, reverse=True))
    parts = re.split(marker_pattern, text, flags=re.I)
    return [part.strip(" ，,。；;") for part in parts if part.strip(" ，,。；;")]


def _ordered_action_chunks(text: str, *, situation: TacticalSituation | None, agent_id: str) -> list[tuple[int, str, str]]:
    chunks = []
    for part in _split_complex_instruction(text):
        decision = keyword_parse_tactical_instruction(part, agent_id=agent_id, situation=situation, allow_complex=False)
        if decision.valid and decision.action_id is not None:
            chunks.append((decision.action_id, part, decision.reason))

    if len(chunks) >= 2:
        return chunks

    matches: list[tuple[int, int, str, str]] = []
    normalized = text.lower()
    for action in TACTICAL_ACTIONS:
        for alias in action.aliases:
            position = normalized.find(alias.lower())
            if position >= 0:
                matches.append((position, len(alias), action.code, alias))
    matches.sort(key=lambda item: (item[0], -item[1]))

    ordered: list[tuple[int, str, str]] = []
    seen_positions: set[int] = set()
    for position, _, code, alias in matches:
        action_id = next(action.action_id for action in TACTICAL_ACTIONS if action.code == code)
        if position in seen_positions:
            continue
        seen_positions.add(position)
        if ordered and ordered[-1][0] == action_id:
            continue
        ordered.append((action_id, alias, f"关键词匹配到“{alias}”。"))
    return ordered


def keyword_parse_tactical_plan(
    user_input: str,
    *,
    agent_id: str = "A0100",
    situation: TacticalSituation | None = None,
    max_plan_actions: int = 4,
    default_min_steps: int = 3,
    default_max_steps: int = 10,
) -> TacticalCommand:
    text = user_input.strip()
    if not text:
        return TacticalCommand.invalid("输入为空。", raw_text=user_input)

    normalized = text.lower()
    if any(term.lower() in normalized for term in UNSUPPORTED_COMPLEX_TERMS):
        return TacticalCommand.invalid("复杂指令包含当前系统不支持的任务边界。", raw_text=user_input)

    if not _looks_like_complex_task_chain(normalized):
        return TacticalCommand.from_decision(
            keyword_parse_tactical_instruction(text, agent_id=agent_id, situation=situation, allow_complex=False)
        )

    chunks = _ordered_action_chunks(text, situation=situation, agent_id=agent_id)
    if len(chunks) < 2:
        return TacticalCommand.invalid("未能从复杂指令中提取至少两个有效战术动作。", raw_text=user_input)
    if len(chunks) > max_plan_actions:
        return TacticalCommand.invalid(f"复杂计划包含 {len(chunks)} 个动作，超过上限 {max_plan_actions}。", raw_text=user_input)

    steps = [
        TacticalPlanStep.build(
            action_id,
            min_steps=default_min_steps,
            max_steps=default_max_steps,
            until=_default_until_for_action(action_id),
            reason=reason,
        )
        for action_id, _, reason in chunks
    ]
    plan = TacticalPlan.build(
        steps,
        reason="关键词解析得到有限多步战术计划。",
        source="keyword_plan",
        raw_text=user_input,
        agent_id=agent_id,
    )
    return TacticalCommand.from_plan(plan)


def parse_tactical_plan_json(
    text: str,
    *,
    raw_instruction: str,
    agent_id: str = "A0100",
    source: str = "llm_plan",
    max_plan_actions: int = 4,
    default_min_steps: int = 3,
    default_max_steps: int = 10,
) -> TacticalCommand:
    try:
        parsed = _json_object(text)
    except Exception as exc:
        return TacticalCommand.invalid(f"无法解析 LLM 计划 JSON: {exc}", raw_text=raw_instruction)

    if parsed.get("valid") is False:
        return TacticalCommand.invalid(str(parsed.get("reason", "")).strip() or "LLM 判断该复杂指令无效。", raw_text=raw_instruction)

    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list):
        return TacticalCommand.invalid("LLM 计划 JSON 缺少 steps 数组。", raw_text=raw_instruction)
    if not 2 <= len(raw_steps) <= max_plan_actions:
        return TacticalCommand.invalid(f"LLM 计划步骤数必须在 2 到 {max_plan_actions} 之间。", raw_text=raw_instruction)

    steps: list[TacticalPlanStep] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            return TacticalCommand.invalid(f"第 {index} 个计划步骤不是 JSON 对象。", raw_text=raw_instruction)
        action_id = coerce_action_id(raw_step.get("tactical_action_id"))
        if action_id is None:
            return TacticalCommand.invalid(f"第 {index} 个计划步骤动作编号非法。", raw_text=raw_instruction)
        action_name_value = str(raw_step.get("tactical_action_name", "")).strip()
        if action_name_value and not action_name_matches(action_id, action_name_value):
            return TacticalCommand.invalid(f"第 {index} 个计划步骤动作名与编号不一致。", raw_text=raw_instruction)
        try:
            steps.append(
                TacticalPlanStep.build(
                    action_id,
                    tactical_action_name=action_name(action_id),
                    min_steps=int(raw_step.get("min_steps", default_min_steps)),
                    max_steps=int(raw_step.get("max_steps", default_max_steps)),
                    until=str(raw_step.get("until", _default_until_for_action(action_id))),
                    reason=str(raw_step.get("reason", "")).strip(),
                )
            )
        except ValueError as exc:
            return TacticalCommand.invalid(str(exc), raw_text=raw_instruction)

    try:
        plan = TacticalPlan.build(
            steps,
            reason=str(parsed.get("reason", "")).strip() or "LLM 解析得到有限多步战术计划。",
            source=source,
            raw_text=raw_instruction,
            agent_id=agent_id,
        )
    except ValueError as exc:
        return TacticalCommand.invalid(str(exc), raw_text=raw_instruction)
    return TacticalCommand.from_plan(plan)


def parse_tactical_command(
    user_input: str,
    *,
    client: SiliconFlowClient | None = None,
    agent_id: str = "A0100",
    situation: TacticalSituation | None = None,
    enable_complex_plan: bool = True,
    max_plan_actions: int = 4,
    default_min_steps: int = 3,
    default_max_steps: int = 10,
) -> TacticalCommand:
    text = user_input.strip()
    if not text:
        return TacticalCommand.invalid("输入为空。", raw_text=user_input)

    normalized = text.lower()
    is_complex = enable_complex_plan and _looks_like_complex_task_chain(normalized)
    if not is_complex:
        decision = parse_tactical_instruction(text, client=client, agent_id=agent_id, situation=situation)
        return TacticalCommand.from_decision(decision)

    if client is not None:
        action_table = "\n".join(f"{item.action_id}: {item.code}（{item.chinese_name}）- {item.description}" for item in TACTICAL_ACTIONS)
        situation_text = situation.to_prompt_text() if situation is not None else "当前态势摘要：未提供。"
        try:
            content = client.chat(
                [
                    LLMMessage("system", TACTICAL_PLAN_SYSTEM_PROMPT),
                    LLMMessage(
                        "user",
                        f"可选战术动作如下：\n{action_table}\n\n{situation_text}\n\n用户复杂指令：{text}",
                    ),
                ],
                temperature=0.0,
                max_tokens=900,
                enable_thinking=False,
            )
            command = parse_tactical_plan_json(
                content,
                raw_instruction=user_input,
                agent_id=agent_id,
                max_plan_actions=max_plan_actions,
                default_min_steps=default_min_steps,
                default_max_steps=default_max_steps,
            )
            try:
                llm_marked_invalid = _json_object(content).get("valid") is False
            except Exception:
                llm_marked_invalid = False
            if command.valid or llm_marked_invalid:
                return command
        except Exception:
            pass

    return keyword_parse_tactical_plan(
        text,
        agent_id=agent_id,
        situation=situation,
        max_plan_actions=max_plan_actions,
        default_min_steps=default_min_steps,
        default_max_steps=default_max_steps,
    )
