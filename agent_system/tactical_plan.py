from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .tactical_actions import ACTION_BY_ID, action_name, action_name_matches, coerce_action_id

if TYPE_CHECKING:
    from .tactical_parser import TacticalDecision


VALID_UNTIL_CONDITIONS = {
    "fixed_steps",
    "range_close",
    "range_opened",
    "altitude_advantage",
    "overshoot_risk_reduced",
    "bad_posture_recovered",
}


@dataclass(frozen=True)
class TacticalPlanStep:
    action_id: int
    tactical_action_name: str
    min_steps: int = 3
    max_steps: int = 10
    until: str = "fixed_steps"
    reason: str = ""

    @classmethod
    def build(
        cls,
        action_id: object,
        *,
        tactical_action_name: str = "",
        min_steps: int = 3,
        max_steps: int = 10,
        until: str = "fixed_steps",
        reason: str = "",
    ) -> "TacticalPlanStep":
        coerced = coerce_action_id(action_id)
        if coerced is None:
            raise ValueError(f"计划步骤动作编号非法: {action_id!r}")
        if tactical_action_name and not action_name_matches(coerced, tactical_action_name):
            raise ValueError(f"计划步骤动作名 {tactical_action_name!r} 与编号 {coerced} 不一致。")
        safe_min = max(int(min_steps), 1)
        safe_max = max(int(max_steps), safe_min)
        safe_until = str(until or "fixed_steps").strip()
        if safe_until not in VALID_UNTIL_CONDITIONS:
            safe_until = "fixed_steps"
        return cls(
            action_id=coerced,
            tactical_action_name=action_name(coerced),
            min_steps=safe_min,
            max_steps=safe_max,
            until=safe_until,
            reason=reason,
        )

    def to_log(self) -> dict[str, object]:
        return {
            "tactical_action_id": self.action_id,
            "tactical_action_name": self.tactical_action_name,
            "tactical_action_cn": ACTION_BY_ID[self.action_id].chinese_name,
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "until": self.until,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TacticalPlan:
    steps: tuple[TacticalPlanStep, ...]
    reason: str
    source: str
    raw_text: str
    agent_id: str = "A0100"
    scene: str = "1v1"
    plan_id: str = ""
    valid: bool = True

    @classmethod
    def build(
        cls,
        steps: list[TacticalPlanStep] | tuple[TacticalPlanStep, ...],
        *,
        reason: str,
        source: str,
        raw_text: str,
        agent_id: str = "A0100",
        scene: str = "1v1",
    ) -> "TacticalPlan":
        step_tuple = tuple(steps)
        if len(step_tuple) < 2:
            raise ValueError("复杂计划至少需要 2 个战术步骤。")
        plan_key = f"{agent_id}|{raw_text}|{','.join(str(step.action_id) for step in step_tuple)}"
        plan_id = hashlib.sha1(plan_key.encode("utf-8")).hexdigest()[:10]
        return cls(
            steps=step_tuple,
            reason=reason,
            source=source,
            raw_text=raw_text,
            agent_id=agent_id,
            scene=scene,
            plan_id=plan_id,
            valid=True,
        )

    def to_log(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "source": self.source,
            "scene": self.scene,
            "agent_id": self.agent_id,
            "plan_id": self.plan_id,
            "reason": self.reason,
            "raw_text": self.raw_text,
            "steps": [step.to_log() for step in self.steps],
        }


@dataclass(frozen=True)
class TacticalCommand:
    kind: str
    decision: "TacticalDecision | None" = None
    plan: TacticalPlan | None = None
    reason: str = ""
    raw_text: str = ""

    @classmethod
    def from_decision(cls, decision: "TacticalDecision") -> "TacticalCommand":
        return cls("decision" if decision.valid else "invalid", decision=decision, reason=decision.reason, raw_text=decision.raw_text)

    @classmethod
    def from_plan(cls, plan: TacticalPlan) -> "TacticalCommand":
        return cls("plan", plan=plan, reason=plan.reason, raw_text=plan.raw_text)

    @classmethod
    def invalid(cls, reason: str, *, raw_text: str = "") -> "TacticalCommand":
        return cls("invalid", reason=reason, raw_text=raw_text)

    @property
    def valid(self) -> bool:
        return self.kind == "plan" or (self.decision is not None and self.decision.valid)
