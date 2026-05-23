from __future__ import annotations

from dataclasses import dataclass

from .tactical_actions import ACTION_BY_ID, action_name, coerce_action_id
from .tactical_parser import TacticalDecision


@dataclass(frozen=True)
class ScheduledTacticalAction:
    action_id: int
    source: str
    reason: str
    remaining_manual_steps: int
    actor_action_id: int | None = None
    manual_action_id: int | None = None


class TacticalActionScheduler:
    """在人工接管和 actor fallback 之间切换。"""

    def __init__(self, *, hold_steps: int = 10, default_action_id: int = 0) -> None:
        self.hold_steps = max(int(hold_steps), 1)
        self.default_action_id = default_action_id
        self._manual_decision: TacticalDecision | None = None
        self._remaining_manual_steps = 0

    def clear_manual(self) -> None:
        self._manual_decision = None
        self._remaining_manual_steps = 0

    def update_manual(self, decision: TacticalDecision | None) -> None:
        if decision is None:
            return
        action_id = coerce_action_id(decision.action_id)
        if not decision.valid or action_id is None:
            return
        self._manual_decision = decision
        self._remaining_manual_steps = self.hold_steps

    def select(self, *, actor_action_id: int | None, manual_decision: TacticalDecision | None = None) -> ScheduledTacticalAction:
        self.update_manual(manual_decision)

        if self._manual_decision is not None and self._remaining_manual_steps > 0:
            action_id = int(self._manual_decision.action_id)
            self._remaining_manual_steps -= 1
            return ScheduledTacticalAction(
                action_id=action_id,
                source="manual",
                reason=self._manual_decision.reason or f"人工指令接管为 {action_name(action_id)}。",
                remaining_manual_steps=self._remaining_manual_steps,
                actor_action_id=coerce_action_id(actor_action_id),
                manual_action_id=action_id,
            )

        self.clear_manual()
        actor_action = coerce_action_id(actor_action_id)
        if actor_action is None:
            actor_action = self.default_action_id if self.default_action_id in ACTION_BY_ID else 0
            reason = "actor 输出非法，使用默认战术动作。"
        else:
            reason = "无人工指令，使用 actor fallback 自主动作。"
        return ScheduledTacticalAction(
            action_id=actor_action,
            source="actor_fallback",
            reason=reason,
            remaining_manual_steps=0,
            actor_action_id=actor_action,
            manual_action_id=None,
        )
