from __future__ import annotations

from dataclasses import dataclass

from .tactical_plan import TacticalPlan, TacticalPlanStep
from .tactical_state import TacticalSituation


@dataclass(frozen=True)
class PlanExecutionResult:
    action_id: int
    source: str
    reason: str
    plan_id: str
    plan_step_index: int
    plan_total_steps: int
    plan_until: str
    plan_status: str
    step_elapsed: int
    step_min_steps: int
    step_max_steps: int


class TacticalPlanExecutor:
    """执行有限多步战术计划，并根据态势条件切换步骤。"""

    def __init__(self) -> None:
        self._plan: TacticalPlan | None = None
        self._step_index = 0
        self._step_elapsed = 0
        self.last_status = "idle"

    @property
    def active(self) -> bool:
        return self._plan is not None

    @property
    def current_plan(self) -> TacticalPlan | None:
        return self._plan

    def clear(self) -> None:
        self._plan = None
        self._step_index = 0
        self._step_elapsed = 0
        self.last_status = "idle"

    def start(self, plan: TacticalPlan) -> None:
        self._plan = plan
        self._step_index = 0
        self._step_elapsed = 0
        self.last_status = "started"

    def _current_step(self) -> TacticalPlanStep | None:
        if self._plan is None or self._step_index >= len(self._plan.steps):
            return None
        return self._plan.steps[self._step_index]

    def _advance_step(self, status: str) -> None:
        if self._plan is None:
            self.last_status = "idle"
            return
        self._step_index += 1
        self._step_elapsed = 0
        if self._step_index >= len(self._plan.steps):
            self._plan = None
            self.last_status = "completed"
        else:
            self.last_status = status

    def _should_advance(self, step: TacticalPlanStep, situation: TacticalSituation | None) -> bool:
        if self._step_elapsed < step.min_steps:
            return False
        if self._step_elapsed >= step.max_steps:
            return True
        if step.until == "fixed_steps":
            return False
        return situation is not None and situation.matches_until(step.until)

    def select(self, situation: TacticalSituation | None = None) -> PlanExecutionResult | None:
        while self._plan is not None:
            step = self._current_step()
            if step is None:
                self.clear()
                return None
            if self._should_advance(step, situation):
                status = "step_switched_by_max_steps" if self._step_elapsed >= step.max_steps else f"step_switched_by_{step.until}"
                self._advance_step(status)
                continue

            self._step_elapsed += 1
            self.last_status = "running"
            return PlanExecutionResult(
                action_id=step.action_id,
                source="manual_plan",
                reason=step.reason or self._plan.reason,
                plan_id=self._plan.plan_id,
                plan_step_index=self._step_index + 1,
                plan_total_steps=len(self._plan.steps),
                plan_until=step.until,
                plan_status=self.last_status,
                step_elapsed=self._step_elapsed,
                step_min_steps=step.min_steps,
                step_max_steps=step.max_steps,
            )
        return None
