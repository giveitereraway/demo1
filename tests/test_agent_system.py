from __future__ import annotations

import json

from agent_system.commands import TacticalDemoConfig, _path, build_tactical_demo_command
from agent_system.settings import AgentSettings, DEFAULT_TACTICAL_ACTOR_PATH, REPO_ROOT
from agent_system.tactical_actions import ACTION_BY_ID, action_name_matches
from agent_system.tactical_parser import (
    TACTICAL_SYSTEM_PROMPT,
    keyword_parse_tactical_instruction,
    parse_tactical_instruction,
    parse_tactical_json,
)
from agent_system.tactical_policy import resolve_actor_checkpoint_path
from agent_system.tactical_safety import TacticalSafetyState, apply_tactical_safety
from agent_system.tactical_scheduler import TacticalActionScheduler
from experiments import agent_llm_fusion_experiments as fusion_experiments
from experiments.agent_llm_fusion_experiments import (
    build_instruction_dataset,
    build_safety_cases,
    evaluate_safety_cases,
    parse_manual_schedule,
)


def test_path_normalization_accepts_slashes_and_quotes() -> None:
    normalized = _path('"envs\\JSBSim/model\\actor_heading.pt"')
    assert normalized == (REPO_ROOT / "envs" / "JSBSim" / "model" / "actor_heading.pt").resolve()


def test_tactical_action_catalog_has_12_named_actions() -> None:
    assert len(ACTION_BY_ID) == 12
    assert ACTION_BY_ID[0].code == "PURE_PURSUIT"
    assert ACTION_BY_ID[11].chinese_name == "低悠悠"
    assert action_name_matches(1, "LEAD_PURSUIT")
    assert not action_name_matches(1, "LAG_PURSUIT")


def test_tactical_keyword_parser_maps_chinese_phrases() -> None:
    assert keyword_parse_tactical_instruction("追击敌机").action_id == 0
    assert keyword_parse_tactical_instruction("抢占敌机前方").action_id == 1
    assert keyword_parse_tactical_instruction("减速避免冲过头").action_id == 7
    assert keyword_parse_tactical_instruction("向左防御转弯").action_id == 8
    assert not keyword_parse_tactical_instruction("先提前量追击超过他再爬升占位").valid
    assert not keyword_parse_tactical_instruction("先搜索目标再规划复杂任务").valid


def test_tactical_llm_json_validation() -> None:
    valid = parse_tactical_json(
        json.dumps(
            {
                "scene": "1v1",
                "agent_id": "A0100",
                "tactical_action_id": 1,
                "tactical_action_name": "LEAD_PURSUIT",
                "reason": "抢占前方",
            },
            ensure_ascii=False,
        )
    )
    assert valid.valid
    assert valid.action_id == 1

    mismatch = parse_tactical_json(
        json.dumps(
            {
                "scene": "1v1",
                "agent_id": "A0100",
                "tactical_action_id": 1,
                "tactical_action_name": "LAG_PURSUIT",
                "reason": "错误",
            },
            ensure_ascii=False,
        )
    )
    assert not mismatch.valid

    out_of_range = parse_tactical_json(
        json.dumps(
            {
                "scene": "1v1",
                "agent_id": "A0100",
                "tactical_action_id": 99,
                "tactical_action_name": "UNKNOWN",
                "reason": "错误",
            },
            ensure_ascii=False,
        )
    )
    assert not out_of_range.valid

    invalid_instruction = parse_tactical_json(
        json.dumps(
            {
                "scene": "1v1",
                "agent_id": "A0100",
                "tactical_action_id": -1,
                "tactical_action_name": "INVALID",
                "reason": "与空战战术无关，拒绝执行。",
            },
            ensure_ascii=False,
        )
    )
    assert not invalid_instruction.valid
    assert invalid_instruction.action_id is None
    assert invalid_instruction.tactical_action_name == "INVALID"
    assert "拒绝执行" in invalid_instruction.reason


def test_tactical_llm_prompt_describes_invalid_instruction_handling() -> None:
    assert "无效指令" in TACTICAL_SYSTEM_PROMPT
    assert 'tactical_action_id": -1' in TACTICAL_SYSTEM_PROMPT
    assert 'tactical_action_name": "INVALID"' in TACTICAL_SYSTEM_PROMPT
    assert "超出系统能力边界" in TACTICAL_SYSTEM_PROMPT


def test_tactical_llm_parser_disables_thinking_for_json() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.kwargs = {}

        def chat(self, messages, **kwargs):
            self.kwargs = kwargs
            return json.dumps(
                {
                    "scene": "1v1",
                    "agent_id": "A0100",
                    "tactical_action_id": 4,
                    "tactical_action_name": "CLIMB_POSITION",
                    "reason": "用户要求爬升占位。",
                },
                ensure_ascii=False,
            )

    client = FakeClient()
    decision = parse_tactical_instruction("爬升占位", client=client)

    assert decision.valid
    assert decision.source == "llm"
    assert decision.action_id == 4
    assert client.kwargs["enable_thinking"] is False


def test_tactical_llm_invalid_decision_does_not_keyword_fallback() -> None:
    class FakeClient:
        def chat(self, messages, **kwargs):
            return json.dumps(
                {
                    "scene": "1v1",
                    "agent_id": "A0100",
                    "tactical_action_id": -1,
                    "tactical_action_name": "INVALID",
                    "reason": "该输入是复杂任务链，当前版本只支持单个高层战术动作。",
                },
                ensure_ascii=False,
            )

    decision = parse_tactical_instruction("先提前量追击超过他再爬升占位", client=FakeClient())

    assert not decision.valid
    assert decision.action_id is None
    assert decision.tactical_action_name == "INVALID"


def test_tactical_scheduler_uses_actor_fallback_then_manual_hold() -> None:
    scheduler = TacticalActionScheduler(hold_steps=2)
    first = scheduler.select(actor_action_id=6)
    assert first.source == "actor_fallback"
    assert first.action_id == 6

    manual = keyword_parse_tactical_instruction("爬升占位")
    second = scheduler.select(actor_action_id=6, manual_decision=manual)
    third = scheduler.select(actor_action_id=6)
    fourth = scheduler.select(actor_action_id=6)

    assert second.source == "manual"
    assert second.action_id == 4
    assert third.source == "manual"
    assert third.action_id == 4
    assert fourth.source == "actor_fallback"
    assert fourth.action_id == 6


def test_tactical_safety_overrides_risky_actions() -> None:
    low_altitude = TacticalSafetyState(altitude_m=3200.0, altitude_limit_m=2500.0)
    low_result = apply_tactical_safety(5, state=low_altitude, fallback_action_id=0)
    assert low_result.action_id == 4
    assert low_result.overridden

    close_fast = TacticalSafetyState(distance_m=900.0, speed_mps=300.0)
    close_result = apply_tactical_safety(1, state=close_fast, fallback_action_id=0)
    assert close_result.action_id == 7
    assert close_result.overridden

    invalid = apply_tactical_safety(99, fallback_action_id=2)
    assert invalid.action_id == 2
    assert invalid.overridden


def test_build_tactical_demo_command_uses_default_actor_and_enemy_paths() -> None:
    spec = build_tactical_demo_command(
        TacticalDemoConfig(
            max_steps=5,
            render_mode="none",
            disable_llm=True,
        )
    )
    assert spec.command[0] == AgentSettings.load().runtime_python
    assert "agent_tactical_1v1_demo.py" in spec.preview()
    assert "--actor-path" in spec.command
    assert "--enemy-path" in spec.command
    assert "--enemy-action" not in spec.command
    assert "--render-mode" in spec.command
    assert "none" in spec.command
    assert "--status-interval" in spec.command
    assert "--step-sleep" in spec.command
    assert "--disable-llm" in spec.command
    assert spec.validation_errors == []
    default_checkpoint = str(DEFAULT_TACTICAL_ACTOR_PATH / "actor_latest.pt")
    assert spec.command.count(default_checkpoint) == 2

    directory_spec = build_tactical_demo_command(
        TacticalDemoConfig(actor_path=str(DEFAULT_TACTICAL_ACTOR_PATH), render_mode="none")
    )
    assert directory_spec.command.count(default_checkpoint) == 2
    assert directory_spec.validation_errors == []

    fixed_enemy = build_tactical_demo_command(
        TacticalDemoConfig(enemy_action="PURE_PURSUIT", render_mode="none")
    )
    assert "--enemy-action" in fixed_enemy.command
    assert "PURE_PURSUIT" in fixed_enemy.command
    assert fixed_enemy.validation_errors == []

    missing_enemy = build_tactical_demo_command(
        TacticalDemoConfig(enemy_path="missing_enemy.pt", render_mode="none")
    )
    assert any("Enemy tactical actor" in item for item in missing_enemy.validation_errors)

    invalid_enemy_action = build_tactical_demo_command(
        TacticalDemoConfig(enemy_action="NOT_A_TACTIC", render_mode="none")
    )
    assert any("enemy-action" in item for item in invalid_enemy_action.validation_errors)

    missing = build_tactical_demo_command(TacticalDemoConfig(actor_path="missing_actor.pt"))
    assert any("Tactical actor 文件不存在" in item for item in missing.validation_errors)


def test_resolve_actor_checkpoint_accepts_files_directory() -> None:
    assert resolve_actor_checkpoint_path(DEFAULT_TACTICAL_ACTOR_PATH) == DEFAULT_TACTICAL_ACTOR_PATH / "actor_latest.pt"


def test_agent_llm_experiment_manual_schedule_parser() -> None:
    schedule = parse_manual_schedule("30:爬升占位;90:减速避免冲过头;150:向左防御转弯")
    assert schedule == {
        30: "爬升占位",
        90: "减速避免冲过头",
        150: "向左防御转弯",
    }
    assert parse_manual_schedule("") == {}


def test_agent_llm_experiment_dataset_covers_12_actions() -> None:
    cases = build_instruction_dataset()
    expected_ids = {case.expected_action_id for case in cases if case.expected_action_id >= 0}
    invalid_cases = [case for case in cases if case.expected_action_id == -1]
    assert expected_ids == set(ACTION_BY_ID)
    assert len(invalid_cases) == 6

    for action_id in ACTION_BY_ID:
        action_cases = [case for case in cases if case.expected_action_id == action_id]
        assert len(action_cases) == 10
        assert any(any(ch.isascii() and ch.isalpha() for ch in case.instruction) for case in action_cases)


def test_agent_llm_experiment_keyword_dataset_is_unambiguous() -> None:
    rows, summary = fusion_experiments.evaluate_instruction_cases(
        build_instruction_dataset(),
        parser_mode="keyword",
        agent_id="A0100",
    )
    assert summary["base_case_total"] == 126
    assert summary["repeat_count"] == 1
    assert summary["correct_count"] == summary["total"]
    assert [row for row in rows if not row["correct"]] == []


def test_agent_llm_experiment_llm_mode_repeats_each_case(monkeypatch) -> None:
    monkeypatch.setattr(fusion_experiments, "make_llm_client", lambda parser_mode: None)
    rows, summary = fusion_experiments.evaluate_instruction_cases(
        build_instruction_dataset()[:1],
        parser_mode="llm_fallback",
        agent_id="A0100",
    )
    assert len(rows) == 3
    assert summary["base_case_total"] == 1
    assert summary["repeat_count"] == 3
    assert {row["repeat_index"] for row in rows} == {1, 2, 3}


def test_agent_llm_experiment_safety_cases_cover_reasons() -> None:
    cases = build_safety_cases()
    rows, summary = evaluate_safety_cases(cases)
    categories = {row["reason_category"] for row in rows}
    by_case = {row["case_id"]: row for row in rows}

    assert len(cases) == 50
    assert summary["total"] == 50
    assert {"low_altitude", "close_fast", "bad_posture", "invalid_action", "none"}.issubset(categories)
    assert summary["overridden_count"] >= 20
    assert by_case["low_dive_at_margin"]["final_action_id"] == 4
    assert by_case["low_dive_above_margin"]["overridden"] is False
    assert by_case["close_fast_pure_at_distance"]["final_action_id"] == 7
    assert by_case["close_fast_lead_speed_below"]["overridden"] is False
    assert by_case["bad_posture_pure"]["final_action_id"] == 3
    assert by_case["bad_posture_ta_equal"]["overridden"] is False
    assert by_case["invalid_99_fallback_lag"]["final_action_id"] == 2
