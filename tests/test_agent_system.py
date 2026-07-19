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
from agent_system.tactical_plan_executor import TacticalPlanExecutor
from agent_system.tactical_planner import parse_tactical_command
from agent_system.tactical_state import TacticalSituation, situation_from_obs
from agent_system.tactical_policy import resolve_actor_checkpoint_path
from agent_system.tactical_safety import TacticalSafetyState, apply_tactical_safety
from agent_system.tactical_scheduler import TacticalActionScheduler
from experiments import agent_llm_fusion_experiments as fusion_experiments
from experiments.agent_llm_fusion_experiments import (
    DEFAULT_MANUAL_SCHEDULE,
    SAFETY_REASON_DISPLAY_NAMES,
    build_base_instruction_dataset,
    build_complex_instruction_dataset,
    build_instruction_dataset,
    build_invalid_instruction_dataset,
    build_safety_cases,
    build_state_aware_instruction_dataset,
    evaluate_safety_cases,
    parse_manual_schedule,
    parsing_source_display,
    timeline_source_display_counts,
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


def test_tactical_situation_from_obs_extracts_risk_tags() -> None:
    obs = [0.68, 0.0, 1.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.9, 0.0, -0.4, 2.5, 0.7, 0.1, 1.0]
    situation = situation_from_obs(obs, altitude_limit_m=2500.0)

    assert round(situation.altitude_m or 0.0, 3) == 3400.0
    assert situation.speed_mps == 306.0
    assert situation.delta_altitude_m == -400.0
    assert situation.distance_m == 1000.0
    assert situation.low_altitude_risk
    assert situation.close_fast_risk
    assert situation.bad_posture_risk
    assert situation.has_altitude_advantage
    assert situation.to_log()["side_label"] == "左侧"


def test_tactical_keyword_parser_uses_situation_hints() -> None:
    low = TacticalSituation(altitude_m=3200.0, altitude_limit_m=2500.0)
    close_fast = TacticalSituation(distance_m=900.0, speed_mps=300.0)
    bad_posture = TacticalSituation(ao_rad=2.6, ta_rad=0.5)

    assert keyword_parse_tactical_instruction("高度太低，拉起来", situation=low).action_id == 4
    assert keyword_parse_tactical_instruction("别冲过头", situation=close_fast).action_id == 7
    assert keyword_parse_tactical_instruction("现在太被动，先保命", situation=bad_posture).action_id == 3
    assert keyword_parse_tactical_instruction("速度不够，补能量").action_id == 6


def test_tactical_complex_command_keyword_plan() -> None:
    first = parse_tactical_command("先提前量追击超过他再爬升占位", client=None)
    second = parse_tactical_command("先俯冲加速再高悠悠", client=None)
    invalid = parse_tactical_command("先帮我写摘要再生成代码", client=None)

    assert first.kind == "plan"
    assert first.plan is not None
    assert [step.action_id for step in first.plan.steps] == [1, 4]
    assert [step.until for step in first.plan.steps] == ["range_close", "altitude_advantage"]
    assert second.kind == "plan"
    assert second.plan is not None
    assert [step.action_id for step in second.plan.steps] == [5, 10]
    assert invalid.kind == "invalid"


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


def test_tactical_plan_executor_switches_then_returns_to_actor_fallback() -> None:
    command = parse_tactical_command("先提前量追击超过他再爬升占位", client=None, default_min_steps=2, default_max_steps=4)
    assert command.plan is not None
    executor = TacticalPlanExecutor()
    executor.start(command.plan)

    far = TacticalSituation(distance_m=5000.0, delta_altitude_m=100.0)
    close = TacticalSituation(distance_m=1500.0, delta_altitude_m=100.0)
    high = TacticalSituation(distance_m=1500.0, delta_altitude_m=-500.0)

    assert executor.select(far).action_id == 1
    assert executor.select(far).action_id == 1
    switched = executor.select(close)
    assert switched is not None
    assert switched.action_id == 4
    assert switched.plan_step_index == 2
    assert executor.select(high).action_id == 4
    assert executor.select(high) is None
    assert not executor.active


def test_tactical_plan_output_still_passes_safety_override() -> None:
    command = parse_tactical_command("先俯冲加速再高悠悠", client=None, default_min_steps=1, default_max_steps=3)
    assert command.plan is not None
    executor = TacticalPlanExecutor()
    executor.start(command.plan)
    low = TacticalSituation(altitude_m=3200.0, altitude_limit_m=2500.0)

    selected = executor.select(low)
    assert selected is not None
    assert selected.action_id == 5
    safety = apply_tactical_safety(selected.action_id, state=low.to_safety_state(), fallback_action_id=0)
    assert safety.action_id == 4
    assert safety.overridden


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
    assert "--max-plan-actions" in spec.command
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

    complex_disabled = build_tactical_demo_command(
        TacticalDemoConfig(disable_complex_plan=True, max_plan_actions=3, render_mode="none")
    )
    assert "--disable-complex-plan" in complex_disabled.command
    assert "--max-plan-actions" in complex_disabled.command
    assert "3" in complex_disabled.command

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
    assert parse_manual_schedule(DEFAULT_MANUAL_SCHEDULE) == {
        30: "减速",
        90: "向左转弯",
        150: "先提前量追击接近敌机，再爬升占位",
    }


def test_agent_llm_experiment_dataset_covers_12_actions() -> None:
    cases = build_instruction_dataset()
    base_cases = build_base_instruction_dataset()
    state_cases = build_state_aware_instruction_dataset()
    complex_cases = build_complex_instruction_dataset()
    invalid_cases = build_invalid_instruction_dataset()
    expected_ids = {case.expected_action_id for case in base_cases if case.expected_action_id is not None}

    assert len(cases) == 194
    assert len(base_cases) == 120
    assert len(state_cases) == 30
    assert len(complex_cases) == 24
    assert len(invalid_cases) == 20
    assert expected_ids == set(ACTION_BY_ID)

    for action_id in ACTION_BY_ID:
        action_cases = [case for case in base_cases if case.expected_action_id == action_id]
        assert len(action_cases) == 10
        assert any(any(ch.isascii() and ch.isalpha() for ch in case.instruction) for case in action_cases)

    assert {case.expected_action_id for case in state_cases} == {3, 4, 7}
    assert all(case.situation is not None for case in state_cases)
    assert all(len(case.expected_actions) == 2 for case in complex_cases)
    complex_action_counts = {
        action_id: sum(case.expected_actions.count(action_id) for case in complex_cases)
        for action_id in ACTION_BY_ID
    }
    assert min(complex_action_counts.values()) >= 2
    assert all(case.expected_kind == "invalid" for case in invalid_cases)


def test_agent_llm_experiment_keyword_dataset_is_unambiguous() -> None:
    rows, summary = fusion_experiments.evaluate_instruction_cases(
        build_base_instruction_dataset(),
        parser_mode="keyword",
        agent_id="A0100",
    )
    assert summary["base_case_total"] == 120
    assert summary["repeat_count"] == 1
    assert summary["correct_count"] == summary["total"]
    assert [row for row in rows if not row["correct"]] == []


def test_agent_llm_experiment_complex_keyword_plans_match_two_actions() -> None:
    rows, summary = fusion_experiments.evaluate_instruction_cases(
        build_complex_instruction_dataset(),
        parser_mode="keyword",
        agent_id="A0100",
    )
    assert summary["category_summaries"]["complex_plan"]["total"] == 24
    assert all(row["predicted_kind"] == "plan" for row in rows)
    assert all(row["correct"] for row in rows)


def test_agent_llm_experiment_state_context_ablation_is_paired() -> None:
    cases = build_state_aware_instruction_dataset()
    prompts = {case.instruction for case in cases}
    assert len(prompts) == 10
    for prompt in prompts:
        prompt_cases = [case for case in cases if case.instruction == prompt]
        assert len(prompt_cases) == 3
        assert {case.expected_action_id for case in prompt_cases} == {3, 4, 7}

    with_context_rows, _ = fusion_experiments.evaluate_instruction_cases(
        cases[:3],
        parser_mode="keyword",
        agent_id="A0100",
        include_state_context=True,
    )
    without_context_rows, _ = fusion_experiments.evaluate_instruction_cases(
        cases[:3],
        parser_mode="keyword",
        agent_id="A0100",
        include_state_context=False,
    )
    assert all(row["situation"] for row in with_context_rows)
    assert all(not row["situation"] for row in without_context_rows)


def test_agent_llm_experiment_uses_chinese_display_labels() -> None:
    assert parsing_source_display("llm") == "LLM解析"
    assert parsing_source_display("llm_plan") == "LLM解析"
    assert parsing_source_display("keyword") == "关键词兜底"
    assert parsing_source_display("keyword_plan") == "关键词兜底"
    assert timeline_source_display_counts(
        [{"source": "actor_fallback"}, {"source": "manual"}, {"source": "manual_plan"}]
    ) == {"自主决策": 1, "人类指令": 2}
    assert SAFETY_REASON_DISPLAY_NAMES == {
        "none": "安全动作",
        "invalid_action": "非法动作",
        "low_altitude": "低空",
        "close_fast": "近距高速",
        "bad_posture": "不利姿态",
        "other": "其他",
    }


def test_agent_llm_experiment_timeline_merges_safety_override_ranges() -> None:
    rows = [
        {"step": 66, "safety_overridden": False},
        {"step": 67, "safety_overridden": True},
        {"step": 68, "safety_overridden": "True"},
        {"step": 69, "safety_overridden": False},
        {"step": 208, "safety_overridden": True},
        {"step": 209, "safety_overridden": True},
    ]
    ranges = fusion_experiments._timeline_step_ranges(
        rows,
        lambda row: fusion_experiments._timeline_flag_is_true(row["safety_overridden"]),
    )
    assert ranges == [(67, 68), (208, 209)]


def test_agent_llm_experiment_llm_mode_repeats_each_case(monkeypatch) -> None:
    monkeypatch.setattr(fusion_experiments, "make_llm_client", lambda parser_mode: None)
    rows, summary = fusion_experiments.evaluate_instruction_cases(
        build_base_instruction_dataset()[:1],
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
    assert set(summary["reason_display_counts"]).issubset(set(SAFETY_REASON_DISPLAY_NAMES.values()))
    assert by_case["low_dive_at_margin"]["final_action_id"] == 4
    assert by_case["low_dive_above_margin"]["overridden"] is False
    assert by_case["close_fast_pure_at_distance"]["final_action_id"] == 7
    assert by_case["close_fast_lead_speed_below"]["overridden"] is False
    assert by_case["bad_posture_pure"]["final_action_id"] == 3
    assert by_case["bad_posture_ta_equal"]["overridden"] is False
    assert by_case["invalid_99_fallback_lag"]["final_action_id"] == 2
