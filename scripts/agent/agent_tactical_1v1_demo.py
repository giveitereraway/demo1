#!/usr/bin/env python
"""
python scripts/agent/agent_tactical_1v1_demo.py --render-mode real_time
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_system.llm import SiliconFlowClient
from agent_system.settings import AgentSettings, DEFAULT_TACTICAL_ACTOR_PATH
from agent_system.tactical_actions import ACTION_BY_ID, action_chinese_name, action_name, parse_action_reference
from agent_system.tactical_parser import TacticalDecision, decision_to_log
from agent_system.tactical_plan_executor import PlanExecutionResult, TacticalPlanExecutor
from agent_system.tactical_planner import parse_tactical_command
from agent_system.tactical_policy import TacticalActorPolicy, resolve_actor_checkpoint_path
from agent_system.tactical_safety import apply_tactical_safety
from agent_system.tactical_scheduler import ScheduledTacticalAction, TacticalActionScheduler
from agent_system.tactical_state import situation_from_env


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="1v1 LLM-Agent tactical demo with actor fallback.")
    parser.add_argument(
        "--actor-path",
        default=str(DEFAULT_TACTICAL_ACTOR_PATH),
        help="TacticalHierarchySelfplay actor_latest.pt 文件路径，或包含 actor_latest.pt 的 files 目录。",
    )
    parser.add_argument(
        "--enemy-path",
        default=None,
        help="敌机 tactical actor 路径；不传时复用 --actor-path。显式传 --enemy-action 时该参数不参与敌机决策。",
    )
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/TacticalHierarchySelfplay")
    parser.add_argument("--agent-id", default="A0100")
    parser.add_argument("--enemy-action", default=None, help="敌机固定战术动作；传入后优先于 --enemy-path。")
    parser.add_argument("--hold-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda:0")
    parser.add_argument("--render-mode", choices=["none", "txt", "real_time"], default="txt")
    parser.add_argument("--log-path", default="output/agent_tactical_1v1/demo_log.jsonl")
    parser.add_argument("--acmi-path", default="output/agent_tactical_1v1/demo.txt.acmi")
    parser.add_argument("--step-sleep", type=float, default=0.2)
    parser.add_argument("--status-interval", type=int, default=0, help="终端状态打印间隔；0 表示只在人工指令或安全覆盖时打印。")
    parser.add_argument("--verbose-steps", action="store_true", help="每个仿真步都打印状态，主要用于调试。")
    parser.add_argument("--disable-llm", action="store_true", help="只使用关键词解析，不调用 LLM。")
    parser.add_argument("--disable-complex-plan", action="store_true", help="关闭复杂指令多步计划，只保留单动作解析。")
    parser.add_argument("--max-plan-actions", type=int, default=4, help="复杂指令最多拆解出的战术动作数量。")
    return parser


def start_input_thread() -> queue.Queue[str]:
    input_queue: queue.Queue[str] = queue.Queue()

    def worker() -> None:
        while True:
            try:
                line = input()
            except EOFError:
                break
            input_queue.put(line)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return input_queue


def drain_latest_instruction(input_queue: queue.Queue[str]) -> str | None:
    latest: str | None = None
    while True:
        try:
            latest = input_queue.get_nowait()
        except queue.Empty:
            return latest


def resolve_agent_index(env: SingleCombatEnv, agent_id: str) -> int:
    ordered_ids = (env.ego_ids + env.enm_ids)[: env.num_agents]
    if agent_id not in ordered_ids:
        raise ValueError(f"agent-id {agent_id} 不在当前 1v1 环境智能体列表中: {ordered_ids}")
    return ordered_ids.index(agent_id)


def resolve_opponent_index(env: SingleCombatEnv, agent_id: str) -> int:
    ordered_ids = (env.ego_ids + env.enm_ids)[: env.num_agents]
    opponent_indices = [index for index, current_agent_id in enumerate(ordered_ids) if current_agent_id != agent_id]
    if len(opponent_indices) != 1:
        raise ValueError(f"当前 demo 只支持 1v1，无法为 {agent_id} 唯一确定敌机: {ordered_ids}")
    return opponent_indices[0]


def build_action_array(env: SingleCombatEnv, *, agent_id: str, own_action_id: int, enemy_action_id: int) -> np.ndarray:
    ordered_ids = (env.ego_ids + env.enm_ids)[: env.num_agents]
    actions = []
    for current_agent_id in ordered_ids:
        actions.append(own_action_id if current_agent_id == agent_id else enemy_action_id)
    return np.asarray(actions, dtype=np.int64)


def make_client(disable_llm: bool) -> SiliconFlowClient | None:
    if disable_llm:
        return None
    settings = AgentSettings.load()
    return SiliconFlowClient(settings) if settings.has_llm_credentials else None


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def write_log(log_file, payload: dict[str, object]) -> None:
    log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    log_file.flush()


def should_print_status(
    *,
    step: int,
    status_interval: int,
    verbose_steps: bool,
    had_instruction: bool,
    safety_overridden: bool,
) -> bool:
    if verbose_steps:
        return True
    if had_instruction or safety_overridden:
        return True
    return status_interval > 0 and step % status_interval == 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    fixed_enemy_action_id: int | None = None
    if args.enemy_action:
        fixed_enemy_action_id = parse_action_reference(args.enemy_action)
        if fixed_enemy_action_id is None:
            raise ValueError(f"无法识别 enemy-action: {args.enemy_action}")

    actor_path = resolve_actor_checkpoint_path(args.actor_path)
    enemy_path = resolve_actor_checkpoint_path(args.enemy_path or args.actor_path)
    log_path = resolve_repo_path(args.log_path)
    acmi_path = resolve_repo_path(args.acmi_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    acmi_path.parent.mkdir(parents=True, exist_ok=True)

    from envs.JSBSim.envs import SingleCombatEnv

    env = SingleCombatEnv(args.scenario_name)
    env.seed(args.seed)
    policy = TacticalActorPolicy.load(actor_path, env, device_name=args.device)
    enemy_policy = None if fixed_enemy_action_id is not None else TacticalActorPolicy.load(enemy_path, env, device_name=args.device)
    scheduler = TacticalActionScheduler(hold_steps=args.hold_steps)
    plan_executor = TacticalPlanExecutor()
    input_queue = start_input_thread()
    client = make_client(args.disable_llm)
    tacview = None

    if args.render_mode == "real_time":
        from runner.tacview import Tacview

        tacview = Tacview()

    print("1v1 LLM-Agent tactical demo started.")
    print("输入中文战术短句后回车可临时接管；直接不输入时使用 actor fallback。输入 quit/exit/退出 结束。")
    print("复杂指令会被解析为有限多步战术计划；每一步仍是 0-11 的高层 tactical action。")
    print("终端默认低频打印状态，完整逐步记录写入 JSONL 日志；如需逐步刷屏可加 --verbose-steps。")
    print(f"actor: {actor_path}")
    if fixed_enemy_action_id is None:
        print(f"enemy actor: {enemy_path}")
    else:
        print(f"enemy fixed action: {fixed_enemy_action_id}:{action_chinese_name(fixed_enemy_action_id)}")
    print(f"log: {log_path}")

    obs = env.reset()
    policy.reset()
    if enemy_policy is not None:
        enemy_policy.reset()
    agent_index = resolve_agent_index(env, args.agent_id)
    enemy_index = resolve_opponent_index(env, args.agent_id)
    dones = np.zeros((env.num_agents, 1), dtype=bool)
    last_safe_action_id = 0

    if args.render_mode == "txt":
        env.render(mode="txt", filepath=str(acmi_path))
    elif args.render_mode == "real_time":
        env.render(mode="real_time", tacview=tacview)

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            write_log(
                log_file,
                {
                    "event": "start",
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "scenario_name": args.scenario_name,
                    "agent_id": args.agent_id,
                    "actor_path": str(actor_path),
                    "enemy_source": "fixed_action" if fixed_enemy_action_id is not None else "enemy_actor",
                    "enemy_path": str(enemy_path) if fixed_enemy_action_id is None else "",
                    "enemy_fixed_action_id": fixed_enemy_action_id,
                    "enemy_fixed_action_name": action_name(fixed_enemy_action_id) if fixed_enemy_action_id is not None else "",
                    "hold_steps": args.hold_steps,
                    "disable_complex_plan": args.disable_complex_plan,
                    "max_plan_actions": args.max_plan_actions,
                    "status_interval": args.status_interval,
                    "verbose_steps": args.verbose_steps,
                },
            )

            while env.current_step < args.max_steps and not bool(np.asarray(dones).all()):
                situation = situation_from_env(env, args.agent_id)
                raw_instruction = drain_latest_instruction(input_queue)
                if raw_instruction is not None and raw_instruction.strip().lower() in {"quit", "exit", "q", "退出", "结束"}:
                    break

                manual_decision: TacticalDecision | None = None
                parsed_command_kind = "none"
                parsed_plan_log: dict[str, object] | None = None
                if raw_instruction is not None and raw_instruction.strip():
                    command = parse_tactical_command(
                        raw_instruction,
                        client=client,
                        agent_id=args.agent_id,
                        situation=situation,
                        enable_complex_plan=not args.disable_complex_plan,
                        max_plan_actions=args.max_plan_actions,
                        default_min_steps=3,
                        default_max_steps=args.hold_steps,
                    )
                    parsed_command_kind = command.kind
                    if command.kind == "plan" and command.plan is not None:
                        scheduler.clear_manual()
                        plan_executor.start(command.plan)
                        parsed_plan_log = command.plan.to_log()
                        print(f"[manual plan] {raw_instruction}: {command.reason}")
                    elif command.kind == "decision" and command.decision is not None and command.decision.valid:
                        plan_executor.clear()
                        manual_decision = command.decision
                    else:
                        print(f"[manual ignored] {raw_instruction}: {command.reason}")

                actor_action_id = policy.act(obs[agent_index])
                if fixed_enemy_action_id is None:
                    if enemy_policy is None:
                        raise RuntimeError("enemy_policy 未初始化")
                    enemy_action_id = enemy_policy.act(obs[enemy_index])
                    enemy_source = "enemy_actor"
                else:
                    enemy_action_id = fixed_enemy_action_id
                    enemy_source = "fixed_action"

                plan_result: PlanExecutionResult | None = plan_executor.select(situation)
                if plan_result is not None:
                    scheduled = ScheduledTacticalAction(
                        action_id=plan_result.action_id,
                        source=plan_result.source,
                        reason=plan_result.reason,
                        remaining_manual_steps=0,
                        actor_action_id=actor_action_id,
                        manual_action_id=plan_result.action_id,
                    )
                else:
                    scheduled = scheduler.select(actor_action_id=actor_action_id, manual_decision=manual_decision)
                safety = apply_tactical_safety(
                    scheduled.action_id,
                    state=situation.to_safety_state(),
                    fallback_action_id=last_safe_action_id,
                )
                last_safe_action_id = safety.action_id

                actions = build_action_array(
                    env,
                    agent_id=args.agent_id,
                    own_action_id=safety.action_id,
                    enemy_action_id=enemy_action_id,
                )
                obs, rewards, dones, info = env.step(actions)

                if args.render_mode == "txt":
                    env.render(mode="txt", filepath=str(acmi_path))
                elif args.render_mode == "real_time":
                    env.render(mode="real_time", tacview=tacview)

                log_payload = {
                    "event": "step",
                    "step": int(env.current_step),
                    "source": scheduled.source,
                    "command_kind": parsed_command_kind,
                    "instruction": raw_instruction or "",
                    "situation": situation.to_log(),
                    "actor_action_id": actor_action_id,
                    "actor_action_name": action_name(actor_action_id),
                    "enemy_source": enemy_source,
                    "enemy_action_id": enemy_action_id,
                    "enemy_action_name": action_name(enemy_action_id),
                    "enemy_action_cn": action_chinese_name(enemy_action_id),
                    "scheduled_action_id": scheduled.action_id,
                    "scheduled_action_name": action_name(scheduled.action_id),
                    "final_action_id": safety.action_id,
                    "final_action_name": action_name(safety.action_id),
                    "final_action_cn": action_chinese_name(safety.action_id),
                    "safety_overridden": safety.overridden,
                    "safety_reason": safety.reason,
                    "remaining_manual_steps": scheduled.remaining_manual_steps,
                    "plan_active": plan_executor.active,
                    "plan_id": plan_result.plan_id if plan_result is not None else "",
                    "plan_step_index": plan_result.plan_step_index if plan_result is not None else 0,
                    "plan_total_steps": plan_result.plan_total_steps if plan_result is not None else 0,
                    "plan_until": plan_result.plan_until if plan_result is not None else "",
                    "plan_status": plan_result.plan_status if plan_result is not None else plan_executor.last_status,
                    "reward": np.asarray(rewards).reshape(-1).astype(float).tolist(),
                    "done": np.asarray(dones).reshape(-1).astype(bool).tolist(),
                }
                if manual_decision is not None:
                    log_payload["manual_decision"] = decision_to_log(manual_decision)
                if parsed_plan_log is not None:
                    log_payload["manual_plan"] = parsed_plan_log
                write_log(log_file, log_payload)

                if should_print_status(
                    step=int(env.current_step),
                    status_interval=args.status_interval,
                    verbose_steps=args.verbose_steps,
                    had_instruction=manual_decision is not None or raw_instruction is not None or plan_result is not None,
                    safety_overridden=safety.overridden,
                ):
                    if plan_result is not None:
                        print(
                            f"[step {env.current_step:04d}] source=manual_plan "
                            f"step={plan_result.plan_step_index}/{plan_result.plan_total_steps} "
                            f"action={safety.action_id}:{ACTION_BY_ID[safety.action_id].chinese_name} "
                            f"until={plan_result.plan_until} "
                            f"actor={actor_action_id}:{ACTION_BY_ID[actor_action_id].chinese_name} "
                            f"enemy={enemy_action_id}:{ACTION_BY_ID[enemy_action_id].chinese_name}({enemy_source})"
                            + (f" override={safety.reason}" if safety.overridden else "")
                        )
                    else:
                        print(
                            f"[step {env.current_step:04d}] source={scheduled.source} "
                            f"action={safety.action_id}:{ACTION_BY_ID[safety.action_id].chinese_name} "
                            f"actor={actor_action_id}:{ACTION_BY_ID[actor_action_id].chinese_name} "
                            f"enemy={enemy_action_id}:{ACTION_BY_ID[enemy_action_id].chinese_name}({enemy_source})"
                            + (f" override={safety.reason}" if safety.overridden else "")
                        )

                if args.step_sleep > 0:
                    time.sleep(args.step_sleep)
    finally:
        env.close()

    print("demo finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
