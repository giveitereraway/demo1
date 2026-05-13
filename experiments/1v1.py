#!/usr/bin/env python
import csv
import io
import json
import sys
import contextlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces


# 让脚本可以从 experiments/ 目录直接导入项目代码。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.ppo.ppo_actor import PPOActor
from envs.JSBSim.envs import SingleCombatEnv


# =========================
# 可修改实验配置
# =========================

EXPERIMENT_NAME = "selfA_vs_hierarchyselfB"

# 评估环境必须使用直接控制版 SingleCombat 场景。
EVAL_SCENARIO_NAME = "1v1/NoWeapon/Selfplay"

# 两个 actor 可以来自不同训练场景；脚本会按各自场景构造网络动作空间。
ACTOR_A_PATH = REPO_ROOT / "scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/1v1_follow/wandb/offline-run-20260512_175151-yryla8wg/files/actor_latest.pt"
ACTOR_A_SCENARIO_NAME = "1v1/NoWeapon/Selfplay"

ACTOR_B_PATH = REPO_ROOT / "scripts/results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/1v1_follow_hierarchy_2/wandb/run-20260129_153747-15x1ugrb/files/actor_latest.pt"
ACTOR_B_SCENARIO_NAME = "1v1/NoWeapon/HierarchySelfplay"

# 分层 actor 的低层控制器。
LOWLEVEL_ACTOR_PATH = REPO_ROOT / "envs/JSBSim/model/actor_heading.pt"

NUM_EPISODES = 10
SEED = 1
DEVICE = "cuda:0"  # auto / cpu / cuda:0
DETERMINISTIC = True
WIN_REWARD_MARGIN = 100.0

# 是否交替 actor A/B 控制 A0100 与 B0100，减少固定阵营偏差。
SWAP_ACTOR_ORDER = True

# 是否沿用 SingleCombatEnv 默认的初始状态随机换边。
RANDOM_SIDE_SWAP = True

# 可选：覆盖 YAML 初始状态。保持 None 表示使用场景 YAML。
CUSTOM_INITIAL_STATES = None
# CUSTOM_INITIAL_STATES = {
#     "A0100": {"ic_h_sl_ft": 20000, "ic_psi_true_deg": 0.0, "ic_u_fps": 800.0},
#     "B0100": {"ic_h_sl_ft": 20000, "ic_psi_true_deg": 180.0, "ic_u_fps": 800.0},
# }

SAVE_ACMI = True
ACMI_EPISODES = {0}  # 只保存指定回合，避免大量轨迹文件。
OUTPUT_ROOT = REPO_ROOT / "experiments/results"

# 当前 BetaShootBernoulli 内部有调试 print，射击任务中建议保持开启。
SUPPRESS_POLICY_DEBUG_PRINT = True


TASK_FAMILY = {
    "singlecombat": "NoWeapon",
    "hierarchical_singlecombat": "NoWeapon",
    "singlecombat_dodge_missile": "DodgeMissile",
    "hierarchical_singlecombat_dodge_missile": "DodgeMissile",
    "singlecombat_shoot": "ShootMissile",
    "hierarchical_singlecombat_shoot": "ShootMissile",
}

DIRECT_EVAL_TASKS = {
    "singlecombat",
    "singlecombat_dodge_missile",
    "singlecombat_shoot",
}


def _t2n(x):
    return x.detach().cpu().numpy()


def make_actor_args(use_prior: bool) -> SimpleNamespace:
    """构造与训练脚本默认 PPOActor 一致的网络参数。"""
    return SimpleNamespace(
        gain=0.01,
        hidden_size="128 128",
        act_hidden_size="128 128",
        activation_id=1,
        use_feature_normalization=False,
        use_recurrent_policy=True,
        recurrent_hidden_size=128,
        recurrent_hidden_layers=1,
        use_prior=use_prior,
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def task_family(task_name: str) -> str:
    if task_name not in TASK_FAMILY:
        raise ValueError(f"不支持的 SingleCombat task: {task_name}")
    return TASK_FAMILY[task_name]


def action_dim(action_space) -> int:
    """统计 actor 输出动作维度，用于基础兼容性检查。"""
    if isinstance(action_space, spaces.MultiDiscrete):
        return int(action_space.shape[0])
    if isinstance(action_space, spaces.Tuple):
        return int(action_space[0].shape[0] + 1)
    raise TypeError(f"不支持的动作空间: {action_space}")


@dataclass
class ActorController:
    name: str
    path: Path
    scenario_name: str
    task_name: str
    family: str
    is_hierarchical: bool
    is_shoot: bool
    obs_dim: int
    actor_action_dim: int
    actor: PPOActor
    lowlevel_policy: Optional[PPOActor]
    device: torch.device

    def reset(self) -> None:
        # PPO actor 与分层低层 actor 都是 GRU 策略，每局开始清空隐状态。
        self.rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
        self.lowlevel_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
        self.masks = np.ones((1, 1), dtype=np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_batch = np.expand_dims(obs, axis=0)
        if SUPPRESS_POLICY_DEBUG_PRINT:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                actor_action, _, self.rnn_states = self.actor(
                    obs_batch,
                    self.rnn_states,
                    self.masks,
                    deterministic=DETERMINISTIC,
                )
        else:
            actor_action, _, self.rnn_states = self.actor(
                obs_batch,
                self.rnn_states,
                self.masks,
                deterministic=DETERMINISTIC,
            )

        raw_action = _t2n(actor_action).squeeze(0)
        self.rnn_states = _t2n(self.rnn_states)

        if not self.is_hierarchical:
            return raw_action

        lowlevel_action = self._hierarchical_to_lowlevel(obs, raw_action)
        if self.is_shoot:
            return np.concatenate([lowlevel_action, raw_action[-1:]], axis=0)
        return lowlevel_action

    @torch.no_grad()
    def _hierarchical_to_lowlevel(self, obs: np.ndarray, high_action: np.ndarray) -> np.ndarray:
        if self.lowlevel_policy is None:
            raise RuntimeError(f"{self.name} 是分层 actor，但未加载低层控制器。")

        delta_altitude = np.array([0.1, 0.0, -0.1], dtype=np.float32)
        delta_heading = np.array(
            [-np.pi / 6, -np.pi / 12, 0.0, np.pi / 12, np.pi / 6],
            dtype=np.float32,
        )
        delta_velocity = np.array([0.05, 0.0, -0.05], dtype=np.float32)

        # 分层高层动作含义：高度变化、航向变化、速度变化。
        high_action = high_action.astype(np.int64)
        lowlevel_obs = np.zeros(12, dtype=np.float32)
        lowlevel_obs[0] = delta_altitude[high_action[0]]
        lowlevel_obs[1] = delta_heading[high_action[1]]
        lowlevel_obs[2] = delta_velocity[high_action[2]]
        lowlevel_obs[3:12] = obs[:9]

        lowlevel_action, _, self.lowlevel_rnn_states = self.lowlevel_policy(
            np.expand_dims(lowlevel_obs, axis=0),
            self.lowlevel_rnn_states,
            self.masks,
            deterministic=True,
        )
        self.lowlevel_rnn_states = _t2n(self.lowlevel_rnn_states)
        return _t2n(lowlevel_action).squeeze(0)


def load_lowlevel_policy(device: torch.device) -> PPOActor:
    if not LOWLEVEL_ACTOR_PATH.exists():
        raise FileNotFoundError(f"找不到低层控制器: {LOWLEVEL_ACTOR_PATH}")

    obs_space = spaces.Box(low=-10, high=10.0, shape=(12,))
    act_space = spaces.MultiDiscrete([41, 41, 41, 30])
    policy = PPOActor(make_actor_args(use_prior=False), obs_space, act_space, device=device)
    policy.load_state_dict(torch.load(str(LOWLEVEL_ACTOR_PATH), map_location=device))
    policy.eval()
    return policy


def load_controller(
    name: str,
    actor_path: Path,
    scenario_name: str,
    lowlevel_policy: PPOActor,
    device: torch.device,
) -> ActorController:
    if not actor_path.exists():
        raise FileNotFoundError(f"{name} actor 文件不存在: {actor_path}")

    # 用 actor 自己的训练场景读取 observation/action space。
    actor_env = SingleCombatEnv(scenario_name)
    try:
        task_name = getattr(actor_env.config, "task", "")
        family = task_family(task_name)
        obs_dim = int(actor_env.observation_space.shape[0])
        actor_action_dim = action_dim(actor_env.action_space)
        actor = PPOActor(
            make_actor_args(use_prior=True),
            actor_env.observation_space,
            actor_env.action_space,
            device=device,
        )
        actor.load_state_dict(torch.load(str(actor_path), map_location=device))
        actor.eval()
    finally:
        actor_env.close()

    controller = ActorController(
        name=name,
        path=actor_path,
        scenario_name=scenario_name,
        task_name=task_name,
        family=family,
        is_hierarchical=task_name.startswith("hierarchical_"),
        is_shoot="shoot" in task_name,
        obs_dim=obs_dim,
        actor_action_dim=actor_action_dim,
        actor=actor,
        lowlevel_policy=lowlevel_policy,
        device=device,
    )
    controller.reset()
    return controller


def validate_setup(env: SingleCombatEnv, controllers: Iterable[ActorController]) -> None:
    eval_task = getattr(env.config, "task", "")
    if eval_task not in DIRECT_EVAL_TASKS:
        raise ValueError(
            f"EVAL_SCENARIO_NAME 必须是直接控制版 SingleCombat 场景，当前 task={eval_task}。"
        )
    if env.num_agents != 2:
        raise ValueError(
            "评估场景必须有两个 RL agent；请不要使用 use_baseline: true 的 vsBaseline 场景。"
        )

    eval_family = task_family(eval_task)
    eval_obs_dim = int(env.observation_space.shape[0])
    eval_act_dim = action_dim(env.action_space)

    for controller in controllers:
        if controller.family != eval_family:
            raise ValueError(
                f"{controller.name} 的训练任务族是 {controller.family}，"
                f"但评估场景任务族是 {eval_family}。"
            )

        # 分层 actor 会被转换成直接底层动作，因此最终动作维度必须匹配评估环境。
        expected_dim = 5 if controller.is_shoot else 4
        if expected_dim != eval_act_dim:
            raise ValueError(
                f"{controller.name} 转换后的动作维度为 {expected_dim}，"
                f"但评估环境需要 {eval_act_dim}。"
            )

        # 同一任务族的直接/分层 actor 应共享观测维度。
        if controller.obs_dim != eval_obs_dim:
            raise ValueError(
                f"{controller.name} 观测维度与评估环境不一致: "
                f"actor={controller.obs_dim}, env={eval_obs_dim}"
            )


def reset_eval_env(env: SingleCombatEnv) -> np.ndarray:
    """按脚本配置重置 1v1 环境，支持关闭随机换边或覆盖初始状态。"""
    env.current_step = 0

    # 保存一份不随 sim.reload() 改变的基准初始状态，避免随机换边后下一局状态漂移。
    if not hasattr(env, "_experiment_base_init_states"):
        base_init_states = []
        for agent_id, sim in env.agents.items():
            init_state = sim.init_state.copy()
            if CUSTOM_INITIAL_STATES and agent_id in CUSTOM_INITIAL_STATES:
                init_state.update(CUSTOM_INITIAL_STATES[agent_id])
            base_init_states.append(init_state)
        env._experiment_base_init_states = base_init_states

    env.init_states = [state.copy() for state in env._experiment_base_init_states]

    init_states = [state.copy() for state in env.init_states]
    if RANDOM_SIDE_SWAP:
        env.np_random.shuffle(init_states)

    for sim, init_state in zip(env.agents.values(), init_states):
        sim.reload(init_state)
    env._tempsims.clear()
    env.task.reset(env)
    return env._pack(env.get_obs())


def status_name(sim) -> str:
    if sim.is_alive:
        return "alive"
    if sim.is_crash:
        return "crash"
    if sim.is_shotdown:
        return "shotdown"
    return "unknown"


def missile_stats(env: SingleCombatEnv, agent_id: str) -> Dict[str, int]:
    sim = env.agents[agent_id]
    launched = len(sim.launch_missiles)
    hits = sum(1 for missile in sim.launch_missiles if missile.is_success)
    remaining_by_task = getattr(env.task, "remaining_missiles", {})
    remaining = remaining_by_task.get(agent_id, max(int(sim.num_missiles) - launched, 0))
    return {
        "missiles_launched": int(launched),
        "missile_hits": int(hits),
        "missiles_remaining": int(remaining),
    }


def decide_winner(
    actor_a_reward: float,
    actor_b_reward: float,
    actor_a_status: str,
    actor_b_status: str,
) -> str:
    actor_a_alive = actor_a_status == "alive"
    actor_b_alive = actor_b_status == "alive"
    if actor_a_alive and not actor_b_alive:
        return "actor_a"
    if actor_b_alive and not actor_a_alive:
        return "actor_b"

    reward_margin = actor_a_reward - actor_b_reward
    if reward_margin > WIN_REWARD_MARGIN:
        return "actor_a"
    if reward_margin < -WIN_REWARD_MARGIN:
        return "actor_b"
    return "tie"


def make_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    if SAVE_ACMI:
        (output_dir / "acmi").mkdir(parents=True, exist_ok=True)
    return output_dir


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def average(rows: List[Dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def build_summary(rows: List[Dict[str, object]], output_dir: Path) -> Dict[str, object]:
    total = len(rows)
    actor_a_wins = sum(row["winner"] == "actor_a" for row in rows)
    actor_b_wins = sum(row["winner"] == "actor_b" for row in rows)
    ties = sum(row["winner"] == "tie" for row in rows)

    actor_a_launched = sum(int(row["actor_a_missiles_launched"]) for row in rows)
    actor_b_launched = sum(int(row["actor_b_missiles_launched"]) for row in rows)
    actor_a_hits = sum(int(row["actor_a_missile_hits"]) for row in rows)
    actor_b_hits = sum(int(row["actor_b_missile_hits"]) for row in rows)

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "output_dir": str(output_dir),
        "eval_scenario_name": EVAL_SCENARIO_NAME,
        "actor_a_path": str(ACTOR_A_PATH),
        "actor_a_scenario_name": ACTOR_A_SCENARIO_NAME,
        "actor_b_path": str(ACTOR_B_PATH),
        "actor_b_scenario_name": ACTOR_B_SCENARIO_NAME,
        "num_episodes": total,
        "actor_a_win_rate": actor_a_wins / total if total else 0.0,
        "actor_b_win_rate": actor_b_wins / total if total else 0.0,
        "tie_rate": ties / total if total else 0.0,
        "actor_a_avg_reward": average(rows, "actor_a_reward"),
        "actor_b_avg_reward": average(rows, "actor_b_reward"),
        "avg_reward_margin": average(rows, "reward_margin"),
        "avg_steps": average(rows, "steps"),
        "avg_duration_sec": average(rows, "duration_sec"),
        "actor_a_crash_rate": sum(row["actor_a_status"] == "crash" for row in rows) / total if total else 0.0,
        "actor_b_crash_rate": sum(row["actor_b_status"] == "crash" for row in rows) / total if total else 0.0,
        "actor_a_shotdown_rate": sum(row["actor_a_status"] == "shotdown" for row in rows) / total if total else 0.0,
        "actor_b_shotdown_rate": sum(row["actor_b_status"] == "shotdown" for row in rows) / total if total else 0.0,
        "actor_a_missiles_launched": actor_a_launched,
        "actor_b_missiles_launched": actor_b_launched,
        "actor_a_missile_hits": actor_a_hits,
        "actor_b_missile_hits": actor_b_hits,
        "actor_a_missile_hit_rate": actor_a_hits / actor_a_launched if actor_a_launched else 0.0,
        "actor_b_missile_hit_rate": actor_b_hits / actor_b_launched if actor_b_launched else 0.0,
    }
    return summary


def run_episode(
    episode: int,
    env: SingleCombatEnv,
    actor_a: ActorController,
    actor_b: ActorController,
    output_dir: Path,
) -> Dict[str, object]:
    actor_a.reset()
    actor_b.reset()
    obs = reset_eval_env(env)

    # 偶数局按 A/B，奇数局交换阵营。
    if SWAP_ACTOR_ORDER and episode % 2 == 1:
        slot_controllers = [actor_b, actor_a]
    else:
        slot_controllers = [actor_a, actor_b]

    agent_ids = (env.ego_ids + env.enm_ids)[: env.num_agents]
    actor_to_slot = {controller.name: idx for idx, controller in enumerate(slot_controllers)}
    actor_to_agent_id = {
        controller.name: agent_ids[idx] for idx, controller in enumerate(slot_controllers)
    }

    actor_rewards = {"actor_a": 0.0, "actor_b": 0.0}
    acmi_path = None
    if SAVE_ACMI and episode in ACMI_EPISODES:
        acmi_path = output_dir / "acmi" / f"episode_{episode:04d}.txt.acmi"
        env._create_records = False
        env.render(mode="txt", filepath=str(acmi_path))

    while True:
        actions = [controller.act(obs[idx]) for idx, controller in enumerate(slot_controllers)]
        obs, rewards, dones, info = env.step(np.stack(actions, axis=0))

        for controller in slot_controllers:
            slot = actor_to_slot[controller.name]
            actor_rewards[controller.name] += float(rewards[slot, 0])

        if acmi_path is not None:
            env.render(mode="txt", filepath=str(acmi_path))

        if bool(np.asarray(dones).all()):
            break

    actor_a_agent_id = actor_to_agent_id["actor_a"]
    actor_b_agent_id = actor_to_agent_id["actor_b"]
    actor_a_sim = env.agents[actor_a_agent_id]
    actor_b_sim = env.agents[actor_b_agent_id]

    actor_a_status = status_name(actor_a_sim)
    actor_b_status = status_name(actor_b_sim)
    winner = decide_winner(
        actor_rewards["actor_a"],
        actor_rewards["actor_b"],
        actor_a_status,
        actor_b_status,
    )

    actor_a_missiles = missile_stats(env, actor_a_agent_id)
    actor_b_missiles = missile_stats(env, actor_b_agent_id)

    row = {
        "episode": episode,
        "winner": winner,
        "steps": int(env.current_step),
        "duration_sec": float(env.current_step * env.time_interval),
        "actor_a_agent_id": actor_a_agent_id,
        "actor_b_agent_id": actor_b_agent_id,
        "actor_a_reward": actor_rewards["actor_a"],
        "actor_b_reward": actor_rewards["actor_b"],
        "reward_margin": actor_rewards["actor_a"] - actor_rewards["actor_b"],
        "actor_a_bloods": float(actor_a_sim.bloods),
        "actor_b_bloods": float(actor_b_sim.bloods),
        "actor_a_status": actor_a_status,
        "actor_b_status": actor_b_status,
        "actor_a_missiles_launched": actor_a_missiles["missiles_launched"],
        "actor_b_missiles_launched": actor_b_missiles["missiles_launched"],
        "actor_a_missile_hits": actor_a_missiles["missile_hits"],
        "actor_b_missile_hits": actor_b_missiles["missile_hits"],
        "actor_a_missiles_remaining": actor_a_missiles["missiles_remaining"],
        "actor_b_missiles_remaining": actor_b_missiles["missiles_remaining"],
        "acmi_path": str(acmi_path) if acmi_path is not None else "",
    }
    print(
        f"[{episode + 1}/{NUM_EPISODES}] winner={winner} "
        f"A_reward={row['actor_a_reward']:.2f} B_reward={row['actor_b_reward']:.2f} "
        f"steps={row['steps']}"
    )
    return row


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = resolve_device(DEVICE)
    output_dir = make_output_dir()
    print(f"输出目录: {output_dir}")
    print(f"使用设备: {device}")

    env = SingleCombatEnv(EVAL_SCENARIO_NAME)
    env.seed(SEED)

    try:
        lowlevel_policy = load_lowlevel_policy(device)
        actor_a = load_controller(
            "actor_a",
            Path(ACTOR_A_PATH),
            ACTOR_A_SCENARIO_NAME,
            lowlevel_policy,
            device,
        )
        actor_b = load_controller(
            "actor_b",
            Path(ACTOR_B_PATH),
            ACTOR_B_SCENARIO_NAME,
            lowlevel_policy,
            device,
        )
        validate_setup(env, [actor_a, actor_b])
        rows = [
            run_episode(episode, env, actor_a, actor_b, output_dir)
            for episode in range(NUM_EPISODES)
        ]
    finally:
        env.close()

    save_csv(rows, output_dir / "episodes.csv")
    summary = build_summary(rows, output_dir)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
