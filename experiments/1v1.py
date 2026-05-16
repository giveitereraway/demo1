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

# matplotlib 只用于实验后处理，缺失时不影响对战评估。
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# 让脚本可以从 experiments/ 目录直接导入项目代码。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.ppo.ppo_actor import PPOActor
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.envs import SingleCombatEnv
from envs.JSBSim.utils.utils import get_AO_TA_R


# =========================
# 可修改实验配置
# =========================

EXPERIMENT_NAME = "selfA_vs_tacticalhierarchyselfB"

# 评估环境必须使用直接控制版 SingleCombat 场景。
EVAL_SCENARIO_NAME = "1v1/NoWeapon/Selfplay"

# 两个 actor 可以来自不同训练场景；脚本会按各自场景构造网络动作空间。
ACTOR_A_PATH = REPO_ROOT / "scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/1v1_follow/wandb/offline-run-20260512_175151-yryla8wg/files/actor_latest.pt"
ACTOR_A_SCENARIO_NAME = "1v1/NoWeapon/Selfplay"

ACTOR_B_PATH = REPO_ROOT / "scripts/results/SingleCombat/1v1/NoWeapon/TacticalHierarchySelfplay/ppo/1v1_tactical_hierarchy/wandb/latest-run/files/actor_latest.pt"
ACTOR_B_SCENARIO_NAME = "1v1/NoWeapon/TacticalHierarchySelfplay"

# 分层 actor 的低层控制器。
LOWLEVEL_ACTOR_PATH = REPO_ROOT / "envs/JSBSim/model/actor_heading.pt"

NUM_EPISODES = 50
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

# 是否在实验结束后保存图表，默认输出到结果目录下的 plots/。
SAVE_PLOTS = True
PLOT_DPI = 180
MOVING_AVERAGE_WINDOW = 5
PLOT_ONLY_RESULT_DIR = None
#PLOT_ONLY_RESULT_DIR = OUTPUT_ROOT / "selfA_vs_hierarchyselfB_20260513_104605"

# 当前 BetaShootBernoulli 内部有调试 print，射击任务中建议保持开启。
SUPPRESS_POLICY_DEBUG_PRINT = True


TASK_FAMILY = {
    "singlecombat": "NoWeapon",
    "hierarchical_singlecombat": "NoWeapon",
    "tactical_hierarchical_singlecombat": "NoWeapon",
    "singlecombat_dodge_missile": "DodgeMissile",
    "hierarchical_singlecombat_dodge_missile": "DodgeMissile",
    "tactical_hierarchical_singlecombat_dodge_missile": "DodgeMissile",
    "singlecombat_shoot": "ShootMissile",
    "hierarchical_singlecombat_shoot": "ShootMissile",
    "tactical_hierarchical_singlecombat_shoot": "ShootMissile",
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
    if isinstance(action_space, spaces.Discrete):
        return 1
    if isinstance(action_space, spaces.MultiDiscrete):
        return int(action_space.shape[0])
    if isinstance(action_space, spaces.Tuple):
        return int(sum(action_dim(space) for space in action_space.spaces))
    raise TypeError(f"不支持的动作空间: {action_space}")


@dataclass
class ActorController:
    name: str
    path: Path
    scenario_name: str
    task_name: str
    family: str
    is_hierarchical: bool
    is_tactical: bool
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
    def act(self, obs: np.ndarray, env: Optional[SingleCombatEnv] = None, agent_id: Optional[str] = None) -> np.ndarray:
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

        raw_action = np.asarray(_t2n(actor_action).squeeze(0)).reshape(-1)
        self.rnn_states = _t2n(self.rnn_states)

        if not self.is_hierarchical:
            return raw_action

        if self.is_tactical:
            if env is None or agent_id is None:
                raise ValueError(f"{self.name} 是 tactical_hierarchical actor，act() 需要 env 和 agent_id。")
            lowlevel_action = self._tactical_to_lowlevel(obs, raw_action, env, agent_id)
        else:
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
        delta_control = np.array(
            [
                delta_altitude[high_action[0]],
                delta_heading[high_action[1]],
                delta_velocity[high_action[2]],
            ],
            dtype=np.float32,
        )
        return self._delta_control_to_lowlevel(obs, delta_control)

    def _delta_control_to_lowlevel(self, obs: np.ndarray, delta_control: np.ndarray) -> np.ndarray:
        lowlevel_obs = np.zeros(12, dtype=np.float32)
        lowlevel_obs[0:3] = np.asarray(delta_control, dtype=np.float32)
        lowlevel_obs[3:12] = obs[:9]

        lowlevel_action, _, self.lowlevel_rnn_states = self.lowlevel_policy(
            np.expand_dims(lowlevel_obs, axis=0),
            self.lowlevel_rnn_states,
            self.masks,
            deterministic=True,
        )
        self.lowlevel_rnn_states = _t2n(self.lowlevel_rnn_states)
        return _t2n(lowlevel_action).squeeze(0)

    def _heading_error_to_vector(self, env: SingleCombatEnv, agent_id: str, target_xy: np.ndarray) -> float:
        ego = env.agents[agent_id]
        ego_xy = np.asarray(ego.get_velocity()[:2], dtype=np.float32)
        target_xy = np.asarray(target_xy, dtype=np.float32)

        if np.linalg.norm(ego_xy) < 1e-6:
            heading = ego.get_property_value(c.attitude_heading_true_rad)
            ego_xy = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
        if np.linalg.norm(target_xy) < 1e-6:
            return 0.0

        ego_xy = ego_xy / (np.linalg.norm(ego_xy) + 1e-8)
        target_xy = target_xy / (np.linalg.norm(target_xy) + 1e-8)
        dot = np.clip(np.dot(ego_xy, target_xy), -1.0, 1.0)
        cross_z = ego_xy[0] * target_xy[1] - ego_xy[1] * target_xy[0]
        heading_error = np.arctan2(cross_z, dot)
        return float(np.clip(heading_error, -np.pi / 6, np.pi / 6))

    def _altitude_step_to_enemy(self, env: SingleCombatEnv, agent_id: str) -> float:
        ego = env.agents[agent_id]
        enm = ego.enemies[0]
        delta_altitude = float(enm.get_position()[2] - ego.get_position()[2])
        if delta_altitude > 250:
            return 0.1
        if delta_altitude < -250:
            return -0.1
        return 0.0

    def _tactical_action_to_delta_control(
        self,
        env: SingleCombatEnv,
        agent_id: str,
        action_id: int,
    ) -> np.ndarray:
        ego = env.agents[agent_id]
        enm = ego.enemies[0]
        relative_xy = np.asarray((enm.get_position() - ego.get_position())[:2], dtype=np.float32)
        if np.linalg.norm(relative_xy) < 1e-6:
            relative_xy = np.asarray(ego.get_velocity()[:2], dtype=np.float32)

        target_xy = relative_xy
        delta_altitude = 0.0
        delta_velocity = 0.0

        # 与 TacticalHierarchicalSingleCombatTask 保持一致的 6 个战术动作编号。
        if action_id == 0:  # 追击
            target_xy = relative_xy
            delta_altitude = self._altitude_step_to_enemy(env, agent_id)
            delta_velocity = 0.05
        elif action_id == 1:  # 脱离
            target_xy = -relative_xy
            delta_velocity = 0.05
        elif action_id == 2:  # 爬升
            target_xy = relative_xy
            delta_altitude = 0.1
            delta_velocity = -0.05
        elif action_id == 3:  # 俯冲
            target_xy = relative_xy
            delta_altitude = -0.1
            delta_velocity = 0.05
        elif action_id == 4:  # 左转
            target_xy = np.array([-relative_xy[1], relative_xy[0]], dtype=np.float32)
        elif action_id == 5:  # 右转
            target_xy = np.array([relative_xy[1], -relative_xy[0]], dtype=np.float32)
        else:
            raise ValueError(f"未知 tactical action id: {action_id}")

        delta_heading = self._heading_error_to_vector(env, agent_id, target_xy)
        return np.array([delta_altitude, delta_heading, delta_velocity], dtype=np.float32)

    def _tactical_to_lowlevel(
        self,
        obs: np.ndarray,
        raw_action: np.ndarray,
        env: SingleCombatEnv,
        agent_id: str,
    ) -> np.ndarray:
        action_id = int(np.asarray(raw_action).reshape(-1)[0])
        delta_control = self._tactical_action_to_delta_control(env, agent_id, action_id)
        return self._delta_control_to_lowlevel(obs, delta_control)


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

    is_tactical = task_name.startswith("tactical_hierarchical_")
    controller = ActorController(
        name=name,
        path=actor_path,
        scenario_name=scenario_name,
        task_name=task_name,
        family=family,
        is_hierarchical=task_name.startswith("hierarchical_") or is_tactical,
        is_tactical=is_tactical,
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
    if SAVE_PLOTS:
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    return output_dir


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_summary(summary: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def load_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def average(rows: List[Dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def metric_values(rows: List[Dict[str, object]], key: str) -> List[float]:
    """读取某个指标列，兼容旧版 episodes.csv 中不存在新指标的情况。"""
    values = []
    for row in rows:
        if key not in row or row[key] in ("", None):
            continue
        values.append(float(row[key]))
    return values


def metric_mean(rows: List[Dict[str, object]], key: str) -> float:
    values = metric_values(rows, key)
    return float(np.mean(values)) if values else 0.0


def metric_std(rows: List[Dict[str, object]], key: str) -> float:
    """按回合样本计算标准差；只有 1 个样本时记为 0。"""
    values = metric_values(rows, key)
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def has_metric(rows: List[Dict[str, object]], key: str) -> bool:
    return bool(metric_values(rows, key))


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
        "actor_a_reward_std": metric_std(rows, "actor_a_reward"),
        "actor_b_reward_std": metric_std(rows, "actor_b_reward"),
        "avg_reward_margin": average(rows, "reward_margin"),
        "reward_margin_std": metric_std(rows, "reward_margin"),
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
    if has_metric(rows, "final_distance_m"):
        summary.update(
            {
                "final_distance_avg_m": metric_mean(rows, "final_distance_m"),
                "final_distance_std_m": metric_std(rows, "final_distance_m"),
                "final_distance_avg_km": metric_mean(rows, "final_distance_m") / 1000.0,
                "final_distance_std_km": metric_std(rows, "final_distance_m") / 1000.0,
            }
        )

    for prefix in ("actor_a", "actor_b"):
        metric_prefix = f"{prefix}_final"
        if has_metric(rows, f"{metric_prefix}_ao_deg"):
            summary.update(
                {
                    f"{metric_prefix}_ao_avg_deg": metric_mean(rows, f"{metric_prefix}_ao_deg"),
                    f"{metric_prefix}_ao_std_deg": metric_std(rows, f"{metric_prefix}_ao_deg"),
                    f"{metric_prefix}_ta_avg_deg": metric_mean(rows, f"{metric_prefix}_ta_deg"),
                    f"{metric_prefix}_ta_std_deg": metric_std(rows, f"{metric_prefix}_ta_deg"),
                }
            )

        metric_prefix = f"{prefix}_episode_mean"
        if has_metric(rows, f"{metric_prefix}_ao_deg"):
            summary.update(
                {
                    f"{metric_prefix}_ao_avg_deg": metric_mean(rows, f"{metric_prefix}_ao_deg"),
                    f"{metric_prefix}_ao_std_deg": metric_std(rows, f"{metric_prefix}_ao_deg"),
                    f"{metric_prefix}_ta_avg_deg": metric_mean(rows, f"{metric_prefix}_ta_deg"),
                    f"{metric_prefix}_ta_std_deg": metric_std(rows, f"{metric_prefix}_ta_deg"),
                }
            )
    return summary


def configure_plot_style() -> None:
    """配置适合中文实验图表的 matplotlib 样式。"""
    if plt is None:
        return
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.autolayout": True,
        }
    )


def numeric_series(rows: List[Dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def safe_numeric_series(rows: List[Dict[str, object]], key: str) -> Optional[np.ndarray]:
    values = metric_values(rows, key)
    if not values:
        return None
    return np.asarray(values, dtype=np.float64)


def simulator_feature(sim) -> np.ndarray:
    """提取计算相对态势所需的 3D 位置和速度。"""
    return np.hstack([sim.get_position(), sim.get_velocity()])


def relative_state_metrics(actor_a_sim, actor_b_sim) -> Dict[str, float]:
    """计算双方相对距离、AO 和 TA；角度同时从两个 actor 视角记录。"""
    actor_a_feature = simulator_feature(actor_a_sim)
    actor_b_feature = simulator_feature(actor_b_sim)

    actor_a_ao, actor_a_ta, distance_m = get_AO_TA_R(actor_a_feature, actor_b_feature)
    actor_b_ao, actor_b_ta, _ = get_AO_TA_R(actor_b_feature, actor_a_feature)
    return {
        "distance_m": float(distance_m),
        "actor_a_ao_deg": float(np.rad2deg(actor_a_ao)),
        "actor_a_ta_deg": float(np.rad2deg(actor_a_ta)),
        "actor_b_ao_deg": float(np.rad2deg(actor_b_ao)),
        "actor_b_ta_deg": float(np.rad2deg(actor_b_ta)),
    }


def append_metric_history(history: Dict[str, List[float]], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        history.setdefault(key, []).append(value)


def metric_history_mean(history: Dict[str, List[float]], key: str) -> float:
    values = history.get(key, [])
    return float(np.mean(values)) if values else 0.0


def metric_history_std(history: Dict[str, List[float]], key: str) -> float:
    values = history.get(key, [])
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def summarize_relative_history(history: Dict[str, List[float]]) -> Dict[str, float]:
    """汇总单局内的末态距离、末态 AO/TA 和整局平均 AO/TA。"""
    if not history:
        return {
            "final_distance_m": 0.0,
            "actor_a_final_ao_deg": 0.0,
            "actor_a_final_ta_deg": 0.0,
            "actor_b_final_ao_deg": 0.0,
            "actor_b_final_ta_deg": 0.0,
            "actor_a_episode_mean_ao_deg": 0.0,
            "actor_a_episode_mean_ta_deg": 0.0,
            "actor_b_episode_mean_ao_deg": 0.0,
            "actor_b_episode_mean_ta_deg": 0.0,
            "actor_a_episode_std_ao_deg": 0.0,
            "actor_a_episode_std_ta_deg": 0.0,
            "actor_b_episode_std_ao_deg": 0.0,
            "actor_b_episode_std_ta_deg": 0.0,
        }

    return {
        "final_distance_m": float(history["distance_m"][-1]),
        "actor_a_final_ao_deg": float(history["actor_a_ao_deg"][-1]),
        "actor_a_final_ta_deg": float(history["actor_a_ta_deg"][-1]),
        "actor_b_final_ao_deg": float(history["actor_b_ao_deg"][-1]),
        "actor_b_final_ta_deg": float(history["actor_b_ta_deg"][-1]),
        "actor_a_episode_mean_ao_deg": metric_history_mean(history, "actor_a_ao_deg"),
        "actor_a_episode_mean_ta_deg": metric_history_mean(history, "actor_a_ta_deg"),
        "actor_b_episode_mean_ao_deg": metric_history_mean(history, "actor_b_ao_deg"),
        "actor_b_episode_mean_ta_deg": metric_history_mean(history, "actor_b_ta_deg"),
        "actor_a_episode_std_ao_deg": metric_history_std(history, "actor_a_ao_deg"),
        "actor_a_episode_std_ta_deg": metric_history_std(history, "actor_a_ta_deg"),
        "actor_b_episode_std_ao_deg": metric_history_std(history, "actor_b_ao_deg"),
        "actor_b_episode_std_ta_deg": metric_history_std(history, "actor_b_ta_deg"),
    }


def moving_average(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """计算滑动平均，用于在回合数较多时观察整体趋势。"""
    if window <= 1 or values.size < window:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    averaged = np.convolve(values, kernel, mode="valid")
    x_offset = np.arange(window - 1, values.size, dtype=np.int64)
    return x_offset, averaged


def save_plot(fig, plot_dir: Path, filename: str) -> Path:
    path = plot_dir / filename
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def add_line_with_average(
    ax,
    episode_ids: np.ndarray,
    values: np.ndarray,
    label: str,
    color: str,
) -> None:
    ax.plot(episode_ids, values, marker="o", linewidth=1.8, label=label, color=color)
    avg_x, avg_y = moving_average(values, MOVING_AVERAGE_WINDOW)
    if avg_y.size:
        ax.plot(
            episode_ids[avg_x],
            avg_y,
            linestyle="--",
            linewidth=2.0,
            label=f"{label} {MOVING_AVERAGE_WINDOW}局滑动平均",
            color=color,
            alpha=0.75,
        )


def plot_reward_curve(rows: List[Dict[str, object]], plot_dir: Path) -> Path:
    episode_ids = numeric_series(rows, "episode")
    actor_a_rewards = numeric_series(rows, "actor_a_reward")
    actor_b_rewards = numeric_series(rows, "actor_b_reward")

    fig, ax = plt.subplots(figsize=(10, 5))
    add_line_with_average(ax, episode_ids, actor_a_rewards, "Actor A", "#1f77b4")
    add_line_with_average(ax, episode_ids, actor_b_rewards, "Actor B", "#d62728")
    ax.set_title("各回合累计奖励曲线")
    ax.set_xlabel("Episode")
    ax.set_ylabel("累计奖励")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_plot(fig, plot_dir, "reward_curve.png")


def plot_reward_margin(rows: List[Dict[str, object]], plot_dir: Path) -> Path:
    episode_ids = numeric_series(rows, "episode")
    reward_margin = numeric_series(rows, "reward_margin")

    fig, ax = plt.subplots(figsize=(10, 5))
    add_line_with_average(ax, episode_ids, reward_margin, "A-B 奖励差", "#2ca02c")
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.axhline(WIN_REWARD_MARGIN, color="#999999", linestyle=":", linewidth=1.2)
    ax.axhline(-WIN_REWARD_MARGIN, color="#999999", linestyle=":", linewidth=1.2)
    ax.fill_between(
        episode_ids,
        -WIN_REWARD_MARGIN,
        WIN_REWARD_MARGIN,
        color="#999999",
        alpha=0.08,
        label="平局奖励差区间",
    )
    ax.set_title("各回合奖励差曲线")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Actor A 奖励 - Actor B 奖励")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_plot(fig, plot_dir, "reward_margin_curve.png")


def plot_outcome_curve(rows: List[Dict[str, object]], plot_dir: Path) -> Path:
    episode_ids = numeric_series(rows, "episode")
    winners = [str(row["winner"]) for row in rows]
    episode_count = np.arange(1, len(rows) + 1, dtype=np.float64)
    actor_a_rate = np.cumsum([winner == "actor_a" for winner in winners]) / episode_count
    actor_b_rate = np.cumsum([winner == "actor_b" for winner in winners]) / episode_count
    tie_rate = np.cumsum([winner == "tie" for winner in winners]) / episode_count

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = ["Actor A胜", "Actor B胜", "平局"]
    counts = [
        winners.count("actor_a"),
        winners.count("actor_b"),
        winners.count("tie"),
    ]
    bars = axes[0].bar(labels, counts, color=["#1f77b4", "#d62728", "#7f7f7f"])
    axes[0].set_title("胜负结果统计")
    axes[0].set_ylabel("回合数")
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    axes[1].plot(episode_ids, actor_a_rate, marker="o", label="Actor A累计胜率")
    axes[1].plot(episode_ids, actor_b_rate, marker="o", label="Actor B累计胜率")
    axes[1].plot(episode_ids, tie_rate, marker="o", label="累计平局率")
    axes[1].set_title("累计胜率/平局率")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("比例")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    return save_plot(fig, plot_dir, "outcome_curve.png")


def plot_terminal_metrics(rows: List[Dict[str, object]], plot_dir: Path) -> Path:
    episode_ids = numeric_series(rows, "episode")
    steps = numeric_series(rows, "steps")
    duration_sec = numeric_series(rows, "duration_sec")
    actor_a_bloods = numeric_series(rows, "actor_a_bloods")
    actor_b_bloods = numeric_series(rows, "actor_b_bloods")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(episode_ids, steps, marker="o", label="步数", color="#9467bd")
    axes[0].plot(episode_ids, duration_sec, marker="s", label="仿真时长(s)", color="#8c564b")
    axes[0].set_title("每回合终止步数与仿真时长")
    axes[0].set_ylabel("数值")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(episode_ids, actor_a_bloods, marker="o", label="Actor A血量", color="#1f77b4")
    axes[1].plot(episode_ids, actor_b_bloods, marker="o", label="Actor B血量", color="#d62728")
    axes[1].set_title("每回合终局血量")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("血量")
    axes[1].set_ylim(-5, max(105.0, float(np.max([actor_a_bloods.max(), actor_b_bloods.max()])) + 5))
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    return save_plot(fig, plot_dir, "terminal_metrics.png")


def plot_tactical_metrics(rows: List[Dict[str, object]], plot_dir: Path) -> Optional[Path]:
    if not has_metric(rows, "final_distance_m"):
        return None

    episode_ids = numeric_series(rows, "episode")
    final_distance_km = numeric_series(rows, "final_distance_m") / 1000.0
    actor_a_mean_ao = safe_numeric_series(rows, "actor_a_episode_mean_ao_deg")
    actor_b_mean_ao = safe_numeric_series(rows, "actor_b_episode_mean_ao_deg")
    actor_a_mean_ta = safe_numeric_series(rows, "actor_a_episode_mean_ta_deg")
    actor_b_mean_ta = safe_numeric_series(rows, "actor_b_episode_mean_ta_deg")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    add_line_with_average(axes[0], episode_ids, final_distance_km, "最终距离", "#9467bd")
    axes[0].set_title("最终距离曲线")
    axes[0].set_ylabel("距离(km)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if actor_a_mean_ao is not None and actor_b_mean_ao is not None:
        add_line_with_average(axes[1], episode_ids[: actor_a_mean_ao.size], actor_a_mean_ao, "Actor A 平均 AO", "#1f77b4")
        add_line_with_average(axes[1], episode_ids[: actor_b_mean_ao.size], actor_b_mean_ao, "Actor B 平均 AO", "#d62728")
    axes[1].set_title("整局平均 AO 曲线")
    axes[1].set_ylabel("AO(deg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if actor_a_mean_ta is not None and actor_b_mean_ta is not None:
        add_line_with_average(axes[2], episode_ids[: actor_a_mean_ta.size], actor_a_mean_ta, "Actor A 平均 TA", "#1f77b4")
        add_line_with_average(axes[2], episode_ids[: actor_b_mean_ta.size], actor_b_mean_ta, "Actor B 平均 TA", "#d62728")
    axes[2].set_title("整局平均 TA 曲线")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("TA(deg)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    return save_plot(fig, plot_dir, "tactical_metrics.png")


def plot_status_counts(rows: List[Dict[str, object]], plot_dir: Path) -> Path:
    statuses = ["alive", "crash", "shotdown", "unknown"]
    status_labels = ["存活", "坠毁", "被击落", "未知"]
    actor_a_counts = [sum(row["actor_a_status"] == status for row in rows) for status in statuses]
    actor_b_counts = [sum(row["actor_b_status"] == status for row in rows) for status in statuses]
    x = np.arange(len(statuses))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, actor_a_counts, width, label="Actor A", color="#1f77b4")
    ax.bar(x + width / 2, actor_b_counts, width, label="Actor B", color="#d62728")
    ax.set_title("终局状态统计")
    ax.set_xticks(x)
    ax.set_xticklabels(status_labels)
    ax.set_ylabel("回合数")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    return save_plot(fig, plot_dir, "status_counts.png")


def plot_missile_stats(rows: List[Dict[str, object]], plot_dir: Path) -> Optional[Path]:
    missile_keys = [
        "actor_a_missiles_launched",
        "actor_b_missiles_launched",
        "actor_a_missile_hits",
        "actor_b_missile_hits",
        "actor_a_missiles_remaining",
        "actor_b_missiles_remaining",
    ]
    if not any(float(row[key]) > 0 for row in rows for key in missile_keys):
        return None

    episode_ids = numeric_series(rows, "episode")
    actor_a_launched = numeric_series(rows, "actor_a_missiles_launched")
    actor_b_launched = numeric_series(rows, "actor_b_missiles_launched")
    actor_a_hits = numeric_series(rows, "actor_a_missile_hits")
    actor_b_hits = numeric_series(rows, "actor_b_missile_hits")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width = 0.35
    axes[0].bar(episode_ids - width / 2, actor_a_launched, width, label="Actor A发射")
    axes[0].bar(episode_ids + width / 2, actor_b_launched, width, label="Actor B发射")
    axes[0].set_title("各回合导弹发射数")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("发射数")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(episode_ids - width / 2, actor_a_hits, width, label="Actor A命中")
    axes[1].bar(episode_ids + width / 2, actor_b_hits, width, label="Actor B命中")
    axes[1].set_title("各回合导弹命中数")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("命中数")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    return save_plot(fig, plot_dir, "missile_stats.png")


def plot_results(rows: List[Dict[str, object]], output_dir: Path) -> List[Path]:
    if not SAVE_PLOTS:
        return []
    if plt is None:
        print("未安装 matplotlib，跳过图表绘制。可安装 matplotlib 后重新运行。")
        return []
    if not rows:
        return []

    configure_plot_style()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = [
        plot_reward_curve(rows, plot_dir),
        plot_reward_margin(rows, plot_dir),
        plot_outcome_curve(rows, plot_dir),
        plot_terminal_metrics(rows, plot_dir),
        plot_status_counts(rows, plot_dir),
    ]
    tactical_plot = plot_tactical_metrics(rows, plot_dir)
    if tactical_plot is not None:
        plot_paths.append(tactical_plot)

    missile_plot = plot_missile_stats(rows, plot_dir)
    if missile_plot is not None:
        plot_paths.append(missile_plot)

    with (plot_dir / "plot_index.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "plot_files": [str(path) for path in plot_paths],
                "moving_average_window": MOVING_AVERAGE_WINDOW,
                "note": "reward_margin 为 Actor A 累计奖励减 Actor B 累计奖励。",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"图表目录: {plot_dir}")
    return plot_paths


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
    relative_history: Dict[str, List[float]] = {}
    acmi_path = None
    if SAVE_ACMI and episode in ACMI_EPISODES:
        acmi_path = output_dir / "acmi" / f"episode_{episode:04d}.txt.acmi"
        env._create_records = False
        env.render(mode="txt", filepath=str(acmi_path))

    while True:
        actions = [
            controller.act(obs[idx], env=env, agent_id=actor_to_agent_id[controller.name])
            for idx, controller in enumerate(slot_controllers)
        ]
        obs, rewards, dones, info = env.step(np.stack(actions, axis=0))

        for controller in slot_controllers:
            slot = actor_to_slot[controller.name]
            actor_rewards[controller.name] += float(rewards[slot, 0])

        actor_a_sim = env.agents[actor_to_agent_id["actor_a"]]
        actor_b_sim = env.agents[actor_to_agent_id["actor_b"]]
        append_metric_history(
            relative_history,
            relative_state_metrics(actor_a_sim, actor_b_sim),
        )

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
    relative_summary = summarize_relative_history(relative_history)

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
        **relative_summary,
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
        f"final_dist={row['final_distance_m'] / 1000.0:.2f}km steps={row['steps']}"
    )
    return row


def plot_existing_results(result_dir: Path) -> None:
    """只读取已有 episodes.csv 并补充生成图表。"""
    episodes_path = result_dir / "episodes.csv"
    summary_path = result_dir / "summary.json"
    if not episodes_path.exists():
        raise FileNotFoundError(f"未找到已有结果文件: {episodes_path}")

    rows = load_csv(episodes_path)
    summary = load_summary(summary_path) or build_summary(rows, result_dir)
    plot_files = plot_results(rows, result_dir)
    summary["plot_files"] = [str(path) for path in plot_files]
    save_summary(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    if PLOT_ONLY_RESULT_DIR is not None:
        plot_existing_results(Path(PLOT_ONLY_RESULT_DIR))
        return

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
    plot_files = plot_results(rows, output_dir)
    summary["plot_files"] = [str(path) for path in plot_files]
    save_summary(summary, output_dir / "summary.json")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
