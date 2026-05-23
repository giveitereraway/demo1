from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .settings import REPO_ROOT
from .tactical_actions import coerce_action_id


def make_actor_args() -> SimpleNamespace:
    """构造与本项目 PPO 训练脚本一致的 actor 网络参数。"""
    return SimpleNamespace(
        gain=0.01,
        hidden_size="128 128",
        act_hidden_size="128 128",
        activation_id=1,
        use_feature_normalization=False,
        use_recurrent_policy=True,
        recurrent_hidden_size=128,
        recurrent_hidden_layers=1,
        use_prior=False,
    )


def resolve_torch_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _ensure_tactical_space(env: Any) -> None:
    action_space = getattr(env, "action_space", None)
    if action_space is None or action_space.__class__.__name__ != "Discrete" or int(action_space.n) != 12:
        raise ValueError("actor fallback 只支持 Discrete(12) 的 TacticalHierarchySelfplay 场景。")


def resolve_actor_checkpoint_path(actor_path: str | Path) -> Path:
    path = Path(actor_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if path.is_dir():
        path = path / "actor_latest.pt"
    return path


@dataclass
class TacticalActorPolicy:
    actor_path: Path
    actor: Any
    device: Any

    def reset(self) -> None:
        # 高层 tactical actor 使用 GRU，重置环境时同步清空隐状态。
        self.rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
        self.masks = np.ones((1, 1), dtype=np.float32)

    @classmethod
    def load(cls, actor_path: str | Path, env: Any, *, device_name: str = "auto") -> "TacticalActorPolicy":
        import torch
        from algorithms.ppo.ppo_actor import PPOActor

        _ensure_tactical_space(env)
        path = resolve_actor_checkpoint_path(actor_path)
        if not path.exists():
            raise FileNotFoundError(f"actor 文件不存在: {path}")

        device = resolve_torch_device(device_name)
        actor = PPOActor(make_actor_args(), env.observation_space, env.action_space, device=device)
        state_dict = torch.load(str(path), map_location=device)
        actor.load_state_dict(state_dict)
        actor.eval()

        policy = cls(path, actor, device)
        policy.reset()
        return policy

    def act(self, obs: np.ndarray) -> int:
        import torch

        obs_batch = np.expand_dims(np.asarray(obs, dtype=np.float32), axis=0)
        with torch.no_grad():
            action, _, rnn_states = self.actor(
                obs_batch,
                self.rnn_states,
                self.masks,
                deterministic=True,
            )
        raw_action = np.asarray(action.detach().cpu().numpy()).reshape(-1)[0]
        self.rnn_states = rnn_states.detach().cpu().numpy()
        action_id = coerce_action_id(raw_action)
        if action_id is None:
            raise ValueError(f"actor 输出非法 tactical action id: {raw_action}")
        return action_id
