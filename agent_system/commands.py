from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .settings import AgentSettings, DEFAULT_TACTICAL_ACTOR_PATH, REPO_ROOT
from .tactical_actions import parse_action_reference
from .tactical_policy import resolve_actor_checkpoint_path


def _path_text(value: str | Path | None) -> str:
    """清理用户复制来的路径，兼容引号、空格以及两种斜杠。"""
    if value is None:
        return ""
    text = str(value).strip().strip('"').strip("'")
    if os.name == "nt":
        text = text.replace("/", "\\")
    return text


def _path(value: str | Path | None, *, base: Path = REPO_ROOT) -> Path | None:
    text = _path_text(value)
    if text == "":
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _required_existing_actor_checkpoint(label: str, value: str | Path | None) -> tuple[Path | None, list[str]]:
    text = _path_text(value)
    if text == "":
        return None, [f"{label} 路径不能为空。"]
    path = resolve_actor_checkpoint_path(text)
    if not path.exists():
        return path, [f"{label} 文件不存在：{path}"]
    return path, []


def _python() -> str:
    return AgentSettings.load().runtime_python


@dataclass
class CommandSpec:
    name: str
    command: list[str]
    cwd: Path = REPO_ROOT
    env: dict[str, str] = field(default_factory=dict)
    expected_output_dir: Path | None = None
    validation_errors: list[str] = field(default_factory=list)

    def preview(self) -> str:
        return subprocess.list2cmdline(self.command)

    def merged_env(self) -> dict[str, str]:
        merged = os.environ.copy()
        merged.update(self.env)
        return merged


@dataclass
class TacticalDemoConfig:
    actor_path: str = str(DEFAULT_TACTICAL_ACTOR_PATH)
    enemy_path: str = ""
    scenario_name: str = "1v1/NoWeapon/TacticalHierarchySelfplay"
    agent_id: str = "A0100"
    enemy_action: str = ""
    hold_steps: int = 10
    max_steps: int = 1000
    seed: int = 1
    device: str = "auto"
    render_mode: str = "txt"
    log_path: str = "output/agent_tactical_1v1/demo_log.jsonl"
    acmi_path: str = "output/agent_tactical_1v1/demo.txt.acmi"
    step_sleep: float = 0.2
    status_interval: int = 25
    verbose_steps: bool = False
    disable_llm: bool = False
    disable_complex_plan: bool = False
    max_plan_actions: int = 4


def build_tactical_demo_command(config: TacticalDemoConfig) -> CommandSpec:
    script = REPO_ROOT / "scripts" / "agent" / "agent_tactical_1v1_demo.py"
    actor_path, validation_errors = _required_existing_actor_checkpoint("Tactical actor", config.actor_path)

    enemy_path_value = config.enemy_path or config.actor_path
    enemy_path = resolve_actor_checkpoint_path(enemy_path_value)
    if config.enemy_action:
        if parse_action_reference(config.enemy_action) is None:
            validation_errors.append(f"enemy-action 无法识别：{config.enemy_action}")
    else:
        enemy_path, enemy_errors = _required_existing_actor_checkpoint("Enemy tactical actor", enemy_path_value)
        validation_errors.extend(enemy_errors)

    log_path = _path(config.log_path)
    acmi_path = _path(config.acmi_path)
    command = [
        _python(),
        str(script),
        "--actor-path",
        str(actor_path or ""),
        "--enemy-path",
        str(enemy_path or ""),
        "--scenario-name",
        config.scenario_name,
        "--agent-id",
        config.agent_id,
        "--hold-steps",
        str(config.hold_steps),
        "--max-steps",
        str(config.max_steps),
        "--seed",
        str(config.seed),
        "--device",
        config.device,
        "--render-mode",
        config.render_mode,
        "--log-path",
        str(log_path or ""),
        "--acmi-path",
        str(acmi_path or ""),
        "--step-sleep",
        str(config.step_sleep),
        "--status-interval",
        str(config.status_interval),
        "--max-plan-actions",
        str(config.max_plan_actions),
    ]
    if config.enemy_action:
        command.extend(["--enemy-action", config.enemy_action])
    if config.verbose_steps:
        command.append("--verbose-steps")
    if config.disable_llm:
        command.append("--disable-llm")
    if config.disable_complex_plan:
        command.append("--disable-complex-plan")
    return CommandSpec(
        "1v1 LLM-Agent 战术调度演示",
        command,
        REPO_ROOT,
        expected_output_dir=log_path.parent if log_path is not None else None,
        validation_errors=validation_errors,
    )
