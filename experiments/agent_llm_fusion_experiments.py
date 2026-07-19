#!/usr/bin/env python
"""
Agent/LLM 融合实验脚本

本脚本用于生成论文中 Agent/LLM 融合部分的实验数据和图表，包含三组实验：
1. 指令解析准确率实验：保留 12 类动作各 10 条表达，新增 30 条态势感知、24 条双动作计划和 20 条越界指令。
2. manual 接管 + actor fallback 时间轴实验：验证简单指令、双动作计划接管以及无指令时恢复 RL actor。
3. 安全覆盖实验：验证低空、近距高速、不利姿态和非法动作会被安全模块拦截。

最小运行命令：
    python experiments/agent_llm_fusion_experiments.py --experiment parse --parser-mode llm_fallback
    python experiments/agent_llm_fusion_experiments.py --experiment safety
    python experiments/agent_llm_fusion_experiments.py --experiment timeline --max-steps 200 --device auto
    python experiments/agent_llm_fusion_experiments.py --experiment all --max-steps 300 --parser-mode llm_fallback

主要参数说明：
    --experiment       选择实验：parse / timeline / safety / all，默认 all。
    --parser-mode      指令解析方式：keyword 只用关键词；llm_fallback 有 API Key 时先用 LLM，失败后关键词兜底，且每条表达重复测试 3 遍。
                       态势感知样本会额外隐藏态势摘要再运行一遍，用于配对消融对比。
    --actor-path       己方 tactical actor 路径；可传 actor_latest.pt 或包含该文件的目录。
    --enemy-path       敌方 tactical actor 路径；不传时复用 --actor-path。
    --enemy-action     敌方固定战术动作；显式传入后优先于 --enemy-path。
    --manual-schedule  时间轴实验的人工指令注入计划，默认包含一条简单指令和一条双动作复杂指令。
    --max-steps        时间轴实验最大环境步数。
    --hold-steps       每条人工指令接管的环境步数。
    --formats          图片格式，默认 png,pdf。
    --no-plots         只输出 CSV/JSON，不生成图片。

默认输出目录：
    experiments/outputs/agent_llm_fusion/<时间戳>/

主要输出文件：
    parse/instruction_parsing_results.csv
    parse/state_context_ablation_results.csv
    parse/instruction_parsing_summary.json
    timeline/timeline_steps.csv
    timeline/timeline_summary.json
    safety/safety_results.csv
    safety/safety_summary.json

推荐放入论文的图：
    parse/figures/parsing_accuracy_by_action.*
    parse/figures/parsing_confusion_matrix.*
    parse/figures/parsing_source_validity.*
    parse/figures/parsing_state_context_comparison.*
    timeline/figures/timeline_actions.*
    timeline/figures/timeline_source_ratio.*
    safety/figures/safety_reason_counts.*
    safety/figures/safety_replacement_matrix.*
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_system.llm import SiliconFlowClient
from agent_system.settings import AgentSettings, DEFAULT_TACTICAL_ACTOR_PATH
from agent_system.tactical_actions import (
    ACTION_BY_ID,
    TACTICAL_ACTIONS,
    action_chinese_name,
    action_name,
    parse_action_reference,
)
from agent_system.tactical_parser import TacticalDecision, keyword_parse_tactical_instruction, parse_tactical_instruction
from agent_system.tactical_plan_executor import PlanExecutionResult, TacticalPlanExecutor
from agent_system.tactical_planner import parse_tactical_command
from agent_system.tactical_policy import TacticalActorPolicy, resolve_actor_checkpoint_path
from agent_system.tactical_safety import TacticalSafetyState, apply_tactical_safety
from agent_system.tactical_scheduler import ScheduledTacticalAction, TacticalActionScheduler
from agent_system.tactical_state import TacticalSituation, situation_from_env


DEFAULT_MANUAL_SCHEDULE = "30:向左转弯;90:减速;150:先提前量追击接近敌机，再爬升占位"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "outputs" / "agent_llm_fusion"
FIGURE_DPI = 180
LLM_FALLBACK_REPEAT_COUNT = 3


@dataclass(frozen=True)
class InstructionCase:
    instruction: str
    expected_action_id: int | None
    note: str = ""
    category: str = "single_action"
    situation: TacticalSituation | None = None
    expected_action_ids: tuple[int, ...] = ()

    @property
    def expected_kind(self) -> str:
        if self.category == "complex_plan":
            return "plan"
        if self.category == "invalid":
            return "invalid"
        return "decision"

    @property
    def expected_actions(self) -> tuple[int, ...]:
        if self.expected_action_ids:
            return self.expected_action_ids
        if self.expected_action_id is None or self.expected_action_id < 0:
            return ()
        return (self.expected_action_id,)


@dataclass(frozen=True)
class SafetyCase:
    case_id: str
    description: str
    action_id: int | None
    fallback_action_id: int | None
    state: TacticalSafetyState


def resolve_path(value: str | Path | None, *, default: Path | None = None) -> Path | None:
    if value is None or str(value).strip() == "":
        return default
    path = Path(str(value).strip().strip('"').strip("'")).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def parse_formats(value: str) -> list[str]:
    formats = []
    for item in value.split(","):
        suffix = item.strip().lower().lstrip(".")
        if not suffix:
            continue
        if suffix not in {"png", "pdf", "svg"}:
            raise argparse.ArgumentTypeError(f"暂不支持的图片格式: {item}")
        formats.append(suffix)
    if not formats:
        raise argparse.ArgumentTypeError("至少需要指定一种图片格式。")
    return formats


def parse_manual_schedule(value: str) -> dict[int, str]:
    text = str(value or "").strip()
    if not text:
        return {}

    schedule: dict[int, str] = {}
    for raw_item in text.split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"manual-schedule 项缺少冒号: {item}")
        step_text, instruction = item.split(":", 1)
        try:
            step = int(step_text.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"manual-schedule 步数不是整数: {step_text}") from exc
        instruction = instruction.strip()
        if step < 0:
            raise argparse.ArgumentTypeError(f"manual-schedule 步数不能为负数: {step}")
        if not instruction:
            raise argparse.ArgumentTypeError(f"manual-schedule 第 {step} 步指令为空。")
        schedule[step] = instruction
    return dict(sorted(schedule.items()))


def make_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir:
        path = resolve_path(output_dir)
    else:
        path = DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    if path is None:
        raise ValueError("输出目录不能为空。")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - 只有绘图库缺失时触发
        print(f"[plots] matplotlib 不可用，自动改用纯 SVG 兜底图表: {exc}")
        return None

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.6,
        }
    )
    return plt


def save_figure(fig: Any, output_dir: Path, stem: str, formats: Sequence[str]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=FIGURE_DPI)
        saved.append(str(path))
    return saved


def svg_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        "<style>text{font-family:'Microsoft YaHei','SimHei',Arial,sans-serif;fill:#222}</style>\n"
        '<rect width="100%" height="100%" fill="white"/>\n'
        f"{body}\n</svg>\n"
    )
    path.write_text(content, encoding="utf-8")


def write_svg_bar_chart(path: Path, title: str, labels: Sequence[str], values: Sequence[float], *, color: str = "#4C72B0") -> None:
    width, height = 980, 520
    left, right, top, bottom = 70, 30, 70, 115
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max(max(values) if values else 1.0, 1.0)
    bar_gap = 8
    bar_w = max((plot_w - bar_gap * max(len(values) - 1, 0)) / max(len(values), 1), 2)
    parts = [
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-weight="600">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#444"/>',
    ]
    for tick in range(6):
        value = max_value * tick / 5
        y = top + plot_h - plot_h * value / max_value
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#D8D8D8" stroke-width="0.7"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{value:.2g}</text>')
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + index * (bar_w + bar_gap)
        bar_h = plot_h * float(value) / max_value
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="11">{float(value):.2g}</text>')
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 18}" text-anchor="end" '
            f'font-size="11" transform="rotate(-35 {x + bar_w / 2:.1f} {top + plot_h + 18})">{svg_escape(label)}</text>'
        )
    write_svg(path, width, height, "\n".join(parts))


def write_svg_matrix(path: Path, title: str, x_labels: Sequence[str], y_labels: Sequence[str], matrix: Sequence[Sequence[int]], *, color: str) -> None:
    cell = 38
    left, top = 135, 75
    width = left + cell * len(x_labels) + 35
    height = top + cell * len(y_labels) + 120
    max_value = max([max(row) for row in matrix if row] or [1])
    parts = [
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-weight="600">{svg_escape(title)}</text>',
    ]
    base_rgb = {
        "#4C72B0": (76, 114, 176),
        "#D55E00": (213, 94, 0),
    }.get(color, (76, 114, 176))
    for y_index, label in enumerate(y_labels):
        y = top + y_index * cell
        parts.append(f'<text x="{left - 8}" y="{y + cell * 0.62:.1f}" text-anchor="end" font-size="11">{svg_escape(label)}</text>')
        for x_index, value in enumerate(matrix[y_index]):
            x = left + x_index * cell
            alpha = 0.08 + 0.82 * (value / max_value if max_value else 0)
            r, g, b = base_rgb
            fill = f"rgba({r},{g},{b},{alpha:.3f})"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#FFFFFF"/>')
            if value:
                parts.append(f'<text x="{x + cell / 2}" y="{y + cell * 0.62:.1f}" text-anchor="middle" font-size="12">{value}</text>')
    for x_index, label in enumerate(x_labels):
        x = left + x_index * cell + cell / 2
        y = top + cell * len(y_labels) + 18
        parts.append(f'<text x="{x}" y="{y}" text-anchor="end" font-size="11" transform="rotate(-45 {x} {y})">{svg_escape(label)}</text>')
    write_svg(path, width, height, "\n".join(parts))


def _timeline_flag_is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _timeline_step_ranges(
    rows: Sequence[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> list[tuple[int, int]]:
    """把满足条件的时间轴环境步合并成连续区间。"""
    active_steps = [int(row["step"]) for row in rows if predicate(row)]
    if not active_steps:
        return []

    ranges: list[tuple[int, int]] = []
    start = previous = active_steps[0]
    for step in active_steps[1:]:
        if step != previous + 1:
            ranges.append((start, previous))
            start = step
        previous = step
    ranges.append((start, previous))
    return ranges


def write_svg_timeline(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    width, height = 1100, 560
    left, right, top, bottom = 145, 40, 55, 55
    plot_w = width - left - right
    plot_h = height - top - bottom
    steps = [int(row["step"]) for row in rows]
    if not steps:
        write_svg(path, width, height, '<text x="40" y="40">无时间轴数据</text>')
        return
    min_step, max_step = min(steps), max(steps)
    step_span = max(max_step - min_step, 1)

    def x_of(step: int) -> float:
        return left + plot_w * (step - min_step) / step_span

    def y_of(action_id: int) -> float:
        return top + plot_h - plot_h * action_id / 11

    parts = [
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="22" font-weight="600">人类指令接管与自主决策动作时间轴</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#FAFAFA" stroke="#444"/>',
    ]
    manual_ranges = _timeline_step_ranges(rows, lambda row: row["source"] in {"manual", "manual_plan"})
    safety_ranges = _timeline_step_ranges(rows, lambda row: _timeline_flag_is_true(row.get("safety_overridden")))
    for start, end in manual_ranges:
        x1 = x_of(start)
        x2 = x_of(end)
        parts.append(f'<rect x="{x1:.1f}" y="{top}" width="{max(x2 - x1, 1):.1f}" height="{plot_h}" fill="#F0C36D" opacity="0.28"/>')
    for start, end in safety_ranges:
        x1 = x_of(start)
        x2 = x_of(end)
        parts.append(f'<rect x="{x1:.1f}" y="{top}" width="{max(x2 - x1, 1):.1f}" height="{plot_h}" fill="#7A5195" opacity="0.18" stroke="#7A5195" stroke-width="1.2"/>')

    for action in TACTICAL_ACTIONS:
        y = y_of(action.action_id)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#E0E0E0" stroke-width="0.7"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{action.action_id}:{svg_escape(action.chinese_name)}</text>')

    actor_points = " ".join(f"{x_of(int(row['step'])):.1f},{y_of(int(row['actor_action_id'])):.1f}" for row in rows)
    final_points = " ".join(f"{x_of(int(row['step'])):.1f},{y_of(int(row['final_action_id'])):.1f}" for row in rows)
    parts.append(f'<polyline points="{actor_points}" fill="none" stroke="#4C72B0" stroke-width="2"/>')
    parts.append(f'<polyline points="{final_points}" fill="none" stroke="#C44E52" stroke-width="2.2"/>')
    parts.append(f'<text x="{width - 250}" y="{top + 20}" font-size="13" fill="#4C72B0">蓝线：actor 建议动作</text>')
    parts.append(f'<text x="{width - 250}" y="{top + 42}" font-size="13" fill="#C44E52">红线：最终执行动作</text>')
    parts.append(f'<rect x="{width - 250}" y="{top + 52}" width="14" height="10" fill="#F0C36D" opacity="0.45"/>')
    parts.append(f'<text x="{width - 230}" y="{top + 62}" font-size="13">人类指令接管</text>')
    parts.append(f'<rect x="{width - 250}" y="{top + 74}" width="14" height="10" fill="#7A5195" opacity="0.35" stroke="#7A5195"/>')
    parts.append(f'<text x="{width - 230}" y="{top + 84}" font-size="13">安全覆盖区间</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 15}" text-anchor="middle" font-size="13">环境步</text>')
    write_svg(path, width, height, "\n".join(parts))


def build_base_instruction_dataset() -> list[InstructionCase]:
    return [
        InstructionCase("直接追击敌机", 0),
        InstructionCase("咬住敌机继续追", 0),
        InstructionCase("保持纯追", 0),
        InstructionCase("按纯追击方式压向目标", 0),
        InstructionCase("采用纯追击解算目标方位", 0),
        InstructionCase("继续追敌，不要先绕机动", 0),
        InstructionCase("直接追上去压住他", 0),
        InstructionCase("别犹豫，咬住敌机", 0),
        InstructionCase("pure pursuit", 0),
        InstructionCase("use pure pursuit to close the bandit", 0),
        InstructionCase("抢占敌机前方", 1),
        InstructionCase("提前量追击", 1),
        InstructionCase("打提前量", 1),
        InstructionCase("抢到前方位置", 1),
        InstructionCase("计算提前量后切入敌机前方", 1),
        InstructionCase("采用提前追击扩大角度优势", 1),
        InstructionCase("先抢占前方攻击窗口", 1),
        InstructionCase("往敌机前方带一点", 1),
        InstructionCase("抢到前方去卡住他", 1),
        InstructionCase("lead pursuit", 1),
        InstructionCase("保持滞后", 2),
        InstructionCase("跟在后面", 2),
        InstructionCase("保持尾随", 2),
        InstructionCase("跟住后方", 2),
        InstructionCase("采用滞后机动降低过冲风险", 2),
        InstructionCase("维持尾随队形，不要贴得太急", 2),
        InstructionCase("先滞后一点再找机会", 2),
        InstructionCase("跟在后面稳住角度", 2),
        InstructionCase("就跟住后方别抢太前", 2),
        InstructionCase("lag pursuit", 2),
        InstructionCase("脱离当前交战", 3),
        InstructionCase("先撤", 3),
        InstructionCase("拉开距离", 3),
        InstructionCase("退出交战", 3),
        InstructionCase("执行脱离机动，重建态势", 3),
        InstructionCase("立即拉开距离避免近距缠斗", 3),
        InstructionCase("退出交战圈，重新组织进攻", 3),
        InstructionCase("不打了先撤一下", 3),
        InstructionCase("先撤出去，别硬顶", 3),
        InstructionCase("disengage", 3),
        InstructionCase("爬升占位", 4),
        InstructionCase("占高度", 4),
        InstructionCase("获取高度优势", 4),
        InstructionCase("向上爬升", 4),
        InstructionCase("爬升换取势能优势", 4),
        InstructionCase("高处占位后再压下来", 4),
        InstructionCase("通过爬升建立高度优势", 4),
        InstructionCase("先占高度，别在低空耗着", 4),
        InstructionCase("往上爬升拿高度", 4),
        InstructionCase("climb for position", 4),
        InstructionCase("俯冲加速", 5),
        InstructionCase("向下俯冲", 5),
        InstructionCase("下压恢复速度", 5),
        InstructionCase("加速俯冲恢复能量", 5),
        InstructionCase("下压机头转化高度为速度", 5),
        InstructionCase("执行俯冲机动快速增速", 5),
        InstructionCase("往下压一点，把速度找回来", 5),
        InstructionCase("俯冲下去追上他", 5),
        InstructionCase("dive", 5),
        InstructionCase("dive to regain energy", 5),
        InstructionCase("平飞加速", 6),
        InstructionCase("保持平飞加速", 6),
        InstructionCase("继续加速", 6),
        InstructionCase("平飞加速并保持航向稳定", 6),
        InstructionCase("保持高度不变，平飞加速", 6),
        InstructionCase("加速拉高空速，暂不改变高度", 6),
        InstructionCase("油门加上去，继续加速", 6),
        InstructionCase("先平飞加速追一下", 6),
        InstructionCase("level accelerate", 6),
        InstructionCase("level accelerate and keep heading", 6),
        InstructionCase("平飞减速", 7),
        InstructionCase("减速避免冲过头", 7),
        InstructionCase("降低速度", 7),
        InstructionCase("避免冲过头", 7),
        InstructionCase("平飞减速控制接近率", 7),
        InstructionCase("降低速度，保持射击窗口", 7),
        InstructionCase("减速稳住相对位置", 7),
        InstructionCase("减速慢一点，别一下子冲过去", 7),
        InstructionCase("收一点速度，避免冲过头", 7),
        InstructionCase("level decelerate", 7),
        InstructionCase("向左防御转弯", 8),
        InstructionCase("左防御", 8),
        InstructionCase("左转防御", 8),
        InstructionCase("执行左转弯防御机动", 8),
        InstructionCase("向左防御，破坏敌机瞄准", 8),
        InstructionCase("左转弯拉开敌机攻击线", 8),
        InstructionCase("往左打左防御转弯", 8),
        InstructionCase("左边躲一下，做左防御", 8),
        InstructionCase("turn left", 8),
        InstructionCase("turn left defensively", 8),
        InstructionCase("向右防御转弯", 9),
        InstructionCase("右防御", 9),
        InstructionCase("右转防御", 9),
        InstructionCase("执行右转弯防御机动", 9),
        InstructionCase("向右防御，破坏敌机瞄准", 9),
        InstructionCase("右转弯拉开敌机攻击线", 9),
        InstructionCase("往右打右防御转弯", 9),
        InstructionCase("右边躲一下，做右防御", 9),
        InstructionCase("turn right", 9),
        InstructionCase("turn right defensively", 9),
        InstructionCase("高悠悠", 10),
        InstructionCase("执行高 yo", 10),
        InstructionCase("high yoyo", 10),
        InstructionCase("做高 yo-yo", 10),
        InstructionCase("采用高悠悠换高度和角度", 10),
        InstructionCase("做高yo机动控制过冲", 10),
        InstructionCase("高 yo 一下，先换高度", 10),
        InstructionCase("拉一个高悠悠再压回来", 10),
        InstructionCase("来个高 yo-yo 控住角度", 10),
        InstructionCase("high yo-yo", 10),
        InstructionCase("低悠悠", 11),
        InstructionCase("执行低 yo", 11),
        InstructionCase("low yoyo", 11),
        InstructionCase("做低 yo-yo", 11),
        InstructionCase("采用低悠悠先换速度", 11),
        InstructionCase("做低yo机动提升转弯能量", 11),
        InstructionCase("低 yo 一下，先把速度带起来", 11),
        InstructionCase("压一个低悠悠再切回来", 11),
        InstructionCase("来个低 yo-yo 抢回能量", 11),
        InstructionCase("low yo-yo", 11),
    ]


def build_state_aware_instruction_dataset() -> list[InstructionCase]:
    """构造严格配对的态势感知样本：同一表达分别置于三种单一风险态势。"""

    prompts = [
        "根据当前主要风险选择最合适的保命动作",
        "读取态势摘要并优先解除当前风险",
        "不要套用固定口令，按眼下风险采取行动",
        "结合当前高度、距离和角度做安全处置",
        "根据系统标出的风险选择对应战术动作",
        "先处理当前最紧迫的危险",
        "看一下现在的态势，执行最必要的保护机动",
        "按当前风险标签做出战术调整",
        "依据实时态势选择风险解除动作",
        "Analyze the current situation and mitigate the active risk",
    ]
    situations = [
        (
            "低空",
            4,
            TacticalSituation(
                altitude_m=3000.0,
                altitude_limit_m=2500.0,
                speed_mps=220.0,
                distance_m=2500.0,
                ao_rad=1.0,
                ta_rad=1.2,
            ),
        ),
        (
            "近距高速",
            7,
            TacticalSituation(
                altitude_m=6000.0,
                altitude_limit_m=2500.0,
                speed_mps=300.0,
                distance_m=900.0,
                ao_rad=1.0,
                ta_rad=1.2,
            ),
        ),
        (
            "不利姿态",
            3,
            TacticalSituation(
                altitude_m=6000.0,
                altitude_limit_m=2500.0,
                speed_mps=220.0,
                distance_m=2500.0,
                ao_rad=2.8,
                ta_rad=0.3,
            ),
        ),
    ]
    return [
        InstructionCase(
            instruction=prompt,
            expected_action_id=action_id,
            note=f"配对态势消融样本：{risk_name}",
            category="state_aware",
            situation=situation,
        )
        for prompt in prompts
        for risk_name, action_id, situation in situations
    ]


def build_complex_instruction_dataset() -> list[InstructionCase]:
    """构造 24 条双动作复杂指令，并让 12 类动作均获得充分覆盖。"""

    raw_cases = [
        ("先纯追击咬住敌机，再向左防御转弯", (0, 8)),
        ("先提前量追击接近敌机，再爬升占位", (1, 4)),
        ("先保持滞后稳住角度，再做低悠悠", (2, 11)),
        ("先脱离拉开距离，再平飞加速", (3, 6)),
        ("先爬升占位，再提前量追击", (4, 1)),
        ("先俯冲加速恢复能量，再做高悠悠", (5, 10)),
        ("先平飞加速追上去，再平飞减速", (6, 7)),
        ("先平飞减速控制接近率，再保持滞后", (7, 2)),
        ("先向左防御转弯，再脱离交战", (8, 3)),
        ("先向右防御转弯，再纯追击", (9, 0)),
        ("先做高悠悠，再俯冲加速", (10, 5)),
        ("先做低悠悠，再向右防御转弯", (11, 9)),
        ("先采用纯追击压向目标，然后爬升占位", (0, 4)),
        ("先打提前量抢占前方，然后平飞减速", (1, 7)),
        ("先保持尾随，接着脱离当前交战", (2, 3)),
        ("先撤出交战圈，然后执行高悠悠", (3, 10)),
        ("先获取高度优势，然后保持平飞加速", (4, 6)),
        ("先向下俯冲恢复速度，然后保持滞后", (5, 2)),
        ("先平飞加速，随后向右防御转弯", (6, 9)),
        ("先降低速度，随后向左防御转弯", (7, 8)),
        ("先左转防御，接着做低 yo-yo", (8, 11)),
        ("先右转防御，接着俯冲加速", (9, 5)),
        ("high yo-yo then lead pursuit", (10, 1)),
        ("low yo-yo then pure pursuit", (11, 0)),
    ]
    return [
        InstructionCase(
            instruction=text,
            expected_action_id=None,
            note="双动作有限战术计划",
            category="complex_plan",
            expected_action_ids=actions,
        )
        for text, actions in raw_cases
    ]


def build_invalid_instruction_dataset() -> list[InstructionCase]:
    """构造系统能力边界与非战术请求，共 20 条。"""

    instructions = [
        "帮我写一段论文摘要",
        "请给出一个创意名称",
        "飞机的发展历史是什么",
        "解释一下这段 Python 代码",
        "查询明天的天气",
        "把油门直接设置为百分之八十",
        "将升降舵调整到十五度",
        "把航向角改成正北方向",
        "保持速度精确为三百米每秒",
        "关闭飞机发动机",
        "发射导弹攻击目标",
        "先搜索目标再发射导弹",
        "组织两架友机协同夹击",
        "切换成另一个强化学习模型",
        "执行导弹规避并释放干扰弹",
        "打开文件并保存飞行记录",
        "调用知识库查询敌机型号",
        "规划一条跨区域长程航线",
        "实施毁灭性打击",
        "生成一份训练总结报告",
    ]
    return [
        InstructionCase(
            instruction=text,
            expected_action_id=-1,
            note="越界或非战术指令",
            category="invalid",
        )
        for text in instructions
    ]


def build_instruction_dataset() -> list[InstructionCase]:
    return (
        build_base_instruction_dataset()
        + build_state_aware_instruction_dataset()
        + build_complex_instruction_dataset()
        + build_invalid_instruction_dataset()
    )


def make_llm_client(parser_mode: str) -> SiliconFlowClient | None:
    if parser_mode != "llm_fallback":
        return None
    settings = AgentSettings.load()
    return SiliconFlowClient(settings) if settings.has_llm_credentials else None


PARSING_SOURCE_DISPLAY_NAMES = {
    "llm": "LLM解析",
    "llm_plan": "LLM解析",
    "keyword": "关键词兜底",
    "keyword_state": "关键词兜底",
    "keyword_plan": "关键词兜底",
}


def parsing_source_display(source: str) -> str:
    return PARSING_SOURCE_DISPLAY_NAMES.get(source, source)


def _parse_instruction_case(
    case: InstructionCase,
    *,
    parser_mode: str,
    client: SiliconFlowClient | None,
    agent_id: str,
    include_state_context: bool,
) -> tuple[str, tuple[int, ...], bool, str, str]:
    situation = case.situation if include_state_context else None
    if case.category == "complex_plan":
        command = parse_tactical_command(
            case.instruction,
            client=client,
            agent_id=agent_id,
            situation=situation,
            max_plan_actions=2,
        )
        if command.kind == "plan" and command.plan is not None:
            return "plan", tuple(step.action_id for step in command.plan.steps), True, command.plan.source, command.reason
        if command.kind == "decision" and command.decision is not None:
            decision = command.decision
            actions = (decision.action_id,) if decision.valid and decision.action_id is not None else ()
            return command.kind, actions, decision.valid, decision.source, decision.reason
        source = "llm" if client is not None else "keyword"
        return "invalid", (), False, source, command.reason

    if parser_mode == "keyword":
        decision = keyword_parse_tactical_instruction(case.instruction, agent_id=agent_id, situation=situation)
    else:
        decision = parse_tactical_instruction(case.instruction, client=client, agent_id=agent_id, situation=situation)
    actions = (decision.action_id,) if decision.valid and decision.action_id is not None else ()
    kind = "decision" if decision.valid else "invalid"
    return kind, actions, decision.valid, decision.source, decision.reason


def evaluate_instruction_cases(
    cases: Sequence[InstructionCase],
    *,
    parser_mode: str,
    agent_id: str,
    include_state_context: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    client = make_llm_client(parser_mode)
    llm_client_available = client is not None
    llm_attempted = parser_mode == "llm_fallback" and llm_client_available
    repeat_count = LLM_FALLBACK_REPEAT_COUNT if parser_mode == "llm_fallback" else 1
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        for repeat_index in range(repeat_count):
            predicted_kind, predicted_actions, valid, source, reason = _parse_instruction_case(
                case,
                parser_mode=parser_mode,
                client=client,
                agent_id=agent_id,
                include_state_context=include_state_context,
            )
            expected_actions = case.expected_actions
            correct = predicted_kind == case.expected_kind and predicted_actions == expected_actions
            predicted_action_id: int | str = predicted_actions[0] if len(predicted_actions) == 1 else (-1 if not valid else "")
            situation = case.situation if include_state_context else None
            rows.append(
                {
                    "index": len(rows),
                    "case_index": case_index,
                    "repeat_index": repeat_index + 1,
                    "repeat_count": repeat_count,
                    "llm_client_available": llm_client_available,
                    "llm_attempted": llm_attempted,
                    "category": case.category,
                    "instruction": case.instruction,
                    "state_context_enabled": include_state_context,
                    "situation": json.dumps(situation.to_log(), ensure_ascii=False) if situation is not None else "",
                    "expected_kind": case.expected_kind,
                    "predicted_kind": predicted_kind,
                    "expected_action_id": case.expected_action_id if case.expected_action_id is not None else "",
                    "expected_action_ids": ",".join(str(item) for item in expected_actions),
                    "expected_action_name": action_name(case.expected_action_id) if case.expected_action_id in ACTION_BY_ID else ("INVALID" if case.expected_kind == "invalid" else "PLAN"),
                    "expected_action_cn": action_chinese_name(case.expected_action_id) if case.expected_action_id in ACTION_BY_ID else ("无效指令" if case.expected_kind == "invalid" else "双动作计划"),
                    "predicted_action_id": predicted_action_id,
                    "predicted_action_ids": ",".join(str(item) for item in predicted_actions),
                    "predicted_action_name": action_name(predicted_action_id) if predicted_action_id in ACTION_BY_ID else ("INVALID" if not valid else "PLAN"),
                    "predicted_action_cn": action_chinese_name(predicted_action_id) if predicted_action_id in ACTION_BY_ID else ("无效" if not valid else "双动作计划"),
                    "valid": valid,
                    "source": source,
                    "source_display": parsing_source_display(source),
                    "correct": correct,
                    "reason": reason,
                    "note": case.note,
                }
            )

    total = len(rows)
    valid_count = sum(1 for row in rows if row["valid"])
    correct_count = sum(1 for row in rows if row["correct"])
    source_counts = Counter(str(row["source"]) for row in rows)
    source_display_counts = Counter(str(row["source_display"]) for row in rows)
    llm_attempted_count = sum(1 for row in rows if row["llm_attempted"])
    llm_success_count = sum(count for source, count in source_counts.items() if source.startswith("llm"))
    keyword_fallback_count = (
        sum(1 for row in rows if row["llm_attempted"] and not str(row["source"]).startswith("llm"))
        if parser_mode == "llm_fallback"
        else 0
    )
    per_action = []
    for action in TACTICAL_ACTIONS:
        action_rows = [
            row
            for row in rows
            if row["category"] == "single_action" and row["expected_action_id"] == action.action_id
        ]
        if not action_rows:
            continue
        per_action.append(
            {
                "action_id": action.action_id,
                "action_name": action.code,
                "action_cn": action.chinese_name,
                "total": len(action_rows),
                "correct": sum(1 for row in action_rows if row["correct"]),
                "accuracy": sum(1 for row in action_rows if row["correct"]) / len(action_rows),
            }
        )

    category_summaries: dict[str, dict[str, Any]] = {}
    for category in ("single_action", "state_aware", "complex_plan", "invalid"):
        category_rows = [row for row in rows if row["category"] == category]
        if not category_rows:
            continue
        category_correct = sum(1 for row in category_rows if row["correct"])
        category_summaries[category] = {
            "total": len(category_rows),
            "correct": category_correct,
            "accuracy": category_correct / len(category_rows),
        }

    summary = {
        "experiment": "instruction_parsing",
        "parser_mode": parser_mode,
        "state_context_enabled": include_state_context,
        "base_case_total": len(cases),
        "base_action_case_total": sum(1 for case in cases if case.category == "single_action"),
        "repeat_count": repeat_count,
        "llm_client_available": llm_client_available,
        "llm_attempted_count": llm_attempted_count,
        "llm_success_count": llm_success_count,
        "keyword_fallback_count": keyword_fallback_count,
        "total": total,
        "valid_count": valid_count,
        "correct_count": correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "invalid_rate": (total - valid_count) / total if total else 0.0,
        "source_counts": dict(source_counts),
        "source_display_counts": dict(source_display_counts),
        "category_summaries": category_summaries,
        "per_action": per_action,
    }
    return rows, summary


def build_parsing_confusion_matrix(rows: Sequence[dict[str, Any]]) -> tuple[list[int], list[str], list[list[int]]]:
    class_ids = [action.action_id for action in TACTICAL_ACTIONS] + [-1]
    labels = [str(action.action_id) for action in TACTICAL_ACTIONS] + ["无效"]
    index_by_id = {action_id: index for index, action_id in enumerate(class_ids)}
    invalid_index = index_by_id[-1]
    matrix = [[0 for _ in class_ids] for _ in class_ids]

    for row in rows:
        if row.get("category") not in {"single_action", "invalid"}:
            continue
        expected = int(row["expected_action_id"])
        predicted = int(row["predicted_action_id"])
        expected_index = index_by_id.get(expected, invalid_index)
        predicted_index = index_by_id.get(predicted, invalid_index)
        matrix[expected_index][predicted_index] += 1

    return class_ids, labels, matrix


def plot_instruction_results(rows: Sequence[dict[str, Any]], summary: dict[str, Any], output_dir: Path, formats: Sequence[str]) -> None:
    plt = load_matplotlib()
    fig_dir = output_dir / "figures"
    if plt is None:
        per_action = summary["per_action"]
        labels = [f"{item['action_id']}:{item['action_cn']}" for item in per_action]
        values = [float(item["accuracy"]) for item in per_action]
        write_svg_bar_chart(fig_dir / "parsing_accuracy_by_action.svg", "12 类战术指令解析准确率", labels, values)

        _, confusion_labels, matrix = build_parsing_confusion_matrix(rows)
        write_svg_matrix(
            fig_dir / "parsing_confusion_matrix.svg",
            "战术指令解析混淆矩阵（含无效指令）",
            confusion_labels,
            confusion_labels,
            matrix,
            color="#4C72B0",
        )

        source_counts = summary["source_display_counts"]
        labels = list(source_counts.keys()) + ["有效", "无效"]
        values = list(source_counts.values()) + [summary["valid_count"], summary["total"] - summary["valid_count"]]
        write_svg_bar_chart(fig_dir / "parsing_source_validity.svg", "解析来源与有效性统计", labels, values, color="#55A868")
        comparison = summary.get("state_context_comparison", {})
        if comparison:
            write_svg_bar_chart(
                fig_dir / "parsing_state_context_comparison.svg",
                "有/无态势摘要准确率对比",
                ["有态势摘要", "无态势摘要"],
                [float(comparison["with_context_accuracy"]), float(comparison["without_context_accuracy"])],
                color="#4C72B0",
            )
        return

    per_action = summary["per_action"]
    labels = [f"{item['action_id']}\n{item['action_cn']}" for item in per_action]
    values = [float(item["accuracy"]) for item in per_action]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("解析准确率")
    ax.set_title("12 类战术指令解析准确率")
    ax.grid(axis="y", alpha=0.5)
    save_figure(fig, fig_dir, "parsing_accuracy_by_action", formats)
    plt.close(fig)

    _, confusion_labels, matrix = build_parsing_confusion_matrix(rows)

    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    image = ax.imshow(matrix, cmap="Blues")
    tick_positions = list(range(len(confusion_labels)))
    ax.set_xticks(tick_positions, confusion_labels)
    ax.set_yticks(tick_positions, confusion_labels)
    ax.set_xlabel("预测动作编号")
    ax.set_ylabel("期望动作编号")
    ax.set_title("战术指令解析混淆矩阵（含无效指令）")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if value:
                ax.text(x, y, str(value), ha="center", va="center", color="#222222")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, fig_dir, "parsing_confusion_matrix", formats)
    plt.close(fig)

    source_counts = summary["source_display_counts"]
    valid_invalid = {"valid": summary["valid_count"], "invalid": summary["total"] - summary["valid_count"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(list(source_counts.keys()), list(source_counts.values()), color="#55A868")
    axes[0].set_title("解析来源统计")
    axes[0].set_ylabel("样本数")
    axes[1].bar(["有效", "无效"], [valid_invalid["valid"], valid_invalid["invalid"]], color=["#4C72B0", "#C44E52"])
    axes[1].set_title("有效/无效输出统计")
    save_figure(fig, fig_dir, "parsing_source_validity", formats)
    plt.close(fig)

    comparison = summary.get("state_context_comparison", {})
    if comparison:
        labels = ["有态势摘要", "无态势摘要"]
        values = [float(comparison["with_context_accuracy"]), float(comparison["without_context_accuracy"])]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        bars = ax.bar(labels, values, color=["#4C72B0", "#9E9E9E"], width=0.58)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("配对态势样本准确率")
        ax.set_title("有/无态势摘要准确率对比")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}", ha="center", va="bottom")
        save_figure(fig, fig_dir, "parsing_state_context_comparison", formats)
        plt.close(fig)


def run_parse_experiment(args: argparse.Namespace, output_root: Path) -> None:
    output_dir = output_root / "parse"
    cases = build_instruction_dataset()
    rows, summary = evaluate_instruction_cases(cases, parser_mode=args.parser_mode, agent_id=args.agent_id)
    state_cases = build_state_aware_instruction_dataset()
    no_context_rows, no_context_summary = evaluate_instruction_cases(
        state_cases,
        parser_mode=args.parser_mode,
        agent_id=args.agent_id,
        include_state_context=False,
    )
    with_context_rows = [dict(row, ablation_mode="with_context") for row in rows if row["category"] == "state_aware"]
    no_context_rows = [dict(row, ablation_mode="without_context") for row in no_context_rows]
    with_context_summary = summary["category_summaries"]["state_aware"]
    without_context_summary = no_context_summary["category_summaries"]["state_aware"]
    summary["state_context_comparison"] = {
        "paired_case_total": len(state_cases),
        "with_context_total": with_context_summary["total"],
        "with_context_correct": with_context_summary["correct"],
        "with_context_accuracy": with_context_summary["accuracy"],
        "without_context_total": without_context_summary["total"],
        "without_context_correct": without_context_summary["correct"],
        "without_context_accuracy": without_context_summary["accuracy"],
    }
    write_csv(output_dir / "instruction_parsing_results.csv", rows)
    write_csv(output_dir / "state_context_ablation_results.csv", with_context_rows + no_context_rows)
    write_json(output_dir / "instruction_parsing_summary.json", summary)
    if not args.no_plots:
        plot_instruction_results(rows, summary, output_dir, args.formats)
    if args.parser_mode == "llm_fallback":
        if not summary["llm_client_available"]:
            print("[parse] 警告: 未读取到 SILICONFLOW_API_KEY，本次全部使用关键词解析。")
        elif summary["llm_success_count"] == 0:
            print("[parse] 警告: 已尝试调用 LLM，但没有样本由 LLM 成功解析，请查看 CSV 的 reason 字段。")
        elif summary["keyword_fallback_count"] > 0:
            print(f"[parse] 提示: {summary['keyword_fallback_count']} 条样本 LLM 失败后使用关键词兜底。")
    print(f"[parse] 输出目录: {output_dir}")


def resolve_agent_index(env: Any, agent_id: str) -> int:
    ordered_ids = (env.ego_ids + env.enm_ids)[: env.num_agents]
    if agent_id not in ordered_ids:
        raise ValueError(f"agent-id {agent_id} 不在当前环境智能体列表中: {ordered_ids}")
    return ordered_ids.index(agent_id)


def resolve_opponent_index(env: Any, agent_id: str) -> int:
    ordered_ids = (env.ego_ids + env.enm_ids)[: env.num_agents]
    opponent_indices = [index for index, current_agent_id in enumerate(ordered_ids) if current_agent_id != agent_id]
    if len(opponent_indices) != 1:
        raise ValueError(f"当前实验只支持 1v1，无法唯一确定敌机: {ordered_ids}")
    return opponent_indices[0]


def build_action_array(env: Any, *, agent_id: str, own_action_id: int, enemy_action_id: int):
    import numpy as np

    ordered_ids = (env.ego_ids + env.enm_ids)[: env.num_agents]
    return np.asarray(
        [own_action_id if current_agent_id == agent_id else enemy_action_id for current_agent_id in ordered_ids],
        dtype=np.int64,
    )


def timeline_source_display_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {"自主决策": 0, "人类指令": 0}
    for row in rows:
        if row.get("source") in {"manual", "manual_plan"}:
            counts["人类指令"] += 1
        else:
            counts["自主决策"] += 1
    return counts


def run_timeline_experiment(args: argparse.Namespace, output_root: Path) -> None:
    import numpy as np
    from envs.JSBSim.envs import SingleCombatEnv

    output_dir = output_root / "timeline"
    schedule = parse_manual_schedule(args.manual_schedule)
    fixed_enemy_action_id: int | None = None
    if args.enemy_action:
        fixed_enemy_action_id = parse_action_reference(args.enemy_action)
        if fixed_enemy_action_id is None:
            raise ValueError(f"无法识别 enemy-action: {args.enemy_action}")

    actor_path = resolve_actor_checkpoint_path(args.actor_path)
    enemy_path = resolve_actor_checkpoint_path(args.enemy_path or args.actor_path)
    client = make_llm_client(args.parser_mode)

    env = SingleCombatEnv(args.scenario_name)
    env.seed(args.seed)
    policy = TacticalActorPolicy.load(actor_path, env, device_name=args.device)
    enemy_policy = None if fixed_enemy_action_id is not None else TacticalActorPolicy.load(enemy_path, env, device_name=args.device)
    scheduler = TacticalActionScheduler(hold_steps=args.hold_steps)
    plan_executor = TacticalPlanExecutor()

    obs = env.reset()
    policy.reset()
    if enemy_policy is not None:
        enemy_policy.reset()
    agent_index = resolve_agent_index(env, args.agent_id)
    enemy_index = resolve_opponent_index(env, args.agent_id)
    dones = np.zeros((env.num_agents, 1), dtype=bool)
    last_safe_action_id = 0
    rows: list[dict[str, Any]] = []

    try:
        while env.current_step < args.max_steps and not bool(np.asarray(dones).all()):
            decision_step = int(env.current_step)
            situation = situation_from_env(env, args.agent_id)
            raw_instruction = schedule.get(decision_step, "")
            manual_decision: TacticalDecision | None = None
            command_kind = "none"
            command_valid: bool | str = ""
            parsed_plan_action_ids = ""
            if raw_instruction:
                command = parse_tactical_command(
                    raw_instruction,
                    client=client,
                    agent_id=args.agent_id,
                    situation=situation,
                    max_plan_actions=2,
                    default_min_steps=3,
                    default_max_steps=args.hold_steps,
                )
                command_kind = command.kind
                command_valid = command.valid
                if command.kind == "plan" and command.plan is not None:
                    scheduler.clear_manual()
                    plan_executor.start(command.plan)
                    parsed_plan_action_ids = ",".join(str(step.action_id) for step in command.plan.steps)
                elif command.kind == "decision" and command.decision is not None and command.decision.valid:
                    plan_executor.clear()
                    manual_decision = command.decision

            actor_action_id = policy.act(obs[agent_index])
            if fixed_enemy_action_id is None:
                if enemy_policy is None:
                    raise RuntimeError("enemy_policy 未初始化。")
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
            reward_values = np.asarray(rewards).reshape(-1).astype(float).tolist()
            done_values = np.asarray(dones).reshape(-1).astype(bool).tolist()

            rows.append(
                {
                    "step": int(env.current_step),
                    "decision_step": decision_step,
                    "source": scheduled.source,
                    "instruction": raw_instruction,
                    "command_kind": command_kind,
                    "command_valid": command_valid,
                    "manual_valid": manual_decision.valid if manual_decision else "",
                    "manual_action_id": manual_decision.action_id if manual_decision and manual_decision.valid else "",
                    "parsed_plan_action_ids": parsed_plan_action_ids,
                    "actor_action_id": actor_action_id,
                    "actor_action_cn": action_chinese_name(actor_action_id),
                    "scheduled_action_id": scheduled.action_id,
                    "final_action_id": safety.action_id,
                    "final_action_cn": action_chinese_name(safety.action_id),
                    "remaining_manual_steps": scheduled.remaining_manual_steps,
                    "plan_active": plan_executor.active,
                    "plan_id": plan_result.plan_id if plan_result is not None else "",
                    "plan_step_index": plan_result.plan_step_index if plan_result is not None else 0,
                    "plan_total_steps": plan_result.plan_total_steps if plan_result is not None else 0,
                    "plan_until": plan_result.plan_until if plan_result is not None else "",
                    "plan_status": plan_result.plan_status if plan_result is not None else plan_executor.last_status,
                    "situation": json.dumps(situation.to_log(), ensure_ascii=False),
                    "altitude_m": situation.altitude_m if situation.altitude_m is not None else "",
                    "speed_mps": situation.speed_mps if situation.speed_mps is not None else "",
                    "delta_altitude_m": situation.delta_altitude_m if situation.delta_altitude_m is not None else "",
                    "distance_m": situation.distance_m if situation.distance_m is not None else "",
                    "ao_rad": situation.ao_rad if situation.ao_rad is not None else "",
                    "ta_rad": situation.ta_rad if situation.ta_rad is not None else "",
                    "low_altitude_risk": situation.low_altitude_risk,
                    "close_fast_risk": situation.close_fast_risk,
                    "bad_posture_risk": situation.bad_posture_risk,
                    "safety_overridden": safety.overridden,
                    "safety_reason": safety.reason,
                    "enemy_source": enemy_source,
                    "enemy_action_id": enemy_action_id,
                    "enemy_action_cn": action_chinese_name(enemy_action_id),
                    "reward_agent": reward_values[agent_index] if agent_index < len(reward_values) else "",
                    "reward_enemy": reward_values[enemy_index] if enemy_index < len(reward_values) else "",
                    "done_agent": done_values[agent_index] if agent_index < len(done_values) else "",
                    "done_enemy": done_values[enemy_index] if enemy_index < len(done_values) else "",
                }
            )
    finally:
        env.close()

    source_counts = Counter(str(row["source"]) for row in rows)
    source_display_counts = timeline_source_display_counts(rows)
    source_display_ratios = {
        label: count / len(rows) if rows else 0.0
        for label, count in source_display_counts.items()
    }
    final_counts = Counter(int(row["final_action_id"]) for row in rows)
    summary = {
        "experiment": "timeline",
        "scenario_name": args.scenario_name,
        "agent_id": args.agent_id,
        "actor_path": str(actor_path),
        "enemy_path": str(enemy_path) if fixed_enemy_action_id is None else "",
        "enemy_action": args.enemy_action or "",
        "manual_schedule": schedule,
        "max_steps": args.max_steps,
        "actual_steps": len(rows),
        "hold_steps": args.hold_steps,
        "source_counts": dict(source_counts),
        "source_display_counts": source_display_counts,
        "source_display_ratios": source_display_ratios,
        "final_action_counts": {str(key): value for key, value in final_counts.items()},
        "safety_overrides": sum(1 for row in rows if row["safety_overridden"]),
    }
    write_csv(output_dir / "timeline_steps.csv", rows)
    write_json(output_dir / "timeline_summary.json", summary)
    if not args.no_plots:
        plot_timeline_results(rows, summary, output_dir, args.formats)
    print(f"[timeline] 输出目录: {output_dir}")


def plot_timeline_results(rows: Sequence[dict[str, Any]], summary: dict[str, Any], output_dir: Path, formats: Sequence[str]) -> None:
    plt = load_matplotlib()
    fig_dir = output_dir / "figures"
    if plt is None:
        write_svg_timeline(fig_dir / "timeline_actions.svg", rows)
        source_counts = summary["source_display_counts"]
        write_svg_bar_chart(
            fig_dir / "timeline_source_ratio.svg",
            "动作来源环境步数",
            list(source_counts.keys()),
            list(source_counts.values()),
            color="#4C72B0",
        )
        return

    steps = [int(row["step"]) for row in rows]
    actor_actions = [int(row["actor_action_id"]) for row in rows]
    final_actions = [int(row["final_action_id"]) for row in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(steps, actor_actions, where="post", label="actor 建议动作", color="#4C72B0", linewidth=1.6)
    ax.step(steps, final_actions, where="post", label="最终执行动作", color="#C44E52", linewidth=1.8)
    manual_ranges = _timeline_step_ranges(rows, lambda row: row["source"] in {"manual", "manual_plan"})
    safety_ranges = _timeline_step_ranges(rows, lambda row: _timeline_flag_is_true(row.get("safety_overridden")))
    for index, (start, end) in enumerate(manual_ranges):
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color="#F0C36D",
            alpha=0.25,
            label="人类指令接管" if index == 0 else None,
        )
    for index, (start, end) in enumerate(safety_ranges):
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            facecolor="#7A5195",
            edgecolor="#7A5195",
            alpha=0.18,
            hatch="///",
            linewidth=0.8,
            label="安全覆盖区间" if index == 0 else None,
        )

    ax.set_yticks([action.action_id for action in TACTICAL_ACTIONS], [f"{action.action_id}:{action.chinese_name}" for action in TACTICAL_ACTIONS])
    ax.set_xlabel("环境步")
    ax.set_ylabel("战术动作")
    ax.set_title("人类指令接管与自主决策动作时间轴")
    ax.legend(loc="upper right")
    save_figure(fig, fig_dir, "timeline_actions", formats)
    plt.close(fig)

    source_counts = summary["source_display_counts"]
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(source_counts.keys())
    values = list(source_counts.values())
    bars = ax.bar(labels, values, color=["#4C72B0", "#C44E52"])
    ax.set_title("动作来源环境步数")
    ax.set_ylabel("环境步数")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    save_figure(fig, fig_dir, "timeline_source_ratio", formats)
    plt.close(fig)


def build_safety_cases() -> list[SafetyCase]:
    cases: list[SafetyCase] = []

    def add(case_id: str, description: str, action_id: int | None, fallback_action_id: int | None, state: TacticalSafetyState) -> None:
        cases.append(SafetyCase(case_id, description, action_id, fallback_action_id, state))

    # 非法动作编号：覆盖有备用动作、无备用动作、备用动作非法等回退边界。
    add("invalid_99_fallback_lag", "非法动作编号回退到滞后追击", 99, 2, TacticalSafetyState())
    add("invalid_99_no_fallback", "非法动作编号且无备用动作，回退默认纯追击", 99, None, TacticalSafetyState())
    add("invalid_99_bad_fallback", "非法动作编号且备用动作也非法，回退默认纯追击", 99, 99, TacticalSafetyState())
    add("invalid_99_fallback_decelerate", "非法动作编号回退到平飞减速", 99, 7, TacticalSafetyState())

    # 低空边界：默认高度阈值为 altitude_limit_m + 1000m，俯冲类动作在阈值内改为爬升。
    add("low_dive_below_limit", "低于最低高度仍要求俯冲加速", 5, 0, TacticalSafetyState(altitude_m=2400.0, altitude_limit_m=2500.0))
    add("low_yoyo_below_limit", "低于最低高度仍要求低悠悠", 11, 0, TacticalSafetyState(altitude_m=2400.0, altitude_limit_m=2500.0))
    add("low_dive_at_limit", "正好在最低高度要求俯冲加速", 5, 0, TacticalSafetyState(altitude_m=2500.0, altitude_limit_m=2500.0))
    add("low_yoyo_at_limit", "正好在最低高度要求低悠悠", 11, 0, TacticalSafetyState(altitude_m=2500.0, altitude_limit_m=2500.0))
    add("low_dive_at_margin", "正好在低空保护上沿要求俯冲加速", 5, 0, TacticalSafetyState(altitude_m=3500.0, altitude_limit_m=2500.0))
    add("low_yoyo_at_margin", "正好在低空保护上沿要求低悠悠", 11, 0, TacticalSafetyState(altitude_m=3500.0, altitude_limit_m=2500.0))
    add("low_dive_above_margin", "略高于低空保护上沿要求俯冲加速", 5, 0, TacticalSafetyState(altitude_m=3501.0, altitude_limit_m=2500.0))
    add("low_yoyo_above_margin", "略高于低空保护上沿要求低悠悠", 11, 0, TacticalSafetyState(altitude_m=3501.0, altitude_limit_m=2500.0))
    add("altitude_unknown_dive", "高度未知时要求俯冲加速", 5, 0, TacticalSafetyState())
    add("low_pure_no_altitude_override", "低空纯追击不触发低空俯冲保护", 0, 0, TacticalSafetyState(altitude_m=2500.0, altitude_limit_m=2500.0))
    add("low_climb_no_override", "低空爬升占位保持原动作", 4, 0, TacticalSafetyState(altitude_m=2500.0, altitude_limit_m=2500.0))
    add("custom_limit_dive_at_margin", "自定义高度限制上沿要求俯冲加速", 5, 0, TacticalSafetyState(altitude_m=4500.0, altitude_limit_m=3500.0))

    # 近距高速边界：距离不大于 1200m 且速度不低于 260m/s 时，进攻追击改为平飞减速。
    add("close_fast_pure_inside", "近距高速仍要求纯追击", 0, 0, TacticalSafetyState(distance_m=1199.0, speed_mps=261.0))
    add("close_fast_lead_inside", "近距高速仍要求提前量追击", 1, 0, TacticalSafetyState(distance_m=1000.0, speed_mps=290.0))
    add("close_fast_pure_at_distance", "正好在近距阈值要求纯追击", 0, 0, TacticalSafetyState(distance_m=1200.0, speed_mps=300.0))
    add("close_fast_lead_at_speed", "正好在高速阈值要求提前量追击", 1, 0, TacticalSafetyState(distance_m=900.0, speed_mps=260.0))
    add("close_fast_pure_distance_above", "略远于近距阈值要求纯追击", 0, 0, TacticalSafetyState(distance_m=1201.0, speed_mps=300.0))
    add("close_fast_lead_speed_below", "略低于高速阈值要求提前量追击", 1, 0, TacticalSafetyState(distance_m=900.0, speed_mps=259.9))
    add("close_missing_distance", "距离未知时高速纯追击", 0, 0, TacticalSafetyState(speed_mps=300.0))
    add("close_missing_speed", "速度未知时近距纯追击", 0, 0, TacticalSafetyState(distance_m=900.0))
    add("close_fast_lag_no_override", "近距高速滞后追击保持原动作", 2, 0, TacticalSafetyState(distance_m=900.0, speed_mps=300.0))
    add("close_fast_decelerate_no_override", "近距高速平飞减速保持原动作", 7, 0, TacticalSafetyState(distance_m=900.0, speed_mps=300.0))
    add("close_fast_pure_with_low_altitude", "低空但纯追击仍由近距高速规则接管", 0, 0, TacticalSafetyState(altitude_m=2500.0, distance_m=900.0, speed_mps=300.0))
    add("close_fast_lead_extreme", "极近距极高速仍要求提前量追击", 1, 0, TacticalSafetyState(distance_m=100.0, speed_mps=500.0))

    # 不利姿态边界：AO 大于 2.4 且 TA 小于 0.8 时，进攻动作改为脱离。
    add("bad_posture_pure", "不利姿态下仍要求纯追击", 0, 0, TacticalSafetyState(ao_rad=2.41, ta_rad=0.79))
    add("bad_posture_lead", "不利姿态下仍要求提前量追击", 1, 0, TacticalSafetyState(ao_rad=2.8, ta_rad=0.3))
    add("bad_posture_dive", "不利姿态下仍要求俯冲加速", 5, 0, TacticalSafetyState(ao_rad=2.7, ta_rad=0.4))
    add("bad_posture_low_yoyo", "不利姿态下仍要求低悠悠", 11, 0, TacticalSafetyState(ao_rad=2.9, ta_rad=0.2))
    add("bad_posture_ao_equal", "AO 正好等于阈值时纯追击", 0, 0, TacticalSafetyState(ao_rad=2.4, ta_rad=0.7))
    add("bad_posture_ta_equal", "TA 正好等于阈值时纯追击", 0, 0, TacticalSafetyState(ao_rad=2.5, ta_rad=0.8))
    add("bad_posture_ao_below", "AO 略低于阈值时提前量追击", 1, 0, TacticalSafetyState(ao_rad=2.39, ta_rad=0.7))
    add("bad_posture_ta_above", "TA 略高于阈值时俯冲加速", 5, 0, TacticalSafetyState(ao_rad=2.5, ta_rad=0.81))
    add("bad_posture_lag_no_override", "不利姿态下滞后追击保持原动作", 2, 0, TacticalSafetyState(ao_rad=3.0, ta_rad=0.1))
    add("bad_posture_disengage_no_override", "不利姿态下脱离保持原动作", 3, 0, TacticalSafetyState(ao_rad=3.0, ta_rad=0.1))
    add("bad_posture_missing_ao", "AO 未知时纯追击", 0, 0, TacticalSafetyState(ta_rad=0.3))
    add("bad_posture_missing_ta", "TA 未知时纯追击", 0, 0, TacticalSafetyState(ao_rad=2.6))

    # 规则优先级与正常动作：验证 if/elif 顺序以及安全状态下不误覆盖。
    add("precedence_low_over_bad_dive", "低空且姿态不利时俯冲优先触发低空保护", 5, 0, TacticalSafetyState(altitude_m=3000.0, ao_rad=2.8, ta_rad=0.2))
    add("precedence_close_over_bad_pure", "近距高速且姿态不利时纯追优先触发过冲保护", 0, 0, TacticalSafetyState(distance_m=900.0, speed_mps=300.0, ao_rad=2.8, ta_rad=0.2))
    add("precedence_low_over_bad_low_yoyo", "低空且姿态不利时低悠悠优先触发低空保护", 11, 0, TacticalSafetyState(altitude_m=3000.0, ao_rad=2.8, ta_rad=0.2))
    add("safe_pure", "安全状态下纯追击不覆盖", 0, 0, TacticalSafetyState(altitude_m=6000.0, distance_m=3000.0, speed_mps=220.0, ao_rad=1.0, ta_rad=1.2))
    add("safe_lead", "安全状态下提前量追击不覆盖", 1, 0, TacticalSafetyState(altitude_m=6000.0, distance_m=3000.0, speed_mps=220.0, ao_rad=1.0, ta_rad=1.2))
    add("safe_level_accelerate", "安全状态下平飞加速不覆盖", 6, 0, TacticalSafetyState(altitude_m=6000.0, distance_m=3000.0, speed_mps=220.0, ao_rad=1.0, ta_rad=1.2))
    add("safe_defensive_left", "安全状态下左防御转弯不覆盖", 8, 0, TacticalSafetyState(altitude_m=6000.0, distance_m=3000.0, speed_mps=220.0, ao_rad=1.0, ta_rad=1.2))
    add("safe_defensive_right", "安全状态下右防御转弯不覆盖", 9, 0, TacticalSafetyState(altitude_m=6000.0, distance_m=3000.0, speed_mps=220.0, ao_rad=1.0, ta_rad=1.2))
    add("safe_high_yoyo", "安全状态下高悠悠不覆盖", 10, 0, TacticalSafetyState(altitude_m=6000.0, distance_m=3000.0, speed_mps=220.0, ao_rad=1.0, ta_rad=1.2))
    add("safe_low_yoyo_high_altitude", "高空安全状态下低悠悠不覆盖", 11, 0, TacticalSafetyState(altitude_m=6000.0, distance_m=3000.0, speed_mps=220.0, ao_rad=1.0, ta_rad=1.2))

    if len(cases) != 50:
        raise AssertionError(f"安全覆盖实验应包含 50 个样例，当前为 {len(cases)} 个。")
    return cases


def categorize_safety_reason(reason: str, overridden: bool) -> str:
    if not overridden:
        return "none"
    if "低空" in reason or "高度" in reason:
        return "low_altitude"
    if "过冲" in reason or "近距高速" in reason:
        return "close_fast"
    if "姿态不利" in reason:
        return "bad_posture"
    if "非法" in reason or "编号非法" in reason:
        return "invalid_action"
    return "other"


SAFETY_REASON_DISPLAY_NAMES = {
    "none": "安全动作",
    "invalid_action": "非法动作",
    "low_altitude": "低空",
    "close_fast": "近距高速",
    "bad_posture": "不利姿态",
    "other": "其他",
}


def safety_reason_display_counts(reason_counts: dict[str, int]) -> dict[str, int]:
    return {
        display_name: int(reason_counts.get(category, 0))
        for category, display_name in SAFETY_REASON_DISPLAY_NAMES.items()
        if int(reason_counts.get(category, 0)) > 0
    }


def evaluate_safety_cases(cases: Sequence[SafetyCase]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        result = apply_tactical_safety(case.action_id, state=case.state, fallback_action_id=case.fallback_action_id)
        original = result.original_action_id
        final = result.action_id
        reason_category = categorize_safety_reason(result.reason, result.overridden)
        rows.append(
            {
                "case_id": case.case_id,
                "description": case.description,
                "input_action_id": case.action_id,
                "input_action_name": action_name(case.action_id) if case.action_id in ACTION_BY_ID else "INVALID",
                "input_action_cn": action_chinese_name(case.action_id) if case.action_id in ACTION_BY_ID else "非法动作",
                "fallback_action_id": case.fallback_action_id,
                "original_action_id": original if original is not None else "",
                "final_action_id": final,
                "final_action_name": action_name(final),
                "final_action_cn": action_chinese_name(final),
                "overridden": result.overridden,
                "reason": result.reason,
                "reason_category": reason_category,
                "reason_category_cn": SAFETY_REASON_DISPLAY_NAMES[reason_category],
                "altitude_m": case.state.altitude_m if case.state.altitude_m is not None else "",
                "altitude_limit_m": case.state.altitude_limit_m,
                "distance_m": case.state.distance_m if case.state.distance_m is not None else "",
                "speed_mps": case.state.speed_mps if case.state.speed_mps is not None else "",
                "ao_rad": case.state.ao_rad if case.state.ao_rad is not None else "",
                "ta_rad": case.state.ta_rad if case.state.ta_rad is not None else "",
            }
        )

    reason_counts = Counter(str(row["reason_category"]) for row in rows)
    replacement_counts = Counter((str(row["input_action_id"]), str(row["final_action_id"])) for row in rows)
    summary = {
        "experiment": "safety",
        "total": len(rows),
        "overridden_count": sum(1 for row in rows if row["overridden"]),
        "reason_counts": dict(reason_counts),
        "reason_display_counts": safety_reason_display_counts(dict(reason_counts)),
        "replacement_counts": {f"{source}->{target}": count for (source, target), count in replacement_counts.items()},
    }
    return rows, summary


def run_safety_experiment(args: argparse.Namespace, output_root: Path) -> None:
    output_dir = output_root / "safety"
    rows, summary = evaluate_safety_cases(build_safety_cases())
    write_csv(output_dir / "safety_results.csv", rows)
    write_json(output_dir / "safety_summary.json", summary)
    if not args.no_plots:
        plot_safety_results(rows, summary, output_dir, args.formats)
    print(f"[safety] 输出目录: {output_dir}")


def plot_safety_results(rows: Sequence[dict[str, Any]], summary: dict[str, Any], output_dir: Path, formats: Sequence[str]) -> None:
    plt = load_matplotlib()
    fig_dir = output_dir / "figures"
    if plt is None:
        reason_counts = summary["reason_display_counts"]
        write_svg_bar_chart(
            fig_dir / "safety_reason_counts.svg",
            "安全覆盖原因分布",
            list(reason_counts.keys()),
            list(reason_counts.values()),
            color="#C44E52",
        )
        labels = ["99:非法"] + [f"{action.action_id}:{action.chinese_name}" for action in TACTICAL_ACTIONS]
        index_by_input = {"99": 0}
        for index, action in enumerate(TACTICAL_ACTIONS, start=1):
            index_by_input[str(action.action_id)] = index
        output_labels = [f"{action.action_id}:{action.chinese_name}" for action in TACTICAL_ACTIONS]
        matrix = [[0 for _ in output_labels] for _ in labels]
        for row in rows:
            input_id = str(row["input_action_id"])
            final_id = int(row["final_action_id"])
            if input_id in index_by_input:
                matrix[index_by_input[input_id]][final_id] += 1
        write_svg_matrix(
            fig_dir / "safety_replacement_matrix.svg",
            "安全覆盖动作替换矩阵",
            output_labels,
            labels,
            matrix,
            color="#D55E00",
        )
        return

    reason_counts = summary["reason_display_counts"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(reason_counts.keys()), list(reason_counts.values()), color="#C44E52")
    ax.set_title("安全覆盖原因分布")
    ax.set_ylabel("样本数")
    save_figure(fig, fig_dir, "safety_reason_counts", formats)
    plt.close(fig)

    labels = ["99:非法"] + [f"{action.action_id}:{action.chinese_name}" for action in TACTICAL_ACTIONS]
    index_by_input = {"99": 0}
    for index, action in enumerate(TACTICAL_ACTIONS, start=1):
        index_by_input[str(action.action_id)] = index
    output_labels = [f"{action.action_id}:{action.chinese_name}" for action in TACTICAL_ACTIONS]
    matrix = [[0 for _ in output_labels] for _ in labels]
    for row in rows:
        input_id = str(row["input_action_id"])
        final_id = int(row["final_action_id"])
        if input_id in index_by_input:
            matrix[index_by_input[input_id]][final_id] += 1

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, cmap="Oranges")
    ax.set_xticks(range(len(output_labels)), output_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("安全覆盖后动作")
    ax.set_ylabel("输入动作")
    ax.set_title("安全覆盖动作替换矩阵")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if value:
                ax.text(x, y, str(value), ha="center", va="center", color="#222222")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, fig_dir, "safety_replacement_matrix", formats)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent/LLM 融合论文实验脚本。")
    parser.add_argument("--experiment", choices=["parse", "timeline", "safety", "all"], default="all")
    parser.add_argument("--parser-mode", choices=["keyword", "llm_fallback"], default="keyword")
    parser.add_argument("--actor-path", default=str(DEFAULT_TACTICAL_ACTOR_PATH))
    parser.add_argument("--enemy-path", default="")
    parser.add_argument("--enemy-action", default="")
    parser.add_argument("--manual-schedule", default=DEFAULT_MANUAL_SCHEDULE)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--hold-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/TacticalHierarchySelfplay")
    parser.add_argument("--agent-id", default="A0100")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--formats", type=parse_formats, default=parse_formats("png,pdf"))
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_root = make_output_dir(args.output_dir)
    selected = ["parse", "timeline", "safety"] if args.experiment == "all" else [args.experiment]

    if "parse" in selected:
        run_parse_experiment(args, output_root)
    if "timeline" in selected:
        run_timeline_experiment(args, output_root)
    if "safety" in selected:
        run_safety_experiment(args, output_root)

    write_json(
        output_root / "run_config.json",
        {
            "experiment": args.experiment,
            "parser_mode": args.parser_mode,
            "actor_path": args.actor_path,
            "enemy_path": args.enemy_path,
            "enemy_action": args.enemy_action,
            "manual_schedule": args.manual_schedule,
            "max_steps": args.max_steps,
            "hold_steps": args.hold_steps,
            "seed": args.seed,
            "device": args.device,
            "scenario_name": args.scenario_name,
            "agent_id": args.agent_id,
            "formats": args.formats,
            "no_plots": args.no_plots,
            "output_root": str(output_root),
        },
    )
    print(f"全部完成，输出根目录: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
