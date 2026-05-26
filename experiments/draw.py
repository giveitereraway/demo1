"""读取 1v1 实验结果并生成论文插图。

该脚本只做后处理：读取已有 episodes.csv / summary.json，不启动仿真，
默认把图片输出到结果目录的 paper_figures/ 下。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except Exception as exc:  # pragma: no cover - 只有绘图库环境损坏时触发
    raise SystemExit(
        "未能导入 matplotlib，无法绘制图片。请先确认当前 Python 环境已正确安装 matplotlib/Pillow。"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = REPO_ROOT / "experiments" / "outputs"
# N局滑动平均的默认窗口；手动改这里即可改变不传 --window 时的曲线平滑程度。
DEFAULT_MOVING_AVERAGE_WINDOW = 20

ACTOR_A_COLOR = "#C44E52"
ACTOR_B_COLOR = "#4C72B0"
TIE_COLOR = "#7F7F7F"
MARGIN_COLOR = "#55A868"
AO_COLOR = "#8172B3"
TA_COLOR = "#CCB974"
GRID_COLOR = "#D0D0D0"

REQUIRED_RESULT_FILES = ("episodes.csv", "summary.json")


def resolve_path(value: Optional[str], default: Optional[Path] = None) -> Optional[Path]:
    """把命令行路径统一解析为绝对路径。"""
    if value is None:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def parse_formats(value: str) -> List[str]:
    """解析图片格式列表，允许 png,pdf 这种逗号写法。"""
    formats = []
    for item in value.split(","):
        suffix = item.strip().lower().lstrip(".")
        if not suffix:
            continue
        if suffix not in {"png", "pdf", "svg"}:
            raise argparse.ArgumentTypeError(f"暂不支持的图片格式: {item}")
        formats.append(suffix)
    if not formats:
        raise argparse.ArgumentTypeError("至少需要指定一种图片格式")
    return formats


def load_csv(path: Path) -> List[Dict[str, str]]:
    """读取 episodes.csv，保持字段名与原始结果一致。"""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_json(path: Path) -> Dict[str, Any]:
    """读取 summary.json；文件不存在时返回空字典。"""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_result(result_dir: Path) -> Dict[str, Any]:
    """读取单个实验目录。"""
    missing = [name for name in REQUIRED_RESULT_FILES if not (result_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{result_dir} 缺少结果文件: {', '.join(missing)}")

    rows = load_csv(result_dir / "episodes.csv")
    summary = load_json(result_dir / "summary.json")
    name = str(summary.get("experiment_name") or result_dir.name)
    return {"result_dir": result_dir, "rows": rows, "summary": summary, "name": name}


def has_columns(rows: Sequence[Dict[str, str]], columns: Sequence[str]) -> bool:
    """检查 CSV 是否包含指定字段。"""
    if not rows:
        return False
    row_keys = set(rows[0].keys())
    return all(column in row_keys for column in columns)


def as_float(value: Any) -> Optional[float]:
    """宽松转换数值，无法转换时返回 None。"""
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def numeric_series(rows: Sequence[Dict[str, str]], key: str) -> np.ndarray:
    """读取数值列；空值会记为 NaN，便于 matplotlib 自然断线。"""
    values = []
    for row in rows:
        value = as_float(row.get(key))
        values.append(np.nan if value is None else value)
    return np.asarray(values, dtype=np.float64)


def clean_numeric_values(rows: Sequence[Dict[str, str]], key: str) -> np.ndarray:
    """读取无 NaN 的数值列，主要用于统计均值和标准差。"""
    values = [as_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return np.asarray(values, dtype=np.float64)


def mean_from_rows(rows: Sequence[Dict[str, str]], key: str) -> Optional[float]:
    values = clean_numeric_values(rows, key)
    if values.size == 0:
        return None
    return float(np.mean(values))


def std_from_rows(rows: Sequence[Dict[str, str]], key: str) -> Optional[float]:
    values = clean_numeric_values(rows, key)
    if values.size <= 1:
        return 0.0 if values.size == 1 else None
    return float(np.std(values, ddof=1))


def summary_or_mean(summary: Dict[str, Any], rows: Sequence[Dict[str, str]], summary_key: str, csv_key: str) -> Optional[float]:
    """优先从 summary.json 读取统计值，缺失时回退到 CSV 均值。"""
    value = as_float(summary.get(summary_key))
    if value is not None:
        return value
    return mean_from_rows(rows, csv_key)


def summary_or_std(summary: Dict[str, Any], rows: Sequence[Dict[str, str]], summary_key: str, csv_key: str) -> Optional[float]:
    """优先从 summary.json 读取标准差，缺失时回退到 CSV 计算。"""
    value = as_float(summary.get(summary_key))
    if value is not None:
        return value
    return std_from_rows(rows, csv_key)


def winner_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    """统计胜负平数量。"""
    winners = [str(row.get("winner", "")).strip() for row in rows]
    return {
        "actor_a": winners.count("actor_a"),
        "actor_b": winners.count("actor_b"),
        "tie": winners.count("tie"),
    }


def outcome_rates(summary: Dict[str, Any], rows: Sequence[Dict[str, str]]) -> Dict[str, float]:
    """优先使用 summary 中的胜率，缺失时从逐回合 winner 计算。"""
    total = max(len(rows), 1)
    counts = winner_counts(rows)
    rates = {
        "actor_a": as_float(summary.get("actor_a_win_rate")),
        "actor_b": as_float(summary.get("actor_b_win_rate")),
        "tie": as_float(summary.get("tie_rate")),
    }
    return {
        key: float(value) if value is not None else counts[key] / total
        for key, value in rates.items()
    }


def reward_margin(rows: Sequence[Dict[str, str]]) -> np.ndarray:
    """读取奖励差；旧结果缺少 reward_margin 时用 A-B 回退计算。"""
    if has_columns(rows, ["reward_margin"]):
        return numeric_series(rows, "reward_margin")
    return numeric_series(rows, "actor_a_reward") - numeric_series(rows, "actor_b_reward")


def episode_axis(rows: Sequence[Dict[str, str]]) -> np.ndarray:
    """读取 episode 轴；旧结果缺少 episode 时使用 0..N-1。"""
    if has_columns(rows, ["episode"]):
        return numeric_series(rows, "episode")
    return np.arange(len(rows), dtype=np.float64)


def moving_average(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """计算滑动平均，自动忽略窗口内 NaN。"""
    if window <= 1 or values.size < window:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    averaged = []
    indexes = []
    for end in range(window, values.size + 1):
        chunk = values[end - window : end]
        if np.all(np.isnan(chunk)):
            continue
        averaged.append(float(np.nanmean(chunk)))
        indexes.append(end - 1)
    return np.asarray(indexes, dtype=np.int64), np.asarray(averaged, dtype=np.float64)


def configure_plot_style() -> None:
    """配置适合论文插图的 matplotlib 风格。"""
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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def setup_axis(ax, xlabel: str, ylabel: str, title: str) -> None:
    """统一坐标轴标题、网格和标签。"""
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.45)


def plot_with_average(ax, x: np.ndarray, y: np.ndarray, label: str, color: str, window: int) -> None:
    """绘制原始曲线和滑动平均曲线。"""
    ax.plot(x, y, color=color, linewidth=1.0, alpha=0.35, label=f"{label} 原始值")
    indexes, averaged = moving_average(y, window)
    if averaged.size:
        ax.plot(x[indexes], averaged, color=color, linewidth=2.2, label=f"{label} {window}局滑动平均")


def save_figure(fig, output_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> List[Path]:
    """按指定格式保存图片。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    plt.close(fig)
    return paths


def append_figure(index: List[Dict[str, Any]], name: str, description: str, paths: Sequence[Path]) -> None:
    index.append(
        {
            "name": name,
            "description": description,
            "files": [str(path) for path in paths],
        }
    )


def annotate_bars(ax, bars, fmt: str = "{:.2f}", dy: float = 3.0) -> None:
    """给柱状图添加数值标签。"""
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.015
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        va = "bottom" if height >= 0 else "top"
        y = height + offset if height >= 0 else height - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=8,
            rotation=0,
        )


def plot_reward_curve(result: Dict[str, Any], output_dir: Path, formats: Sequence[str], dpi: int, window: int) -> Optional[List[Path]]:
    rows = result["rows"]
    if not has_columns(rows, ["actor_a_reward", "actor_b_reward"]):
        print(f"跳过奖励曲线，缺少 actor_a_reward 或 actor_b_reward: {result['result_dir']}")
        return None

    x = episode_axis(rows)
    actor_a = numeric_series(rows, "actor_a_reward")
    actor_b = numeric_series(rows, "actor_b_reward")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    plot_with_average(ax, x, actor_a, "Actor A", ACTOR_A_COLOR, window)
    plot_with_average(ax, x, actor_b, "Actor B", ACTOR_B_COLOR, window)
    setup_axis(ax, "Episode", "累计奖励", "每回合累计奖励变化")
    ax.legend(frameon=False, ncol=2)
    return save_figure(fig, output_dir, "reward_curve", formats, dpi)


def plot_reward_margin_curve(result: Dict[str, Any], output_dir: Path, formats: Sequence[str], dpi: int, window: int) -> Optional[List[Path]]:
    rows = result["rows"]
    if not has_columns(rows, ["actor_a_reward", "actor_b_reward"]):
        print(f"跳过奖励差曲线，缺少奖励字段: {result['result_dir']}")
        return None

    x = episode_axis(rows)
    margin = reward_margin(rows)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.fill_between(x, margin, 0, where=margin >= 0, color=ACTOR_A_COLOR, alpha=0.14, interpolate=True)
    ax.fill_between(x, margin, 0, where=margin < 0, color=ACTOR_B_COLOR, alpha=0.14, interpolate=True)
    plot_with_average(ax, x, margin, "A-B 奖励差", MARGIN_COLOR, window)
    setup_axis(ax, "Episode", "Actor A 奖励 - Actor B 奖励", "奖励差变化")
    ax.legend(frameon=False)
    return save_figure(fig, output_dir, "reward_margin_curve", formats, dpi)


def plot_win_rate_curve(result: Dict[str, Any], output_dir: Path, formats: Sequence[str], dpi: int) -> Optional[List[Path]]:
    rows = result["rows"]
    if not has_columns(rows, ["winner"]):
        print(f"跳过累计胜率曲线，缺少 winner 字段: {result['result_dir']}")
        return None

    x = episode_axis(rows)
    winners = [str(row.get("winner", "")).strip() for row in rows]
    denominator = np.arange(1, len(winners) + 1, dtype=np.float64)
    actor_a_rate = np.cumsum([winner == "actor_a" for winner in winners]) / denominator
    actor_b_rate = np.cumsum([winner == "actor_b" for winner in winners]) / denominator
    tie_rate = np.cumsum([winner == "tie" for winner in winners]) / denominator

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(x, actor_a_rate, color=ACTOR_A_COLOR, linewidth=2.0, label="Actor A 累计胜率")
    ax.plot(x, actor_b_rate, color=ACTOR_B_COLOR, linewidth=2.0, label="Actor B 累计胜率")
    ax.plot(x, tie_rate, color=TIE_COLOR, linewidth=2.0, label="累计平局率")
    setup_axis(ax, "Episode", "比例", "累计胜率和平局率")
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, ncol=3)
    return save_figure(fig, output_dir, "win_rate_curve", formats, dpi)


def plot_outcome_bar(result: Dict[str, Any], output_dir: Path, formats: Sequence[str], dpi: int) -> Optional[List[Path]]:
    rows = result["rows"]
    if not rows:
        print(f"跳过胜负平柱状图，episodes.csv 为空: {result['result_dir']}")
        return None

    rates = outcome_rates(result["summary"], rows)
    labels = ["Actor A 胜", "Actor B 胜", "平局"]
    values = [rates["actor_a"], rates["actor_b"], rates["tie"]]
    colors = [ACTOR_A_COLOR, ACTOR_B_COLOR, TIE_COLOR]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    bars = ax.bar(labels, values, color=colors, width=0.58, edgecolor="#333333", linewidth=0.6)
    setup_axis(ax, "", "比例", "最终胜负平分布")
    ax.set_ylim(0, min(1.05, max(values) + 0.16))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    annotate_bars(ax, bars, fmt="{:.1%}")
    return save_figure(fig, output_dir, "outcome_bar", formats, dpi)


def has_aota_columns(rows: Sequence[Dict[str, str]]) -> bool:
    return has_columns(
        rows,
        [
            "actor_a_episode_mean_ao_deg",
            "actor_a_episode_mean_ta_deg",
            "actor_b_episode_mean_ao_deg",
            "actor_b_episode_mean_ta_deg",
        ],
    )


def plot_aota_curve(result: Dict[str, Any], output_dir: Path, formats: Sequence[str], dpi: int, window: int) -> Optional[List[Path]]:
    rows = result["rows"]
    if not has_aota_columns(rows):
        print(f"跳过 AO/TA 曲线，缺少 episode mean AO/TA 字段: {result['result_dir']}")
        return None

    x = episode_axis(rows)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)

    plot_with_average(axes[0], x, numeric_series(rows, "actor_a_episode_mean_ao_deg"), "Actor A AO", ACTOR_A_COLOR, window)
    plot_with_average(axes[0], x, numeric_series(rows, "actor_b_episode_mean_ao_deg"), "Actor B AO", ACTOR_B_COLOR, window)
    setup_axis(axes[0], "", "AO (deg)", "平均方位进入角 AO")
    axes[0].set_ylim(0, 180)
    axes[0].axhline(90, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
    axes[0].legend(frameon=False, ncol=2)

    plot_with_average(axes[1], x, numeric_series(rows, "actor_a_episode_mean_ta_deg"), "Actor A TA", ACTOR_A_COLOR, window)
    plot_with_average(axes[1], x, numeric_series(rows, "actor_b_episode_mean_ta_deg"), "Actor B TA", ACTOR_B_COLOR, window)
    setup_axis(axes[1], "Episode", "TA (deg)", "平均目标方位角 TA")
    axes[1].set_ylim(0, 180)
    axes[1].axhline(90, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
    axes[1].legend(frameon=False, ncol=2)

    return save_figure(fig, output_dir, "aota_curve", formats, dpi)


def plot_aota_summary_bar(result: Dict[str, Any], output_dir: Path, formats: Sequence[str], dpi: int) -> Optional[List[Path]]:
    rows = result["rows"]
    if not has_aota_columns(rows):
        print(f"跳过 AO/TA 统计柱状图，缺少 episode mean AO/TA 字段: {result['result_dir']}")
        return None

    summary = result["summary"]
    metrics = [
        ("AO", "actor_a_episode_mean_ao_avg_deg", "actor_b_episode_mean_ao_avg_deg", "actor_a_episode_mean_ao_std_deg", "actor_b_episode_mean_ao_std_deg", "actor_a_episode_mean_ao_deg", "actor_b_episode_mean_ao_deg"),
        ("TA", "actor_a_episode_mean_ta_avg_deg", "actor_b_episode_mean_ta_avg_deg", "actor_a_episode_mean_ta_std_deg", "actor_b_episode_mean_ta_std_deg", "actor_a_episode_mean_ta_deg", "actor_b_episode_mean_ta_deg"),
    ]

    actor_a_means = []
    actor_b_means = []
    actor_a_stds = []
    actor_b_stds = []
    labels = []
    for label, a_mean_key, b_mean_key, a_std_key, b_std_key, a_csv_key, b_csv_key in metrics:
        labels.append(label)
        actor_a_means.append(summary_or_mean(summary, rows, a_mean_key, a_csv_key) or 0.0)
        actor_b_means.append(summary_or_mean(summary, rows, b_mean_key, b_csv_key) or 0.0)
        actor_a_stds.append(summary_or_std(summary, rows, a_std_key, a_csv_key) or 0.0)
        actor_b_stds.append(summary_or_std(summary, rows, b_std_key, b_csv_key) or 0.0)

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bars_a = ax.bar(
        x - width / 2,
        actor_a_means,
        width,
        yerr=actor_a_stds,
        capsize=4,
        color=ACTOR_A_COLOR,
        label="Actor A",
        edgecolor="#333333",
        linewidth=0.6,
    )
    bars_b = ax.bar(
        x + width / 2,
        actor_b_means,
        width,
        yerr=actor_b_stds,
        capsize=4,
        color=ACTOR_B_COLOR,
        label="Actor B",
        edgecolor="#333333",
        linewidth=0.6,
    )
    setup_axis(ax, "", "角度 (deg)", "AO/TA 均值与波动")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 180)
    ax.legend(frameon=False)
    annotate_bars(ax, bars_a, fmt="{:.1f}")
    annotate_bars(ax, bars_b, fmt="{:.1f}")
    return save_figure(fig, output_dir, "aota_summary_bar", formats, dpi)


def collect_key_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """整理 figure_index.json 中需要记录的关键指标。"""
    rows = result["rows"]
    summary = result["summary"]
    rates = outcome_rates(summary, rows) if rows else {"actor_a": 0.0, "actor_b": 0.0, "tie": 0.0}
    metrics = {
        "experiment_name": result["name"],
        "result_dir": str(result["result_dir"]),
        "num_episodes": int(as_float(summary.get("num_episodes")) or len(rows)),
        "actor_a_win_rate": rates["actor_a"],
        "actor_b_win_rate": rates["actor_b"],
        "tie_rate": rates["tie"],
        "actor_a_avg_reward": summary_or_mean(summary, rows, "actor_a_avg_reward", "actor_a_reward"),
        "actor_b_avg_reward": summary_or_mean(summary, rows, "actor_b_avg_reward", "actor_b_reward"),
        "avg_reward_margin": summary_or_mean(summary, rows, "avg_reward_margin", "reward_margin"),
        "actor_a_reward_std": summary_or_std(summary, rows, "actor_a_reward_std", "actor_a_reward"),
        "actor_b_reward_std": summary_or_std(summary, rows, "actor_b_reward_std", "actor_b_reward"),
    }
    if has_aota_columns(rows):
        metrics.update(
            {
                "actor_a_episode_mean_ao_avg_deg": summary_or_mean(summary, rows, "actor_a_episode_mean_ao_avg_deg", "actor_a_episode_mean_ao_deg"),
                "actor_b_episode_mean_ao_avg_deg": summary_or_mean(summary, rows, "actor_b_episode_mean_ao_avg_deg", "actor_b_episode_mean_ao_deg"),
                "actor_a_episode_mean_ta_avg_deg": summary_or_mean(summary, rows, "actor_a_episode_mean_ta_avg_deg", "actor_a_episode_mean_ta_deg"),
                "actor_b_episode_mean_ta_avg_deg": summary_or_mean(summary, rows, "actor_b_episode_mean_ta_avg_deg", "actor_b_episode_mean_ta_deg"),
            }
        )
    return metrics


def write_figure_index(output_dir: Path, payload: Dict[str, Any]) -> Path:
    path = output_dir / "figure_index.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def draw_single_result(result_dir: Path, output_dir: Path, formats: Sequence[str], dpi: int, window: int) -> Dict[str, Any]:
    """为单个实验目录绘制论文插图。"""
    result = load_result(result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_records: List[Dict[str, Any]] = []
    plotters = [
        ("reward_curve", "每回合 Actor A/B 累计奖励曲线", lambda: plot_reward_curve(result, output_dir, formats, dpi, window)),
        ("reward_margin_curve", "每回合 Actor A-B 奖励差曲线", lambda: plot_reward_margin_curve(result, output_dir, formats, dpi, window)),
        ("win_rate_curve", "累计胜率和平局率曲线", lambda: plot_win_rate_curve(result, output_dir, formats, dpi)),
        ("outcome_bar", "最终胜负平比例柱状图", lambda: plot_outcome_bar(result, output_dir, formats, dpi)),
        ("aota_curve", "双方 episode mean AO/TA 曲线", lambda: plot_aota_curve(result, output_dir, formats, dpi, window)),
        ("aota_summary_bar", "双方 episode mean AO/TA 均值与标准差柱状图", lambda: plot_aota_summary_bar(result, output_dir, formats, dpi)),
    ]

    for name, description, plotter in plotters:
        paths = plotter()
        if paths:
            append_figure(figure_records, name, description, paths)

    payload = {
        "mode": "single",
        "input_result_dir": str(result_dir),
        "output_dir": str(output_dir),
        "moving_average_window": window,
        "dpi": dpi,
        "formats": list(formats),
        "key_metrics": collect_key_metrics(result),
        "figures": figure_records,
    }
    index_path = write_figure_index(output_dir, payload)
    print(f"已生成单实验论文插图: {output_dir}")
    print(f"索引文件: {index_path}")
    return payload


def scan_result_dirs(result_root: Path) -> List[Path]:
    """扫描包含 episodes.csv 和 summary.json 的实验目录。"""
    if all((result_root / name).exists() for name in REQUIRED_RESULT_FILES):
        return [result_root]
    if not result_root.exists():
        raise FileNotFoundError(f"结果根目录不存在: {result_root}")
    return sorted(
        [
            child
            for child in result_root.iterdir()
            if child.is_dir() and all((child / name).exists() for name in REQUIRED_RESULT_FILES)
        ],
        key=lambda path: path.name,
    )


def compact_label(result: Dict[str, Any]) -> str:
    """生成适合横轴显示的实验名。"""
    name = str(result["name"])
    if "_vs_" in name:
        task, pair = name.split("_", 1)
        left, right = pair.split("_vs_", 1)
        task_labels = {"NoWeapon": "无武器", "ShootMissile": "导弹"}
        model_labels = {"self": "自博弈", "hierarchy": "分层", "tactical": "战术"}
        task_label = task_labels.get(task, task)
        left_label = model_labels.get(left, left)
        right_label = model_labels.get(right, right)
        return f"{task_label}\n{left_label} vs {right_label}"
    name = name.replace("NoWeapon_", "无武器\n")
    name = name.replace("ShootMissile_", "导弹\n")
    name = name.replace("_vs_", " vs ")
    return name


def comparison_win_rate_bar(results: Sequence[Dict[str, Any]], output_dir: Path, formats: Sequence[str], dpi: int) -> List[Path]:
    labels = [compact_label(result) for result in results]
    rates = [outcome_rates(result["summary"], result["rows"]) for result in results]
    actor_a = [item["actor_a"] for item in rates]
    actor_b = [item["actor_b"] for item in rates]
    ties = [item["tie"] for item in rates]

    x = np.arange(len(results))
    width = 0.25
    fig_width = max(9.0, len(results) * 1.45)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    ax.bar(x - width, actor_a, width, color=ACTOR_A_COLOR, label="Actor A 胜率", edgecolor="#333333", linewidth=0.5)
    ax.bar(x, actor_b, width, color=ACTOR_B_COLOR, label="Actor B 胜率", edgecolor="#333333", linewidth=0.5)
    ax.bar(x + width, ties, width, color=TIE_COLOR, label="平局率", edgecolor="#333333", linewidth=0.5)
    setup_axis(ax, "", "比例", "多组实验胜负平对比")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, ncol=3)
    return save_figure(fig, output_dir, "comparison_win_rate_bar", formats, dpi)


def comparison_reward_bar(results: Sequence[Dict[str, Any]], output_dir: Path, formats: Sequence[str], dpi: int) -> List[Path]:
    labels = [compact_label(result) for result in results]
    actor_a_mean = [summary_or_mean(result["summary"], result["rows"], "actor_a_avg_reward", "actor_a_reward") or 0.0 for result in results]
    actor_b_mean = [summary_or_mean(result["summary"], result["rows"], "actor_b_avg_reward", "actor_b_reward") or 0.0 for result in results]
    actor_a_std = [summary_or_std(result["summary"], result["rows"], "actor_a_reward_std", "actor_a_reward") or 0.0 for result in results]
    actor_b_std = [summary_or_std(result["summary"], result["rows"], "actor_b_reward_std", "actor_b_reward") or 0.0 for result in results]

    x = np.arange(len(results))
    width = 0.36
    fig_width = max(9.0, len(results) * 1.45)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.bar(
        x - width / 2,
        actor_a_mean,
        width,
        yerr=actor_a_std,
        capsize=3,
        color=ACTOR_A_COLOR,
        label="Actor A 平均奖励",
        edgecolor="#333333",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        actor_b_mean,
        width,
        yerr=actor_b_std,
        capsize=3,
        color=ACTOR_B_COLOR,
        label="Actor B 平均奖励",
        edgecolor="#333333",
        linewidth=0.5,
    )
    setup_axis(ax, "", "平均累计奖励", "多组实验平均奖励对比")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, ncol=2)
    return save_figure(fig, output_dir, "comparison_reward_bar", formats, dpi)


def comparison_aota_bar(results: Sequence[Dict[str, Any]], output_dir: Path, formats: Sequence[str], dpi: int) -> Optional[List[Path]]:
    aota_results = [result for result in results if has_aota_columns(result["rows"])]
    if not aota_results:
        print("跳过多实验 AO/TA 对比图，未发现包含 AO/TA 字段的实验。")
        return None

    labels = [compact_label(result) for result in aota_results]
    x = np.arange(len(aota_results))
    width = 0.36
    fig_width = max(9.0, len(aota_results) * 1.45)
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 8.2), sharex=True)

    metric_specs = [
        (
            axes[0],
            "AO (deg)",
            "多组实验平均 AO 对比",
            "actor_a_episode_mean_ao_avg_deg",
            "actor_b_episode_mean_ao_avg_deg",
            "actor_a_episode_mean_ao_std_deg",
            "actor_b_episode_mean_ao_std_deg",
            "actor_a_episode_mean_ao_deg",
            "actor_b_episode_mean_ao_deg",
        ),
        (
            axes[1],
            "TA (deg)",
            "多组实验平均 TA 对比",
            "actor_a_episode_mean_ta_avg_deg",
            "actor_b_episode_mean_ta_avg_deg",
            "actor_a_episode_mean_ta_std_deg",
            "actor_b_episode_mean_ta_std_deg",
            "actor_a_episode_mean_ta_deg",
            "actor_b_episode_mean_ta_deg",
        ),
    ]

    for ax, ylabel, title, a_mean_key, b_mean_key, a_std_key, b_std_key, a_csv_key, b_csv_key in metric_specs:
        actor_a_mean = [summary_or_mean(result["summary"], result["rows"], a_mean_key, a_csv_key) or 0.0 for result in aota_results]
        actor_b_mean = [summary_or_mean(result["summary"], result["rows"], b_mean_key, b_csv_key) or 0.0 for result in aota_results]
        actor_a_std = [summary_or_std(result["summary"], result["rows"], a_std_key, a_csv_key) or 0.0 for result in aota_results]
        actor_b_std = [summary_or_std(result["summary"], result["rows"], b_std_key, b_csv_key) or 0.0 for result in aota_results]
        ax.bar(
            x - width / 2,
            actor_a_mean,
            width,
            yerr=actor_a_std,
            capsize=3,
            color=ACTOR_A_COLOR,
            label="Actor A",
            edgecolor="#333333",
            linewidth=0.5,
        )
        ax.bar(
            x + width / 2,
            actor_b_mean,
            width,
            yerr=actor_b_std,
            capsize=3,
            color=ACTOR_B_COLOR,
            label="Actor B",
            edgecolor="#333333",
            linewidth=0.5,
        )
        setup_axis(ax, "", ylabel, title)
        ax.set_ylim(0, 180)
        ax.axhline(90, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
        ax.legend(frameon=False, ncol=2)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    return save_figure(fig, output_dir, "comparison_aota_bar", formats, dpi)


def draw_all_results(result_root: Path, output_dir: Path, formats: Sequence[str], dpi: int, window: int) -> Dict[str, Any]:
    """扫描结果根目录并生成跨实验对比图。"""
    result_dirs = scan_result_dirs(result_root)
    if not result_dirs:
        raise FileNotFoundError(f"未在 {result_root} 下找到包含 episodes.csv 和 summary.json 的实验目录")

    results = [load_result(path) for path in result_dirs]
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_records: List[Dict[str, Any]] = []

    paths = comparison_win_rate_bar(results, output_dir, formats, dpi)
    append_figure(figure_records, "comparison_win_rate_bar", "多实验 A/B/Tie 比例分组柱状图", paths)

    paths = comparison_reward_bar(results, output_dir, formats, dpi)
    append_figure(figure_records, "comparison_reward_bar", "多实验 A/B 平均奖励分组柱状图", paths)

    paths = comparison_aota_bar(results, output_dir, formats, dpi)
    if paths:
        append_figure(figure_records, "comparison_aota_bar", "多实验双方 episode mean AO/TA 对比图", paths)

    payload = {
        "mode": "all",
        "input_result_root": str(result_root),
        "input_result_dirs": [str(path) for path in result_dirs],
        "output_dir": str(output_dir),
        "moving_average_window": window,
        "dpi": dpi,
        "formats": list(formats),
        "experiments": [collect_key_metrics(result) for result in results],
        "figures": figure_records,
    }
    index_path = write_figure_index(output_dir, payload)
    print(f"已生成多实验论文插图: {output_dir}")
    print(f"索引文件: {index_path}")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="读取 1v1 实验结果并生成论文插图。")
    parser.add_argument("--result-dir", default=None, help="单个实验结果目录，需包含 episodes.csv 和 summary.json。")
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT), help="批量扫描的实验输出根目录。")
    parser.add_argument("--all", action="store_true", help="扫描 result-root 下所有实验目录并生成跨实验对比图。")
    parser.add_argument("--output-dir", default=None, help="图片输出目录；默认写入 paper_figures/。")
    parser.add_argument("--window", type=int, default=DEFAULT_MOVING_AVERAGE_WINDOW, help="滑动平均窗口大小。")
    parser.add_argument("--dpi", type=int, default=300, help="图片导出 DPI。")
    parser.add_argument("--formats", type=parse_formats, default=parse_formats("png,pdf"), help="导出格式，逗号分隔，例如 png,pdf。")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    configure_plot_style()
    window = max(1, int(args.window))
    dpi = max(72, int(args.dpi))
    formats = args.formats

    if args.all:
        result_root = resolve_path(args.result_root)
        assert result_root is not None
        output_dir = resolve_path(args.output_dir, result_root / "paper_figures")
        assert output_dir is not None
        draw_all_results(result_root, output_dir, formats, dpi, window)
        return

    if not args.result_dir:
        parser.error("非批量模式必须指定 --result-dir，或使用 --all --result-root。")

    result_dir = resolve_path(args.result_dir)
    assert result_dir is not None
    output_dir = resolve_path(args.output_dir, result_dir / "paper_figures")
    assert output_dir is not None
    draw_single_result(result_dir, output_dir, formats, dpi, window)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"绘图失败: {exc}", file=sys.stderr)
        raise
