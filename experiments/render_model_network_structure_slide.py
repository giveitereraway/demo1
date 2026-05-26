# -*- coding: utf-8 -*-
"""渲染用于论文/PPT展示的模型神经网络结构图。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


CANVAS = (1920, 1080)
FONT_DIR = Path("C:/Windows/Fonts")

BLUE = "#0f4fa8"
DEEP_BLUE = "#0b3e86"
LIGHT_BLUE = "#eaf2ff"
MID_BLUE = "#4d7cc9"
RED = "#c94b55"
GREEN = "#2f8f70"
GOLD = "#d29a20"
INK = "#1b2b45"
MUTED = "#5d6b82"
GRID = "#d9e2f2"
WHITE = "#ffffff"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """加载微软雅黑字体，保证中文在 Windows 上稳定显示。"""
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    return ImageFont.truetype(str(FONT_DIR / name), size=size)


def rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    """把十六进制颜色转换为 RGBA。"""
    color = hex_color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def cover_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """按覆盖模式缩放并居中裁剪背景图。"""
    image = image.convert("RGB")
    scale = max(size[0] / image.width, size[1] / image.height)
    new_size = (int(image.width * scale), int(image.height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    left = (image.width - size[0]) // 2
    top = (image.height - size[1]) // 2
    return image.crop((left, top, left + size[0], top + size[1]))


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """计算单行文字尺寸。"""
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str | tuple[int, int, int, int],
) -> None:
    """在指定矩形中居中绘制单行文字。"""
    x0, y0, x1, y1 = box
    tw, th = text_size(draw, text, fnt)
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - 2), text, font=fnt, fill=fill)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """按像素宽度对中英文混排文本做简单换行。"""
    lines: list[str] = []
    current = ""
    for ch in text:
        candidate = current + ch
        if text_size(draw, candidate, fnt)[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    pos: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    line_gap: int = 6,
) -> int:
    """绘制自动换行文本，返回绘制后的 y 坐标。"""
    x, y = pos
    for line in wrap_text(draw, text, fnt, max_width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += text_size(draw, line, fnt)[1] + line_gap
    return y


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = MID_BLUE,
    width: int = 4,
) -> None:
    """绘制带箭头的连接线。"""
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    dx = x1 - x0
    dy = y1 - y0
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = 14
    wing = 7
    points = [
        (x1, y1),
        (x1 - ux * head + px * wing, y1 - uy * head + py * wing),
        (x1 - ux * head - px * wing, y1 - uy * head - py * wing),
    ]
    draw.polygon(points, fill=color)


def panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    accent: str,
    subtitle: str | None = None,
) -> None:
    """绘制内容面板。"""
    draw.rounded_rectangle(box, radius=8, fill=rgba(WHITE, 236), outline=rgba(accent, 190), width=2)
    x0, y0, x1, _ = box
    draw.rectangle((x0, y0, x1, y0 + 8), fill=accent)
    draw.text((x0 + 26, y0 + 22), title, font=font(32, True), fill=accent)
    if subtitle:
        draw.text((x0 + 26, y0 + 64), subtitle, font=font(18), fill=MUTED)


def layer_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    note: str,
    fill: str,
    outline: str,
) -> None:
    """绘制网络层块。"""
    draw.rounded_rectangle(box, radius=7, fill=fill, outline=outline, width=2)
    x0, y0, x1, y1 = box
    draw_center_text(draw, (x0 + 10, y0 + 6, x1 - 10, y0 + 34), label, font(20, True), INK)
    draw_center_text(draw, (x0 + 10, y0 + 35, x1 - 10, y1 - 5), note, font(13), MUTED)


def draw_stack(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    layers: Sequence[tuple[str, str, str]],
    accent: str,
) -> None:
    """绘制纵向网络结构栈。"""
    h = 60
    gap = 13
    for idx, (label, note, fill) in enumerate(layers):
        top = y + idx * (h + gap)
        layer_box(draw, (x, top, x + w, top + h), label, note, fill, accent)
        if idx < len(layers) - 1:
            arrow(draw, (x + w // 2, top + h + 2), (x + w // 2, top + h + gap - 3), accent, 3)


def draw_small_badge(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: str) -> None:
    """绘制小标签。"""
    draw.rounded_rectangle(box, radius=7, fill=fill)
    draw_center_text(draw, box, text, font(18, True), WHITE)


def draw_table(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    """绘制高层/底层模型差异表。"""
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill=rgba(WHITE, 242), outline=rgba(BLUE, 190), width=2)
    draw.rectangle((x0, y0, x1, y0 + 46), fill=BLUE)
    draw.text((x0 + 26, y0 + 9), "高层模型与底层模型的结构区别", font=font(27, True), fill=WHITE)
    col_w = [190, 420, 430, 736]
    headers = ["对象", "输入", "输出动作空间", "角色与训练状态"]
    xs = [x0]
    for width in col_w:
        xs.append(xs[-1] + width)
    header_y = y0 + 58
    row_h = 48
    for i, header in enumerate(headers):
        draw.rectangle((xs[i], header_y, xs[i + 1], header_y + 34), fill=rgba(LIGHT_BLUE, 255), outline=GRID)
        draw_center_text(draw, (xs[i], header_y, xs[i + 1], header_y + 34), header, font(17, True), DEEP_BLUE)
    rows = [
        (
            "高层 Actor",
            "15维普通空战；21维导弹任务",
            "Discrete(12) 或 Tuple(12类战术, 2类发射)",
            "参与 PPO 训练，学习战术/发射决策；训练时配套 Critic 估计 V(s)",
        ),
        (
            "底层 Actor",
            "12维 = Δh/Δψ/Δv + 本机9维状态",
            "MultiDiscrete([41,41,41,30])",
            "加载 actor_heading.pt 并冻结，只把高层意图转成舵面和油门",
        ),
    ]
    for r, row in enumerate(rows):
        top = header_y + 34 + r * row_h
        bg = rgba("#f8fbff" if r == 0 else "#eef4ff", 255)
        for c in range(4):
            draw.rectangle((xs[c], top, xs[c + 1], top + row_h), fill=bg, outline=GRID)
            content = row[c]
            fnt = font(18, True) if c == 0 else font(14)
            fill = accent_for_row(row[0]) if c == 0 else INK
            draw_wrapped(draw, (xs[c] + 14, top + 11), content, fnt, fill, col_w[c] - 28, 4)
    note = "结论：高层和底层可使用相同 PPOActor 骨架；区别主要来自输入语义、动作空间和底层 actor 的冻结执行角色。"
    draw.text((x0 + 26, y1 - 31), note, font=font(18, True), fill=DEEP_BLUE)


def accent_for_row(name: str) -> str:
    """给不同表格行返回强调色。"""
    return RED if "高层" in name else GREEN


def draw_pipeline(draw: ImageDraw.ImageDraw) -> None:
    """绘制高层到 JSBSim 的执行链路。"""
    y = 668
    x = 686
    items = [
        ("高层 Actor", "12类战术 + 发射标志", RED),
        ("动作翻译", "Δh / Δψ / Δv", GOLD),
        ("底层 Actor", "actor_heading.pt", GREEN),
        ("飞控执行", "舵面 + 油门", BLUE),
        ("JSBSim", "物理仿真闭环", DEEP_BLUE),
    ]
    w = 100
    h = 58
    gap = 15
    for idx, (title, note, color) in enumerate(items):
        left = x + idx * (w + gap)
        draw.rounded_rectangle((left, y, left + w, y + h), radius=8, fill=rgba(color, 235), outline=color, width=2)
        draw_center_text(draw, (left + 5, y + 6, left + w - 5, y + 31), title, font(15, True), WHITE)
        draw_center_text(draw, (left + 5, y + 32, left + w - 5, y + h - 3), note, font(10), WHITE)
        if idx < len(items) - 1:
            arrow(draw, (left + w + 6, y + h // 2), (left + w + gap - 8, y + h // 2), MID_BLUE, 4)


def render(background: Path, output: Path, background_copy: Path | None) -> None:
    """组合底图和结构图，输出最终 PNG。"""
    if background_copy:
        background_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(background, background_copy)

    base = cover_image(Image.open(background), CANVAS).convert("RGBA")
    overlay = Image.new("RGBA", CANVAS, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # 主标题区采用确定性文字叠加，避免图像模型生成错字。
    draw.rounded_rectangle((710, 132, 1210, 184), radius=8, fill=BLUE)
    draw_center_text(draw, (710, 132, 1210, 184), "第三章  分层强化学习", font(31, True), WHITE)
    draw.text((512, 202), "策略模型、价值模型与分层网络结构", font=font(58, True), fill=DEEP_BLUE)
    draw.text((596, 278), "PPO Actor-Critic：高层战术决策 + 冻结底层航向控制器", font=font(25), fill=MUTED)
    draw.line((432, 255, 512, 255), fill=BLUE, width=3)
    draw.ellipse((420, 247, 436, 263), fill=BLUE)
    draw.line((1408, 255, 1488, 255), fill=BLUE, width=3)
    draw.ellipse((1392, 247, 1408, 263), fill=BLUE)

    # Actor 面板。
    actor_box = (72, 340, 610, 808)
    panel(draw, actor_box, "策略模型 Actor", RED, "PPOActor：输出动作分布与 logπ(a|s)")
    actor_layers = [
        ("观测输入 obs", "15维普通空战 / 21维导弹任务", "#fff3f4"),
        ("MLPBase", "Linear + ReLU + LayerNorm ×2；128,128", "#fff8f8"),
        ("GRU", "1层循环记忆；hidden=128", "#fff8f1"),
        ("ACTLayer", "动作头前 MLP；128,128", "#f5fbff"),
        ("动作分布", "Categorical / MultiDiscrete / Shoot Bernoulli", "#eef7ff"),
    ]
    draw_stack(draw, 128, 430, 426, actor_layers, RED)

    # Critic 面板。
    critic_box = (1310, 340, 1848, 808)
    panel(draw, critic_box, "价值模型 Critic", BLUE, "PPOCritic：输出状态价值 V(s)")
    critic_layers = [
        ("观测输入 obs", "PPO用本地观测；MAPPO用集中观测", "#f1f7ff"),
        ("MLPBase", "Linear + ReLU + LayerNorm ×2；128,128", "#f7fbff"),
        ("GRU", "1层循环记忆；hidden=128", "#f5fbff"),
        ("Value MLP", "价值头前 MLP；128,128", "#eef7ff"),
        ("Linear(128,1)", "输出标量 V(s)", "#e7f1ff"),
    ]
    draw_stack(draw, 1366, 430, 426, critic_layers, BLUE)

    # 中央共享说明。
    center = (675, 375, 1245, 638)
    draw.rounded_rectangle(center, radius=8, fill=rgba("#f8fbff", 246), outline=rgba(DEEP_BLUE, 185), width=2)
    draw.text((715, 400), "共享骨架", font=font(34, True), fill=DEEP_BLUE)
    bullets = [
        "MLPBase 先把观测展平并编码成 128 维特征",
        "GRU 处理时序信息，episode 结束时由 mask 清空隐状态",
        "Actor 与 Critic 分开建模、同一个 Adam 优化器联合更新",
        "导弹任务可用 use_prior 调整发射分布先验",
    ]
    yy = 456
    for b in bullets:
        draw.ellipse((720, yy + 8, 730, yy + 18), fill=MID_BLUE)
        yy = draw_wrapped(draw, (742, yy), b, font(20), INK, 452, 8) + 2
    arrow(draw, (610, 560), (675, 510), RED, 4)
    arrow(draw, (1310, 560), (1245, 510), BLUE, 4)

    # 高层/底层差异表和执行流水线。
    draw_table(draw, (72, 812, 1848, 1048))
    draw_pipeline(draw)

    # 页脚来源。
    footer = "Source: algorithms/ppo/*, algorithms/utils/*, envs/JSBSim/tasks/*  |  default hidden=128,128; GRU=128×1"
    draw.text((92, 1056), footer, font=font(15), fill=MUTED)

    image = Image.alpha_composite(base, overlay).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="渲染模型网络结构 PPT 图片")
    parser.add_argument("--background", type=Path, required=True, help="Image gen 生成的背景图")
    parser.add_argument("--output", type=Path, required=True, help="最终输出 PNG")
    parser.add_argument("--background-copy", type=Path, default=None, help="可选：把背景图复制到工作区")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.background, args.output, args.background_copy)


if __name__ == "__main__":
    main()
