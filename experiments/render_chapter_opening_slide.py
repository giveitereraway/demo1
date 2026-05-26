# -*- coding: utf-8 -*-
"""渲染章节开头 PPT 图片。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CANVAS = (1920, 1080)
FONT_DIR = Path("C:/Windows/Fonts")
BLUE = "#0f4fa8"
DEEP_BLUE = "#0b3f8f"
MID_BLUE = "#1c63c7"
LIGHT_BLUE = "#e7f0ff"
MUTED = "#5f6f86"
WHITE = "#ffffff"


def font(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    """加载中文字体，数字可用斜体增强章节感。"""
    if italic:
        return ImageFont.truetype(str(FONT_DIR / "cambriai.ttf"), size=size)
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    return ImageFont.truetype(str(FONT_DIR / name), size=size)


def rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    """十六进制颜色转 RGBA。"""
    value = color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def cover_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """按覆盖模式缩放并裁切背景。"""
    image = image.convert("RGB")
    scale = max(size[0] / image.width, size[1] / image.height)
    new_size = (int(image.width * scale), int(image.height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    left = (image.width - size[0]) // 2
    top = (image.height - size[1]) // 2
    return image.crop((left, top, left + size[0], top + size[1]))


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """计算文字尺寸。"""
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str | tuple[int, int, int, int],
) -> None:
    """在矩形内居中绘制文字。"""
    x0, y0, x1, y1 = box
    tw, th = text_size(draw, text, fnt)
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - 4), text, font=fnt, fill=fill)


def line_with_dot(draw: ImageDraw.ImageDraw, y: int, center_x: int, side: str) -> None:
    """绘制标题两侧的细线和圆点。"""
    if side == "left":
        draw.line((center_x - 360, y, center_x - 120, y), fill=BLUE, width=3)
        draw.ellipse((center_x - 128, y - 8, center_x - 112, y + 8), fill=BLUE)
    else:
        draw.line((center_x + 120, y, center_x + 360, y), fill=BLUE, width=3)
        draw.ellipse((center_x + 112, y - 8, center_x + 128, y + 8), fill=BLUE)


def render(background: Path, output: Path, background_copy: Path | None) -> None:
    """把 Image gen 底图和确定性文字合成最终页面。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    if background_copy:
        background_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(background, background_copy)

    base = cover_image(Image.open(background), CANVAS).convert("RGBA")
    overlay = Image.new("RGBA", CANVAS, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # 中心标题区域。
    center_x = CANVAS[0] // 2
    draw.rounded_rectangle((788, 320, 1132, 388), radius=10, fill=BLUE)
    center_text(draw, (788, 320, 1132, 388), "第一章", font(38, True), WHITE)

    line_with_dot(draw, 455, center_x, "left")
    line_with_dot(draw, 455, center_x, "right")

    # 大号章节编号与主标题。
    draw.rounded_rectangle((555, 510, 780, 642), radius=14, fill=rgba(BLUE, 245))
    center_text(draw, (555, 510, 780, 642), "01", font(86, bold=True, italic=True), WHITE)
    draw.rounded_rectangle((780, 510, 1370, 642), radius=8, fill=rgba(WHITE, 224), outline=rgba(MID_BLUE, 185), width=2)
    draw.rectangle((1360, 510, 1378, 642), fill=BLUE)
    draw.text((855, 530), "研究背景", font=font(76, True), fill=DEEP_BLUE)

    # 简短副标题，承担章节导入功能。
    subtitle = "飞行器智能自主决策技术的发展需求"
    center_text(draw, (500, 690, 1420, 742), subtitle, font(30), MUTED)
    draw.line((690, 766, 1230, 766), fill=rgba(MID_BLUE, 170), width=2)

    # 底部三枚关键词，和目录页的蓝白风格保持一致。
    tags = ["复杂空战环境", "自主决策需求", "强化学习方法"]
    tag_w = 250
    gap = 34
    start_x = center_x - (tag_w * 3 + gap * 2) // 2
    y = 810
    for i, tag in enumerate(tags):
        x = start_x + i * (tag_w + gap)
        draw.rounded_rectangle((x, y, x + tag_w, y + 54), radius=8, fill=rgba(LIGHT_BLUE, 232), outline=rgba(MID_BLUE, 180), width=2)
        center_text(draw, (x, y, x + tag_w, y + 54), tag, font(22, True), DEEP_BLUE)

    # 细小页脚保持低调，不干扰章节感。
    draw.text((88, 1036), "Background and Motivation", font=font(20), fill=rgba(MUTED, 150))

    image = Image.alpha_composite(base, overlay).convert("RGB")
    image.save(output, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染章节开头页")
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background-copy", type=Path, default=None)
    args = parser.parse_args()
    render(args.background, args.output, args.background_copy)


if __name__ == "__main__":
    main()
