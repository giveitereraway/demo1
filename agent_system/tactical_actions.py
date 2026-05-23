from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TacticalAction:
    action_id: int
    code: str
    chinese_name: str
    description: str
    aliases: tuple[str, ...]


TACTICAL_ACTIONS: tuple[TacticalAction, ...] = (
    TacticalAction(
        0,
        "PURE_PURSUIT",
        "纯追击",
        "直接指向敌机当前位置，快速压向目标。",
        ("追击", "追敌", "直接追", "纯追", "咬住敌机", "pure pursuit"),
    ),
    TacticalAction(
        1,
        "LEAD_PURSUIT",
        "提前量追击",
        "指向敌机短时预测位置，争取提前占位。",
        ("提前量", "抢占前方", "抢到前方", "敌机前方", "提前追击", "lead pursuit"),
    ),
    TacticalAction(
        2,
        "LAG_PURSUIT",
        "滞后追击",
        "瞄准敌机后方位置，降低过冲风险。",
        ("滞后", "跟在后面", "跟住后方", "别冲过头", "尾随", "lag pursuit"),
    ),
    TacticalAction(
        3,
        "DISENGAGE",
        "脱离",
        "背离敌机并加速拉开距离。",
        ("脱离", "拉开距离", "退出交战", "先撤", "disengage"),
    ),
    TacticalAction(
        4,
        "CLIMB_POSITION",
        "爬升占位",
        "保持对敌压力，同时换取高度优势。",
        ("爬升", "占高度", "高度优势", "高处占位", "climb"),
    ),
    TacticalAction(
        5,
        "DIVE_ACCELERATE",
        "俯冲加速",
        "向敌机方向压低高度并恢复能量。",
        ("俯冲", "下压", "加速俯冲", "dive"),
    ),
    TacticalAction(
        6,
        "LEVEL_ACCELERATE",
        "平飞加速",
        "保持当前航向并提升速度。",
        ("平飞加速", "保持平飞加速", "加速", "level accelerate"),
    ),
    TacticalAction(
        7,
        "LEVEL_DECELERATE",
        "平飞减速",
        "降低速度，缓解近距离过冲。",
        ("减速", "降低速度", "别冲过头", "避免冲过头", "level decelerate"),
    ),
    TacticalAction(
        8,
        "DEFENSIVE_TURN_LEFT",
        "左防御转弯",
        "基于当前航向向左转，用于防御机动。",
        ("左防御", "向左防御", "左转防御", "左转弯", "turn left"),
    ),
    TacticalAction(
        9,
        "DEFENSIVE_TURN_RIGHT",
        "右防御转弯",
        "基于当前航向向右转，用于防御机动。",
        ("右防御", "向右防御", "右转防御", "右转弯", "turn right"),
    ),
    TacticalAction(
        10,
        "HIGH_YOYO",
        "高悠悠",
        "偏向敌机、爬升、减速，换取高度和角度优势。",
        ("高悠悠", "高yo", "高 yo", "high yoyo", "high yo-yo"),
    ),
    TacticalAction(
        11,
        "LOW_YOYO",
        "低悠悠",
        "偏向敌机、俯冲、加速，先换速度再转入进攻。",
        ("低悠悠", "低yo", "低 yo", "low yoyo", "low yo-yo"),
    ),
)

ACTION_BY_ID: dict[int, TacticalAction] = {item.action_id: item for item in TACTICAL_ACTIONS}
ACTION_BY_CODE: dict[str, TacticalAction] = {item.code: item for item in TACTICAL_ACTIONS}


def normalize_action_name(name: str | None) -> str:
    return str(name or "").strip().replace("-", "_").replace(" ", "_").upper()


def is_valid_action_id(action_id: object) -> bool:
    try:
        return int(action_id) in ACTION_BY_ID
    except (TypeError, ValueError):
        return False


def action_name(action_id: int) -> str:
    return ACTION_BY_ID[int(action_id)].code


def action_chinese_name(action_id: int) -> str:
    return ACTION_BY_ID[int(action_id)].chinese_name


def action_name_matches(action_id: int, name: str | None) -> bool:
    if not is_valid_action_id(action_id):
        return False
    action = ACTION_BY_ID[int(action_id)]
    normalized = normalize_action_name(name)
    return normalized == action.code or str(name or "").strip() == action.chinese_name


def coerce_action_id(value: object) -> int | None:
    try:
        action_id = int(value)
    except (TypeError, ValueError):
        return None
    return action_id if action_id in ACTION_BY_ID else None


def parse_action_reference(value: str | int) -> int | None:
    action_id = coerce_action_id(value)
    if action_id is not None:
        return action_id

    text = str(value).strip()
    normalized = normalize_action_name(text)
    if normalized in ACTION_BY_CODE:
        return ACTION_BY_CODE[normalized].action_id

    for action in TACTICAL_ACTIONS:
        if text == action.chinese_name:
            return action.action_id
    return None
