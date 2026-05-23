from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tactical_actions import ACTION_BY_ID, coerce_action_id


@dataclass(frozen=True)
class TacticalSafetyState:
    altitude_m: float | None = None
    altitude_limit_m: float = 2500.0
    distance_m: float | None = None
    speed_mps: float | None = None
    ao_rad: float | None = None
    ta_rad: float | None = None


@dataclass(frozen=True)
class TacticalSafetyResult:
    action_id: int
    original_action_id: int | None
    overridden: bool
    reason: str = ""


def state_from_obs(obs: Any, *, altitude_limit_m: float = 2500.0) -> TacticalSafetyState:
    try:
        values = list(obs)
    except TypeError:
        return TacticalSafetyState(altitude_limit_m=altitude_limit_m)

    altitude_m = float(values[0]) * 5000.0 if len(values) > 0 else None
    speed_mps = float(values[5]) * 340.0 if len(values) > 5 else None
    ao_rad = float(values[11]) if len(values) > 11 else None
    ta_rad = float(values[12]) if len(values) > 12 else None
    distance_m = float(values[13]) * 10000.0 if len(values) > 13 else None
    return TacticalSafetyState(
        altitude_m=altitude_m,
        altitude_limit_m=altitude_limit_m,
        distance_m=distance_m,
        speed_mps=speed_mps,
        ao_rad=ao_rad,
        ta_rad=ta_rad,
    )


def state_from_env(env: Any, agent_id: str) -> TacticalSafetyState:
    altitude_limit = float(getattr(getattr(env, "config", object()), "altitude_limit", 2500.0))
    try:
        obs = env.task.get_obs(env, agent_id)
    except Exception:
        return TacticalSafetyState(altitude_limit_m=altitude_limit)
    return state_from_obs(obs, altitude_limit_m=altitude_limit)


def _safe_fallback(fallback_action_id: int | None, default_action_id: int = 0) -> int:
    action_id = coerce_action_id(fallback_action_id)
    if action_id is not None:
        return action_id
    return default_action_id


def apply_tactical_safety(
    action_id: int | None,
    *,
    state: TacticalSafetyState | None = None,
    fallback_action_id: int | None = None,
    low_altitude_margin_m: float = 1000.0,
    close_distance_m: float = 1200.0,
    high_speed_mps: float = 260.0,
) -> TacticalSafetyResult:
    original_action_id = coerce_action_id(action_id)
    if original_action_id is None:
        fallback = _safe_fallback(fallback_action_id)
        return TacticalSafetyResult(fallback, None, True, "动作编号非法，回退到备用动作。")

    state = state or TacticalSafetyState()
    final_action_id = original_action_id
    reason = ""

    # 低空附近禁止继续俯冲，避免 LLM 或 actor 把飞机推向低高度终止线。
    if (
        state.altitude_m is not None
        and state.altitude_m <= state.altitude_limit_m + low_altitude_margin_m
        and original_action_id in (5, 11)
    ):
        final_action_id = 4
        reason = "高度接近低空阈值，俯冲类动作改为爬升占位。"

    # 近距且速度较高时避免纯追/提前追继续压向目标，优先减速防止过冲。
    elif (
        state.distance_m is not None
        and state.speed_mps is not None
        and state.distance_m <= close_distance_m
        and state.speed_mps >= high_speed_mps
        and original_action_id in (0, 1)
    ):
        final_action_id = 7
        reason = "近距高速追击存在过冲风险，改为平飞减速。"

    # 极端不利姿态下保留防御/脱离类动作；若仍要求进攻，先脱离重整。
    elif (
        state.ao_rad is not None
        and state.ta_rad is not None
        and state.ao_rad > 2.4
        and state.ta_rad < 0.8
        and original_action_id in (0, 1, 5, 11)
    ):
        final_action_id = 3
        reason = "当前姿态不利，进攻动作改为脱离。"

    if final_action_id not in ACTION_BY_ID:
        fallback = _safe_fallback(fallback_action_id)
        return TacticalSafetyResult(fallback, original_action_id, True, "安全覆盖结果非法，回退到备用动作。")

    return TacticalSafetyResult(
        final_action_id,
        original_action_id,
        final_action_id != original_action_id,
        reason,
    )
