from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Any


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if isnan(result) else result


def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


@dataclass(frozen=True)
class TacticalSituation:
    """当前 1v1 战术态势摘要，供解析、计划、日志和安全覆盖共用。"""

    altitude_m: float | None = None
    altitude_limit_m: float = 2500.0
    speed_mps: float | None = None
    delta_altitude_m: float | None = None
    distance_m: float | None = None
    ao_rad: float | None = None
    ta_rad: float | None = None
    side_flag: int | None = None
    low_altitude_margin_m: float = 1000.0
    close_distance_m: float = 1200.0
    high_speed_mps: float = 260.0
    altitude_advantage_m: float = 300.0
    close_range_m: float = 2000.0
    opened_range_m: float = 3000.0

    @property
    def low_altitude_risk(self) -> bool:
        return self.altitude_m is not None and self.altitude_m <= self.altitude_limit_m + self.low_altitude_margin_m

    @property
    def close_fast_risk(self) -> bool:
        return (
            self.distance_m is not None
            and self.speed_mps is not None
            and self.distance_m <= self.close_distance_m
            and self.speed_mps >= self.high_speed_mps
        )

    @property
    def bad_posture_risk(self) -> bool:
        return self.ao_rad is not None and self.ta_rad is not None and self.ao_rad > 2.4 and self.ta_rad < 0.8

    @property
    def has_altitude_advantage(self) -> bool:
        return self.delta_altitude_m is not None and self.delta_altitude_m <= -self.altitude_advantage_m

    @property
    def range_is_close(self) -> bool:
        return self.distance_m is not None and self.distance_m <= self.close_range_m

    @property
    def range_is_opened(self) -> bool:
        return self.distance_m is not None and self.distance_m >= self.opened_range_m

    @property
    def side_label(self) -> str:
        if self.side_flag is None:
            return "未知"
        if self.side_flag > 0:
            return "左侧"
        if self.side_flag < 0:
            return "右侧"
        return "正前/正后"

    def matches_until(self, until: str) -> bool:
        key = str(until or "fixed_steps").strip()
        if key == "range_close":
            return self.range_is_close
        if key == "range_opened":
            return self.range_is_opened
        if key == "altitude_advantage":
            return self.has_altitude_advantage
        if key == "overshoot_risk_reduced":
            return not self.close_fast_risk
        if key == "bad_posture_recovered":
            return not self.bad_posture_risk
        return False

    def to_prompt_text(self) -> str:
        def fmt(value: float | None, unit: str = "") -> str:
            return "未知" if value is None else f"{value:.1f}{unit}"

        risk_labels = []
        if self.low_altitude_risk:
            risk_labels.append("低空风险")
        if self.close_fast_risk:
            risk_labels.append("近距高速过冲风险")
        if self.bad_posture_risk:
            risk_labels.append("不利姿态进攻风险")
        if not risk_labels:
            risk_labels.append("未触发显著风险")

        return (
            "当前态势摘要："
            f"己方高度={fmt(self.altitude_m, 'm')}，"
            f"安全高度下限={self.altitude_limit_m:.1f}m，"
            f"己方速度={fmt(self.speed_mps, 'm/s')}，"
            f"敌我高度差(敌-我)={fmt(self.delta_altitude_m, 'm')}，"
            f"敌我距离={fmt(self.distance_m, 'm')}，"
            f"AO={fmt(self.ao_rad, 'rad')}，"
            f"TA={fmt(self.ta_rad, 'rad')}，"
            f"敌机方位={self.side_label}，"
            f"风险标签={','.join(risk_labels)}。"
        )

    def to_log(self) -> dict[str, object]:
        return {
            "altitude_m": _round_or_none(self.altitude_m, 2),
            "altitude_limit_m": _round_or_none(self.altitude_limit_m, 2),
            "speed_mps": _round_or_none(self.speed_mps, 2),
            "delta_altitude_m": _round_or_none(self.delta_altitude_m, 2),
            "distance_m": _round_or_none(self.distance_m, 2),
            "ao_rad": _round_or_none(self.ao_rad, 4),
            "ta_rad": _round_or_none(self.ta_rad, 4),
            "side_flag": self.side_flag,
            "side_label": self.side_label,
            "low_altitude_risk": self.low_altitude_risk,
            "close_fast_risk": self.close_fast_risk,
            "bad_posture_risk": self.bad_posture_risk,
            "altitude_advantage": self.has_altitude_advantage,
            "range_close": self.range_is_close,
            "range_opened": self.range_is_opened,
        }

    def to_safety_state(self):
        from .tactical_safety import TacticalSafetyState

        return TacticalSafetyState(
            altitude_m=self.altitude_m,
            altitude_limit_m=self.altitude_limit_m,
            distance_m=self.distance_m,
            speed_mps=self.speed_mps,
            ao_rad=self.ao_rad,
            ta_rad=self.ta_rad,
        )


def situation_from_obs(obs: Any, *, altitude_limit_m: float = 2500.0) -> TacticalSituation:
    try:
        values = list(obs)
    except TypeError:
        return TacticalSituation(altitude_limit_m=altitude_limit_m)

    altitude_m = _finite_float(values[0]) * 5000.0 if len(values) > 0 and _finite_float(values[0]) is not None else None
    speed_mps = _finite_float(values[5]) * 340.0 if len(values) > 5 and _finite_float(values[5]) is not None else None
    delta_altitude_m = _finite_float(values[10]) * 1000.0 if len(values) > 10 and _finite_float(values[10]) is not None else None
    ao_rad = _finite_float(values[11]) if len(values) > 11 else None
    ta_rad = _finite_float(values[12]) if len(values) > 12 else None
    distance_m = _finite_float(values[13]) * 10000.0 if len(values) > 13 and _finite_float(values[13]) is not None else None
    raw_side = _finite_float(values[14]) if len(values) > 14 else None
    side_flag = int(raw_side) if raw_side is not None else None
    return TacticalSituation(
        altitude_m=altitude_m,
        altitude_limit_m=altitude_limit_m,
        speed_mps=speed_mps,
        delta_altitude_m=delta_altitude_m,
        distance_m=distance_m,
        ao_rad=ao_rad,
        ta_rad=ta_rad,
        side_flag=side_flag,
    )


def situation_from_env(env: Any, agent_id: str) -> TacticalSituation:
    altitude_limit = float(getattr(getattr(env, "config", object()), "altitude_limit", 2500.0))
    try:
        obs = env.task.get_obs(env, agent_id)
    except Exception:
        return TacticalSituation(altitude_limit_m=altitude_limit)
    return situation_from_obs(obs, altitude_limit_m=altitude_limit)
