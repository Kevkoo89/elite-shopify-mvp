from __future__ import annotations

import math
from dataclasses import dataclass


def _normalize_team_name(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _coerce_float(value: float, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if not math.isfinite(out) else out


def _coerce_int(value: int, default: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return int(default)
    return int(default) if not math.isfinite(float(out)) else out


@dataclass(frozen=True)
class LiveInputPayload:
    home_team: str
    away_team: str
    market_quote: float
    bankroll: float
    min_confidence: int
    sim_count: int
    weight_poisson: float
    weight_elo: float
    weight_ml: float


def with_deterministic_defaults(payload: LiveInputPayload) -> LiveInputPayload:
    return LiveInputPayload(
        home_team=_normalize_team_name(payload.home_team),
        away_team=_normalize_team_name(payload.away_team),
        market_quote=_coerce_float(payload.market_quote, 1.01),
        bankroll=_coerce_float(payload.bankroll, 0.0),
        min_confidence=_coerce_int(payload.min_confidence, 60),
        sim_count=_coerce_int(payload.sim_count, 10000),
        weight_poisson=_coerce_float(payload.weight_poisson, 0.0),
        weight_elo=_coerce_float(payload.weight_elo, 0.0),
        weight_ml=_coerce_float(payload.weight_ml, 0.0),
    )


def validate_live_input_payload(payload: LiveInputPayload) -> list[str]:
    payload = with_deterministic_defaults(payload)
    errors: list[str] = []

    if not payload.home_team or not payload.away_team:
        errors.append("teams_required")
    elif payload.home_team == payload.away_team:
        errors.append("teams_must_differ")

    if payload.market_quote < 1.01 or payload.market_quote > 25:
        errors.append("quote_range")

    if payload.bankroll <= 0:
        errors.append("bankroll_positive")

    if payload.min_confidence < 40 or payload.min_confidence > 95:
        errors.append("confidence_range")

    if payload.sim_count < 1000 or payload.sim_count > 200000:
        errors.append("sim_count_range")

    total_weight = payload.weight_poisson + payload.weight_elo + payload.weight_ml
    if total_weight <= 0:
        errors.append("weights_positive")

    for weight in (payload.weight_poisson, payload.weight_elo, payload.weight_ml):
        if weight < 0:
            errors.append("weights_non_negative")
            break

    return errors
