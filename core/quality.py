from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .validation import LiveInputPayload, validate_live_input_payload


@dataclass(frozen=True)
class InputValidationFuzzReport:
    rounds: int
    exception_count: int
    invalid_payload_count: int
    stable: bool


@dataclass(frozen=True)
class ExportContractReport:
    valid: bool
    missing_columns: list[str]
    wrong_order: bool


@dataclass(frozen=True)
class CacheInvalidationReport:
    should_invalidate: bool
    reasons: list[str]


def run_input_validation_fuzz(rounds: int = 120, seed: int = 42) -> InputValidationFuzzReport:
    rng = random.Random(seed)
    exception_count = 0
    invalid_count = 0
    for _ in range(max(1, int(rounds))):
        payload = LiveInputPayload(
            home_team=rng.choice(["A", " ", "X" * 40]),
            away_team=rng.choice(["B", " ", "Y" * 40]),
            market_quote=rng.choice([0.0, 1.2, 2.4, 30.0]),
            bankroll=rng.choice([0.0, 10.0, 1000.0, -5.0]),
            min_confidence=rng.choice([0, 39, 65, 95, 120]),
            sim_count=rng.choice([0, 999, 10000, 210000]),
            weight_poisson=rng.choice([-0.1, 0.0, 0.5, 1.0]),
            weight_elo=rng.choice([-0.1, 0.0, 0.3, 1.0]),
            weight_ml=rng.choice([-0.1, 0.0, 0.2, 1.0]),
        )
        try:
            errors = validate_live_input_payload(payload)
        except Exception:
            exception_count += 1
            continue
        if errors:
            invalid_count += 1
    return InputValidationFuzzReport(
        rounds=max(1, int(rounds)),
        exception_count=exception_count,
        invalid_payload_count=invalid_count,
        stable=exception_count == 0,
    )


def validate_export_contract(
    df: pd.DataFrame,
    required_columns: list[str],
    ordered_columns: list[str] | None = None,
) -> ExportContractReport:
    required = [c for c in required_columns if c]
    missing = [c for c in required if c not in df.columns]
    wrong_order = False
    if ordered_columns:
        present_order = [c for c in ordered_columns if c in df.columns]
        actual_prefix = list(df.columns[: len(present_order)])
        wrong_order = actual_prefix != present_order
    return ExportContractReport(
        valid=not missing and not wrong_order, missing_columns=missing, wrong_order=wrong_order
    )


def evaluate_cache_invalidation(
    *,
    now: datetime,
    last_sync_at_iso: str,
    cache_schema_version: str,
    active_schema_version: str,
    max_age_minutes: int = 30,
) -> CacheInvalidationReport:
    reasons: list[str] = []
    try:
        last_sync = datetime.fromisoformat(str(last_sync_at_iso))
    except ValueError:
        reasons.append("invalid_last_sync_ts")
        return CacheInvalidationReport(should_invalidate=True, reasons=reasons)

    age_minutes = (now - last_sync).total_seconds() / 60.0
    if age_minutes > max(1, int(max_age_minutes)):
        reasons.append("stale_by_age")
    if str(cache_schema_version) != str(active_schema_version):
        reasons.append("schema_changed")

    return CacheInvalidationReport(should_invalidate=bool(reasons), reasons=reasons)
