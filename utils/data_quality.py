from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "HST",
    "AST",
    "HC",
    "AC",
    "Elo_Home",
    "Elo_Away",
]

DRIFT_FEATURES = ["FTHG", "FTAG", "HST", "AST", "HC", "AC", "Elo_Home", "Elo_Away"]
DEFAULT_QUALITY_WEIGHTS = {"missing": 0.6, "duplicate": 0.25, "outlier": 0.15}

LEAGUE_QUALITY_WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "premier league": {"missing": 0.62, "duplicate": 0.22, "outlier": 0.16},
    "bundesliga": {"missing": 0.64, "duplicate": 0.22, "outlier": 0.14},
    "serie a": {"missing": 0.65, "duplicate": 0.2, "outlier": 0.15},
    "la liga": {"missing": 0.63, "duplicate": 0.22, "outlier": 0.15},
    "ligue 1": {"missing": 0.61, "duplicate": 0.24, "outlier": 0.15},
}


def _quality_weights_for_league(league_name: str | None) -> dict[str, float]:
    key = str(league_name or "").strip().lower()
    if not key:
        return dict(DEFAULT_QUALITY_WEIGHTS)
    for name, weights in LEAGUE_QUALITY_WEIGHT_PRESETS.items():
        if key == name or key in name or name in key:
            return dict(weights)
    return dict(DEFAULT_QUALITY_WEIGHTS)


def compute_data_quality(df: pd.DataFrame, *, league_name: str | None = None) -> dict[str, float]:
    if df is None or df.empty:
        return {
            "score": 0.0,
            "missing_pct": 100.0,
            "duplicate_pct": 0.0,
            "outlier_pct": 0.0,
            "weights_missing": DEFAULT_QUALITY_WEIGHTS["missing"],
            "weights_duplicate": DEFAULT_QUALITY_WEIGHTS["duplicate"],
            "weights_outlier": DEFAULT_QUALITY_WEIGHTS["outlier"],
        }

    frame = df.copy()
    cols = [c for c in REQUIRED_COLUMNS if c in frame.columns]
    if not cols:
        return {
            "score": 0.0,
            "missing_pct": 100.0,
            "duplicate_pct": 0.0,
            "outlier_pct": 0.0,
            "weights_missing": DEFAULT_QUALITY_WEIGHTS["missing"],
            "weights_duplicate": DEFAULT_QUALITY_WEIGHTS["duplicate"],
            "weights_outlier": DEFAULT_QUALITY_WEIGHTS["outlier"],
        }

    missing_pct = float(frame[cols].isna().mean().mean() * 100)

    dedup_cols = [c for c in ["Date", "HomeTeam", "AwayTeam"] if c in frame.columns]
    duplicate_pct = float(frame.duplicated(subset=dedup_cols).mean() * 100) if dedup_cols else 0.0

    numeric_cols = [c for c in ["FTHG", "FTAG", "HST", "AST", "HC", "AC"] if c in frame.columns]
    outlier_pct = 0.0
    if numeric_cols:
        zsum = 0.0
        n = 0
        for col in numeric_cols:
            series = pd.to_numeric(frame[col], errors="coerce")
            std = float(series.std(skipna=True) or 0.0)
            if std <= 1e-9:
                continue
            z = (series - float(series.mean(skipna=True))) / std
            zsum += float((z.abs() > 3).mean() * 100)
            n += 1
        outlier_pct = float(zsum / n) if n else 0.0

    weights = _quality_weights_for_league(league_name)
    score = max(
        0.0,
        min(
            100.0,
            100.0
            - (
                weights["missing"] * missing_pct
                + weights["duplicate"] * duplicate_pct
                + weights["outlier"] * outlier_pct
            ),
        ),
    )
    return {
        "score": round(score, 2),
        "missing_pct": round(missing_pct, 2),
        "duplicate_pct": round(duplicate_pct, 2),
        "outlier_pct": round(outlier_pct, 2),
        "weights_missing": round(float(weights["missing"]), 3),
        "weights_duplicate": round(float(weights["duplicate"]), 3),
        "weights_outlier": round(float(weights["outlier"]), 3),
    }


def compute_feature_drift(
    df: pd.DataFrame, recent_window: int = 120, baseline_window: int = 600
) -> dict[str, object]:
    if df is None or df.empty:
        return {"score": 0.0, "status": "insufficient", "top_feature": "", "details": {}}

    frame = df.copy()
    if len(frame) < max(40, recent_window * 2):
        return {"score": 0.0, "status": "insufficient", "top_feature": "", "details": {}}

    recent = frame.tail(recent_window)
    baseline = frame.iloc[-(recent_window + baseline_window) : -recent_window]
    if baseline.empty:
        return {"score": 0.0, "status": "insufficient", "top_feature": "", "details": {}}

    drift_values: dict[str, float] = {}
    for col in DRIFT_FEATURES:
        if col not in frame.columns:
            continue
        recent_series = pd.to_numeric(recent[col], errors="coerce")
        baseline_series = pd.to_numeric(baseline[col], errors="coerce")
        baseline_std = float(baseline_series.std(skipna=True) or 0.0)
        if baseline_std <= 1e-9:
            continue

        shift = abs(
            float(recent_series.mean(skipna=True)) - float(baseline_series.mean(skipna=True))
        )
        normalized = shift / baseline_std
        drift_values[col] = round(normalized, 4)

    if not drift_values:
        return {"score": 0.0, "status": "insufficient", "top_feature": "", "details": {}}

    avg_shift = float(sum(drift_values.values()) / len(drift_values))
    score = min(100.0, avg_shift * 35.0)
    if avg_shift >= 1.4:
        status = "high"
    elif avg_shift >= 0.8:
        status = "medium"
    else:
        status = "low"

    top_feature = max(drift_values, key=drift_values.get)
    return {
        "score": round(score, 2),
        "status": status,
        "top_feature": top_feature,
        "details": drift_values,
    }


def compute_drift_speed(
    history: list[dict[str, object]],
    *,
    league: str,
    lookback_days: int = 7,
    min_points: int = 4,
) -> dict[str, object]:
    key = str(league or "").strip().lower()
    if not key:
        return {"status": "insufficient", "trend": "stable", "delta": 0.0, "streak_up": 0}

    rows: list[tuple[pd.Timestamp, float]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        if str(item.get("league", "")).strip().lower() != key:
            continue
        ts = pd.to_datetime(item.get("ts"), errors="coerce")
        if pd.isna(ts):
            continue
        score = float(item.get("score", 0.0) or 0.0)
        rows.append((ts, score))

    if len(rows) < min_points:
        return {"status": "insufficient", "trend": "stable", "delta": 0.0, "streak_up": 0}

    rows.sort(key=lambda x: x[0])
    cutoff = rows[-1][0] - pd.Timedelta(days=max(1, int(lookback_days)))
    recent = [(ts, sc) for ts, sc in rows if ts >= cutoff]
    if len(recent) < min_points:
        return {"status": "insufficient", "trend": "stable", "delta": 0.0, "streak_up": 0}

    values = [x[1] for x in recent]
    delta = float(values[-1] - values[0])
    streak_up = 0
    for prev, cur in zip(values[:-1], values[1:], strict=False):
        if cur > prev:
            streak_up += 1
        else:
            streak_up = 0

    if delta >= 8.0 or streak_up >= 3:
        trend = "up"
    elif delta <= -8.0:
        trend = "down"
    else:
        trend = "stable"

    return {
        "status": "ok",
        "trend": trend,
        "delta": round(delta, 2),
        "streak_up": int(streak_up),
        "last_score": round(float(values[-1]), 2),
    }
