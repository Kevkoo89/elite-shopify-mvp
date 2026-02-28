from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class DriftResult:
    drift_score: float
    severity: str
    recommended_action: str
    details: dict[str, float | str]


def _safe_hist(values: Sequence[float], bins: int = 10) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.asarray([])
    hist, _ = np.histogram(arr, bins=bins)
    probs = hist.astype(float) + 1e-8
    probs /= probs.sum()
    return probs


def compute_psi(current: Sequence[float], baseline: Sequence[float], bins: int = 10) -> float:
    c = _safe_hist(current, bins=bins)
    b = _safe_hist(baseline, bins=bins)
    if c.size == 0 or b.size == 0:
        return 0.0
    return float(np.sum((c - b) * np.log(c / b)))


def compute_js_divergence(
    current: Sequence[float], baseline: Sequence[float], bins: int = 10
) -> float:
    c = _safe_hist(current, bins=bins)
    b = _safe_hist(baseline, bins=bins)
    if c.size == 0 or b.size == 0:
        return 0.0
    m = 0.5 * (c + b)
    kl_cm = np.sum(c * np.log(c / m))
    kl_bm = np.sum(b * np.log(b / m))
    return float(0.5 * (kl_cm + kl_bm))


def ewma_zscore(series: Iterable[float], alpha: float = 0.3) -> float:
    arr = np.asarray(list(series), dtype=float)
    if arr.size < 3:
        return 0.0
    ewma = arr[0]
    for value in arr[1:]:
        ewma = (alpha * value) + ((1 - alpha) * ewma)
    std = float(np.std(arr[:-1])) or 1e-6
    mean = float(np.mean(arr[:-1]))
    return float((ewma - mean) / std)


def compute_drift_score(
    *,
    feature_current: dict[str, Sequence[float]],
    feature_baseline: dict[str, Sequence[float]],
    rolling_brier: Sequence[float],
    rolling_logloss: Sequence[float],
    rolling_accuracy: Sequence[float],
) -> DriftResult:
    feature_scores: list[float] = []
    for key, current_values in feature_current.items():
        baseline_values = feature_baseline.get(key, [])
        psi = compute_psi(current_values, baseline_values)
        jsd = compute_js_divergence(current_values, baseline_values)
        feature_scores.append(min(1.0, (psi / 0.25) * 0.6 + (jsd / 0.1) * 0.4))

    feature_drift = float(np.mean(feature_scores)) if feature_scores else 0.0

    z_brier = max(0.0, ewma_zscore(rolling_brier))
    z_logloss = max(0.0, ewma_zscore(rolling_logloss))
    z_acc = max(0.0, -ewma_zscore(rolling_accuracy))
    perf_drift = min(1.0, (z_brier + z_logloss + z_acc) / 9.0)

    combined = max(0.0, min(1.0, (0.55 * feature_drift) + (0.45 * perf_drift)))
    drift_score = round(combined * 100, 2)

    if drift_score >= 75:
        severity = "critical"
        action = "enable safe mode"
    elif drift_score >= 50:
        severity = "warn"
        action = "increase evidence"
    else:
        severity = "ok"
        action = "tighten thresholds"

    return DriftResult(
        drift_score=drift_score,
        severity=severity,
        recommended_action=action,
        details={
            "feature_drift": round(feature_drift, 4),
            "performance_drift": round(perf_drift, 4),
            "z_brier": round(z_brier, 4),
            "z_logloss": round(z_logloss, 4),
            "z_accuracy": round(z_acc, 4),
        },
    )
