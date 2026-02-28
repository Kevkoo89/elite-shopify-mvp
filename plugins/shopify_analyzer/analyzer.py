from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ShopifyAnalysisResult:
    daily_revenue: pd.DataFrame
    last_7_total: float
    previous_7_total: float
    warn_score: float
    summary_text: str


class ShopifyAnalyzer:
    def analyze(
        self,
        df: pd.DataFrame,
        date_column: str,
        revenue_column: str,
    ) -> ShopifyAnalysisResult:
        work = df[[date_column, revenue_column]].copy()
        work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
        work[revenue_column] = pd.to_numeric(work[revenue_column], errors="coerce")
        work = work.dropna(subset=[date_column, revenue_column])

        if work.empty:
            raise ValueError("Keine gueltigen Datums-/Umsatzwerte gefunden.")

        daily = (
            work.assign(date=work[date_column].dt.normalize(), revenue=work[revenue_column])
            .groupby("date", as_index=False)["revenue"]
            .sum()
            .sort_values("date")
            .reset_index(drop=True)
        )
        daily["rolling_7_avg"] = daily["revenue"].rolling(window=7, min_periods=1).mean()

        last_7_total = float(daily["revenue"].tail(7).sum())
        previous_7_slice = daily["revenue"].iloc[max(0, len(daily) - 14) : max(0, len(daily) - 7)]
        previous_7_total = float(previous_7_slice.sum())

        warn_score = self._calculate_warn_score(last_7_total, previous_7_total)
        summary = self._build_summary(last_7_total, previous_7_total, warn_score)

        return ShopifyAnalysisResult(
            daily_revenue=daily,
            last_7_total=last_7_total,
            previous_7_total=previous_7_total,
            warn_score=warn_score,
            summary_text=summary,
        )

    def _calculate_warn_score(self, last_7_total: float, previous_7_total: float) -> float:
        if previous_7_total <= 0:
            return 0.0

        percent_change = ((last_7_total - previous_7_total) / previous_7_total) * 100
        decline_percent = max(0.0, -percent_change)
        return min(100.0, decline_percent)

    def _build_summary(self, last_7_total: float, previous_7_total: float, warn_score: float) -> str:
        if previous_7_total <= 0:
            return (
                "Vorperiode hat keinen gueltigen Umsatz. Warnscore wird auf 0 gesetzt, "
                "bis ein 7-Tage-Vergleich moeglich ist."
            )

        percent_change = ((last_7_total - previous_7_total) / previous_7_total) * 100
        direction = "Rueckgang" if percent_change < 0 else "Anstieg"
        return (
            f"Letzte 7 Tage vs vorherige 7 Tage: {direction} von {percent_change:.2f}%. "
            f"Warnscore: {warn_score:.1f}/100."
        )
