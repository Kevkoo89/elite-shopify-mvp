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
    EPOCH_MIN_ABS = 100_000_000

    def analyze(
        self,
        df: pd.DataFrame,
        date_column: str,
        revenue_column: str,
        revenue_is_currency_text: bool = False,
    ) -> ShopifyAnalysisResult:
        work = df[[date_column, revenue_column]].copy()
        work[date_column] = self._coerce_dates(work[date_column])
        work[revenue_column] = self._coerce_revenue(
            work[revenue_column],
            force_currency_text=revenue_is_currency_text,
        )

        date_parse_ratio = float(work[date_column].notna().mean())
        if date_parse_ratio < 0.6:
            invalid_count = int(work[date_column].isna().sum())
            total_rows = int(len(work))
            raise ValueError(
                "Datumsspalte konnte nicht sicher geparst werden "
                f"({invalid_count}/{total_rows} ungueltig). "
                "Erwartete Formate: YYYY-MM-DD, DD.MM.YYYY oder Timestamp."
            )

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

        has_three_day_under_rolling = self._has_under_rolling_streak(daily, streak_days=3)
        warn_score = self._calculate_warn_score(
            last_7_total,
            previous_7_total,
            has_three_day_under_rolling=has_three_day_under_rolling,
        )
        summary = self._build_summary(
            last_7_total,
            previous_7_total,
            warn_score,
            has_three_day_under_rolling=has_three_day_under_rolling,
        )

        return ShopifyAnalysisResult(
            daily_revenue=daily,
            last_7_total=last_7_total,
            previous_7_total=previous_7_total,
            warn_score=warn_score,
            summary_text=summary,
        )

    def _coerce_dates(self, series: pd.Series) -> pd.Series:
        parsed = pd.to_datetime(series.astype(str), errors="coerce", format="mixed")
        parsed_dayfirst = pd.to_datetime(
            series.astype(str),
            errors="coerce",
            dayfirst=True,
            format="mixed",
        )
        parsed = parsed.fillna(parsed_dayfirst)

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            abs_median = float(numeric.dropna().abs().median())
            if abs_median >= self.EPOCH_MIN_ABS:
                unit = "ms" if abs_median > 10_000_000_000 else "s"
                parsed_epoch = pd.to_datetime(numeric, errors="coerce", unit=unit)
                parsed = parsed.fillna(parsed_epoch)
        return parsed

    def _coerce_revenue(self, series: pd.Series, force_currency_text: bool = False) -> pd.Series:
        if force_currency_text:
            return self._parse_currency_text(series)

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() >= 0.6:
            return numeric

        parsed_cleaned = self._parse_currency_text(series)
        return numeric.fillna(parsed_cleaned)

    def _parse_currency_text(self, series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.replace(r"[^\d,\.\-]", "", regex=True)

        def _normalize_decimal(raw: str) -> str:
            value = str(raw)
            if "," in value and "." in value:
                if value.rfind(",") > value.rfind("."):
                    return value.replace(".", "").replace(",", ".")
                return value.replace(",", "")
            if "," in value:
                return value.replace(",", ".")
            return value

        normalized = cleaned.apply(_normalize_decimal)
        return pd.to_numeric(normalized, errors="coerce")

    def _has_under_rolling_streak(self, daily: pd.DataFrame, streak_days: int) -> bool:
        under_rolling = daily["revenue"] < daily["rolling_7_avg"]
        streak = under_rolling.astype(int).rolling(streak_days, min_periods=streak_days).sum()
        return bool((streak >= streak_days).any())

    def _calculate_warn_score(
        self,
        last_7_total: float,
        previous_7_total: float,
        has_three_day_under_rolling: bool,
    ) -> float:
        if previous_7_total <= 0:
            base_score = 0.0
        else:
            percent_change = ((last_7_total - previous_7_total) / previous_7_total) * 100
            base_score = max(0.0, -percent_change)

        streak_bonus = 10.0 if has_three_day_under_rolling else 0.0
        return min(100.0, max(0.0, base_score + streak_bonus))

    def _build_summary(
        self,
        last_7_total: float,
        previous_7_total: float,
        warn_score: float,
        has_three_day_under_rolling: bool,
    ) -> str:
        streak_text = (
            " Zusaetzliches Signal aktiv: 3 Tage in Folge unter dem Rolling-7-Durchschnitt."
            if has_three_day_under_rolling
            else ""
        )
        if previous_7_total <= 0:
            return (
                "Vorperiode hat keinen gueltigen Umsatz. Warnscore wird auf 0 gesetzt, "
                "bis ein 7-Tage-Vergleich moeglich ist."
                + streak_text
            )

        percent_change = ((last_7_total - previous_7_total) / previous_7_total) * 100
        direction = "Rueckgang" if percent_change < 0 else "Anstieg"
        return (
            f"Letzte 7 Tage vs vorherige 7 Tage: {direction} von {percent_change:.2f}%. "
            f"Warnscore: {warn_score:.1f}/100."
            + streak_text
        )
