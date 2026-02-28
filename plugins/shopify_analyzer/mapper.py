from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ShopifyMappingResult:
    date_column: str
    revenue_column: str


class ShopifyCSVMapper:
    DATE_KEYWORDS = ("date", "datum", "created", "day", "time")
    REVENUE_KEYWORDS = ("revenue", "umsatz", "sales", "total", "amount", "price")

    def load_csv(self, uploaded_file) -> pd.DataFrame:
        return pd.read_csv(uploaded_file)

    def map_columns(self, df: pd.DataFrame) -> ShopifyMappingResult:
        if df.empty:
            raise ValueError("CSV ist leer.")

        date_column = self._find_date_column(df)
        revenue_column = self._find_revenue_column(df, exclude=date_column)

        if not date_column:
            raise ValueError("Keine Datumsspalte erkannt.")
        if not revenue_column:
            raise ValueError("Keine Umsatzspalte erkannt.")

        return ShopifyMappingResult(date_column=date_column, revenue_column=revenue_column)

    def _find_date_column(self, df: pd.DataFrame) -> str:
        for column in df.columns:
            name = str(column).lower()
            if any(keyword in name for keyword in self.DATE_KEYWORDS):
                return str(column)

        for column in df.columns:
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().mean() >= 0.7:
                return str(column)
        return ""

    def _find_revenue_column(self, df: pd.DataFrame, exclude: str) -> str:
        for column in df.columns:
            if str(column) == exclude:
                continue
            name = str(column).lower()
            if any(keyword in name for keyword in self.REVENUE_KEYWORDS):
                return str(column)

        for column in df.columns:
            if str(column) == exclude:
                continue
            parsed = pd.to_numeric(df[column], errors="coerce")
            if parsed.notna().mean() >= 0.7:
                return str(column)
        return ""
