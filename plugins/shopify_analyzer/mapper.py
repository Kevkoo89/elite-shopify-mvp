from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


@dataclass(frozen=True)
class ShopifyMappingResult:
    date_column: str
    revenue_column: str
    mapping_unsure: bool = False
    date_candidates: tuple[str, ...] = ()
    revenue_candidates: tuple[str, ...] = ()
    detected_export_type: str = "Unklar"
    selected_export_type: str = "Auto (empfohlen)"
    available_columns: tuple[str, ...] = ()
    requires_manual_selection: bool = False


class ShopifyCSVMapper:
    AUTO_EXPORT_OPTION = "Auto (empfohlen)"
    ADMIN_EXPORT_OPTION = "Shopify Admin Export"
    REPORTS_EXPORT_OPTION = "Shopify Analytics/Reports Export"
    UNKNOWN_EXPORT_LABEL = "Unklar"

    ADMIN_DATE_CANDIDATES = (
        "created_at",
        "order_created_at",
        "processed_at",
        "paid_at",
        "date",
        "created",
    )
    ADMIN_REVENUE_CANDIDATES = (
        "total_price",
        "total",
        "total_sales",
        "gross_sales",
        "net_sales",
        "subtotal",
        "price",
        "amount",
    )
    REPORTS_DATE_CANDIDATES = (
        "date",
        "day",
        "order_date",
        "report_date",
    )
    REPORTS_REVENUE_CANDIDATES = (
        "revenue",
        "sales",
        "total_sales",
        "gross_sales",
        "net_sales",
    )

    def load_csv(self, uploaded_file) -> pd.DataFrame:
        return pd.read_csv(uploaded_file)

    def map_columns(
        self,
        df: pd.DataFrame,
        export_type: str = AUTO_EXPORT_OPTION,
    ) -> ShopifyMappingResult:
        if df.empty:
            raise ValueError("CSV ist leer.")

        columns = tuple(str(column) for column in df.columns)
        detected_export_type = self.detect_export_type(columns)
        selected_export_type = (
            export_type
            if export_type in self.available_export_types()
            else self.AUTO_EXPORT_OPTION
        )

        base_export_type = (
            detected_export_type
            if selected_export_type == self.AUTO_EXPORT_OPTION
            else selected_export_type
        )

        date_aliases, revenue_aliases = self._aliases_for_export(base_export_type)
        date_candidates = self._find_candidates(columns, date_aliases)
        revenue_candidates = self._find_candidates(columns, revenue_aliases)

        date_column = date_candidates[0] if date_candidates else ""
        revenue_column = self._first_revenue_candidate(revenue_candidates, exclude=date_column)

        mapping_unsure = (
            len(date_candidates) != 1
            or len(revenue_candidates) != 1
            or not date_column
            or not revenue_column
            or date_column == revenue_column
        )
        requires_manual_selection = mapping_unsure

        if not date_candidates:
            date_candidates = columns
        if not revenue_candidates:
            revenue_candidates = columns

        return ShopifyMappingResult(
            date_column=date_column,
            revenue_column=revenue_column,
            mapping_unsure=mapping_unsure,
            date_candidates=date_candidates,
            revenue_candidates=revenue_candidates,
            detected_export_type=detected_export_type,
            selected_export_type=selected_export_type,
            available_columns=columns,
            requires_manual_selection=requires_manual_selection,
        )

    def available_export_types(self) -> tuple[str, ...]:
        return (
            self.AUTO_EXPORT_OPTION,
            self.ADMIN_EXPORT_OPTION,
            self.REPORTS_EXPORT_OPTION,
        )

    def detect_export_type(self, columns: tuple[str, ...] | list[str]) -> str:
        admin_aliases = self.ADMIN_DATE_CANDIDATES + self.ADMIN_REVENUE_CANDIDATES
        reports_aliases = self.REPORTS_DATE_CANDIDATES + self.REPORTS_REVENUE_CANDIDATES
        admin_hits = len(self._find_candidates(tuple(columns), admin_aliases))
        reports_hits = len(self._find_candidates(tuple(columns), reports_aliases))

        if admin_hits == 0 and reports_hits == 0:
            return self.UNKNOWN_EXPORT_LABEL
        if admin_hits == reports_hits:
            return self.UNKNOWN_EXPORT_LABEL
        if admin_hits > reports_hits:
            return self.ADMIN_EXPORT_OPTION
        return self.REPORTS_EXPORT_OPTION

    def _aliases_for_export(self, export_type: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if export_type == self.ADMIN_EXPORT_OPTION:
            return self.ADMIN_DATE_CANDIDATES, self.ADMIN_REVENUE_CANDIDATES
        if export_type == self.REPORTS_EXPORT_OPTION:
            return self.REPORTS_DATE_CANDIDATES, self.REPORTS_REVENUE_CANDIDATES

        date_aliases = tuple(dict.fromkeys(self.ADMIN_DATE_CANDIDATES + self.REPORTS_DATE_CANDIDATES))
        revenue_aliases = tuple(
            dict.fromkeys(self.ADMIN_REVENUE_CANDIDATES + self.REPORTS_REVENUE_CANDIDATES)
        )
        return date_aliases, revenue_aliases

    def _find_candidates(
        self,
        columns: tuple[str, ...],
        aliases: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized_aliases = [self._normalize_name(alias) for alias in aliases]
        matches: list[str] = []
        for column in columns:
            normalized_column = self._normalize_name(column)
            if any(
                alias and (normalized_column == alias or alias in normalized_column)
                for alias in normalized_aliases
            ):
                matches.append(column)
        return tuple(matches)

    def _first_revenue_candidate(
        self,
        revenue_candidates: tuple[str, ...],
        exclude: str,
    ) -> str:
        for candidate in revenue_candidates:
            if candidate != exclude:
                return candidate
        return ""

    def _normalize_name(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
