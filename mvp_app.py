from __future__ import annotations

from typing import Optional

import streamlit as st

from plugins.shopify_analyzer import ShopifyAnalyzer, ShopifyCSVMapper


def _render_empty_kpis() -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Letzte 7 Tage Umsatz", "-")
    col2.metric("Vorherige 7 Tage Umsatz", "-")
    col3.metric("Warnscore", "-")


def _format_currency(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " EUR"


def main() -> None:
    st.set_page_config(page_title="Shopify Revenue Early Warning", layout="wide")
    st.title("Shopify Revenue Early Warning")

    uploaded_file = st.file_uploader("CSV hochladen", type=["csv"])
    start_analysis = st.button("Analyse starten")

    if not uploaded_file:
        _render_empty_kpis()
        st.info("Bitte eine Shopify-CSV hochladen.")
        return

    mapper = ShopifyCSVMapper()
    analyzer = ShopifyAnalyzer()

    if not start_analysis:
        _render_empty_kpis()
        st.info("CSV geladen. Klicke auf 'Analyse starten'.")
        return

    try:
        source_df = mapper.load_csv(uploaded_file)
        mapped = mapper.map_columns(source_df)
        result = analyzer.analyze(
            source_df,
            date_column=mapped.date_column,
            revenue_column=mapped.revenue_column,
        )
    except Exception as exc:
        _render_empty_kpis()
        st.error(f"Analyse fehlgeschlagen: {exc}")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Letzte 7 Tage Umsatz", _format_currency(result.last_7_total))
    col2.metric("Vorherige 7 Tage Umsatz", _format_currency(result.previous_7_total))
    col3.metric("Warnscore", f"{result.warn_score:.1f}")

    st.subheader("Chart")
    chart_df = result.daily_revenue.set_index("date")[["revenue", "rolling_7_avg"]]
    st.line_chart(chart_df)

    if result.warn_score > 60:
        st.error("Fruehwarnung: Warnscore ueber 60.")

    st.subheader("Text-Ausgabe")
    st.write(result.summary_text)

if __name__ == "__main__":
    main()
