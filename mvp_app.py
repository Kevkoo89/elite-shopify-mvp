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

    mapper = ShopifyCSVMapper()
    analyzer = ShopifyAnalyzer()

    export_type = st.selectbox(
        "CSV-Typ",
        options=list(mapper.available_export_types()),
        index=0,
    )

    uploaded_file = st.file_uploader("CSV hochladen", type=["csv"])
    file_token = (
        f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
        if uploaded_file is not None
        else ""
    )
    if st.session_state.get("mvp_file_token", "") != file_token:
        st.session_state["mvp_file_token"] = file_token
        st.session_state["mvp_run_requested"] = False

    if st.button("Analyse starten"):
        st.session_state["mvp_run_requested"] = True
    run_requested = bool(st.session_state.get("mvp_run_requested", False))

    if not uploaded_file:
        st.session_state["mvp_run_requested"] = False
        _render_empty_kpis()
        st.info("Bitte eine Shopify-CSV hochladen.")
        return

    try:
        source_df = mapper.load_csv(uploaded_file)
        detected_export_type = mapper.detect_export_type(tuple(str(c) for c in source_df.columns))
        if export_type == mapper.AUTO_EXPORT_OPTION:
            if detected_export_type != mapper.UNKNOWN_EXPORT_LABEL:
                st.caption(f"Erkannt: {detected_export_type}")
            else:
                st.warning("Unklar -> Bitte auswaehlen")
        mapped = mapper.map_columns(source_df, export_type=export_type)
    except Exception as exc:
        _render_empty_kpis()
        st.error(f"Analyse fehlgeschlagen: {exc}")
        return

    date_column = mapped.date_column
    revenue_column = mapped.revenue_column
    revenue_is_currency_text = False

    if mapped.mapping_unsure:
        st.warning("Spalten nicht eindeutig erkannt. Bitte Datum und Umsatz manuell auswaehlen.")

        date_options = list(mapped.available_columns) or [str(column) for column in source_df.columns]
        date_index = date_options.index(date_column) if date_column in date_options else 0
        date_column = st.selectbox("Datumsspalte", options=date_options, index=date_index)

        revenue_options = list(mapped.available_columns) or [str(column) for column in source_df.columns]
        if not revenue_options:
            _render_empty_kpis()
            st.error("Keine valide Umsatzspalte verfuegbar.")
            return

        revenue_default = revenue_column if revenue_column in revenue_options else revenue_options[0]
        revenue_index = revenue_options.index(revenue_default)
        revenue_column = st.selectbox(
            "Umsatzsspalte",
            options=revenue_options,
            index=revenue_index,
        )
        revenue_is_currency_text = st.checkbox(
            "Umsatz ist Text mit Waehrung",
            value=False,
        )

    if not run_requested:
        _render_empty_kpis()
        st.info("CSV geladen. Klicke auf 'Analyse starten'.")
        return

    if date_column == revenue_column:
        _render_empty_kpis()
        st.error("Datumsspalte und Umsatzsspalte duerfen nicht identisch sein.")
        return

    try:
        result = analyzer.analyze(
            source_df,
            date_column=date_column,
            revenue_column=revenue_column,
            revenue_is_currency_text=revenue_is_currency_text,
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
    st.markdown("Next checks:")
    st.markdown("- Ads/ROAS pruefen.")
    st.markdown("- Conversion Rate pruefen.")
    st.markdown("- AOV/Bestellwert pruefen.")

if __name__ == "__main__":
    main()
