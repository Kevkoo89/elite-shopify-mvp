from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from plugins.shopify_analyzer import ShopifyAnalyzer, ShopifyCSVMapper


def _render_empty_kpis() -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Letzte 7 Tage Umsatz", "-")
    col2.metric("Vorherige 7 Tage Umsatz", "-")
    col3.metric("Change (%)", "-")
    col4.metric("Status", "-")


def _format_currency(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " EUR"


def _calculate_change_percent(last_7_total: float, previous_7_total: float) -> float:
    if previous_7_total <= 0:
        return 0.0
    return ((last_7_total - previous_7_total) / previous_7_total) * 100


def _status_from_change(change_percent: float) -> str:
    if change_percent <= -20:
        return "Critical"
    if change_percent < -5:
        return "Watch"
    return "Stable"


def _build_executive_summary(change_percent: float) -> str:
    if change_percent <= -20:
        return (
            "Es zeigt sich ein klarer Abwaertstrend beim Umsatz. Die letzten 7 Tage liegen "
            "deutlich unter der Vorperiode und signalisieren ein erhoehtes kurzfristiges Risiko."
        )
    if -20 < change_percent < -5:
        return (
            "Der Umsatz zeigt im Vergleich zur vorherigen 7-Tage-Periode einen leichten "
            "Negativtrend. Die Entwicklung sollte eng beobachtet werden."
        )
    if change_percent >= 0:
        return (
            "Der Umsatz ist im Vergleich zur Vorperiode stabil oder positiv. Aktuell ist "
            "kein unmittelbares Umsatzrisiko erkennbar."
        )
    return (
        "Der Umsatz liegt leicht unter der Vorperiode. Der Trend ist noch nicht kritisch, "
        "sollte aber regelmaessig beobachtet werden."
    )


def _build_recommendations(status_label: str) -> list[str]:
    if status_label == "Critical":
        return [
            "Bezahlte Werbekanaele pruefen und schwache Kampagnen pausieren.",
            "Conversion-Funnel pruefen (Landingpage, Checkout, Zahlungsfehler).",
            "Preislogik, Rabattlogik und Warenbestand der Topseller validieren.",
        ]
    if status_label == "Watch":
        return [
            "Taeglichen Umsatz und Traffic-Mix in den naechsten 3-5 Tagen beobachten.",
            "ROAS und Conversion-Rate nach Kampagne/Quelle erneut pruefen.",
            "Eine gezielte Optimierung in Angebot, Creatives oder Checkout testen.",
        ]
    return [
        "Aktuelle Akquisitions- und Retention-Strategie beibehalten.",
        "Umsatz und Rolling-Trend weiterhin woechentlich beobachten.",
        "Vorsorgliche Massnahmen vorbereiten, falls der Trend nachlaesst.",
    ]


def _save_chart_png(chart_df, output_path: str) -> None:
    from reportlab.graphics import renderPM
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.lib.colors import HexColor

    output = Path(output_path)
    if output.parent and str(output.parent) != ".":
        output.parent.mkdir(parents=True, exist_ok=True)

    revenue_values = [float(value) for value in chart_df["revenue"].tolist()]
    rolling_values = [float(value) for value in chart_df["rolling_7_avg"].tolist()]
    if not revenue_values:
        raise ValueError("No chart data available for export.")

    revenue_points = [(idx + 1, value) for idx, value in enumerate(revenue_values)]
    rolling_points = [(idx + 1, value) for idx, value in enumerate(rolling_values)]
    all_values = revenue_values + rolling_values
    y_min = min(all_values)
    y_max = max(all_values)
    y_delta = max(y_max - y_min, 1.0)
    y_padding = y_delta * 0.1

    drawing = Drawing(1000, 480)
    chart = LinePlot()
    chart.x = 70
    chart.y = 80
    chart.width = 860
    chart.height = 300
    chart.data = [revenue_points, rolling_points]
    chart.joinedLines = 1
    chart.lines[0].strokeColor = HexColor("#1E1E1E")
    chart.lines[0].strokeWidth = 2
    chart.lines[1].strokeColor = HexColor("#00AFAF")
    chart.lines[1].strokeWidth = 2

    chart.xValueAxis.valueMin = 1
    chart.xValueAxis.valueMax = max(1, len(revenue_points))
    if len(revenue_points) == 1:
        x_steps = [1]
    else:
        step = max(1, len(revenue_points) // 8)
        x_steps = list(range(1, len(revenue_points) + 1, step))
        if x_steps[-1] != len(revenue_points):
            x_steps.append(len(revenue_points))
    chart.xValueAxis.valueSteps = x_steps

    date_labels = [
        value.strftime("%m-%d") if hasattr(value, "strftime") else str(value)
        for value in chart_df.index
    ]

    def _x_label_formatter(value: float) -> str:
        idx = int(round(value)) - 1
        if 0 <= idx < len(date_labels):
            return date_labels[idx]
        return ""

    chart.xValueAxis.labelTextFormat = _x_label_formatter
    chart.xValueAxis.labels.angle = 25
    chart.xValueAxis.labels.fontSize = 8
    chart.xValueAxis.labels.fillColor = HexColor("#5A5A5A")

    chart.yValueAxis.valueMin = max(0.0, y_min - y_padding)
    chart.yValueAxis.valueMax = y_max + y_padding
    chart.yValueAxis.visibleGrid = True
    chart.yValueAxis.gridStrokeColor = HexColor("#E0E0E0")
    chart.yValueAxis.gridStrokeWidth = 0.6
    chart.yValueAxis.labels.fontSize = 8
    chart.yValueAxis.labels.fillColor = HexColor("#5A5A5A")

    drawing.add(chart)
    drawing.add(
        String(
            70,
            420,
            "Revenue vs Rolling 7-Day Average",
            fontName="Helvetica-Bold",
            fontSize=14,
            fillColor=HexColor("#1E1E1E"),
        )
    )
    drawing.add(String(74, 54, "Revenue", fontName="Helvetica", fontSize=9, fillColor=HexColor("#5A5A5A")))
    drawing.add(String(185, 54, "Rolling 7 Avg", fontName="Helvetica", fontSize=9, fillColor=HexColor("#5A5A5A")))

    renderPM.drawToFile(drawing, str(output), fmt="PNG")


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

    change_percent = _calculate_change_percent(result.last_7_total, result.previous_7_total)
    status_label = _status_from_change(change_percent)
    executive_summary = _build_executive_summary(change_percent)
    recommendations = _build_recommendations(status_label)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Letzte 7 Tage Umsatz", _format_currency(result.last_7_total))
    col2.metric("Vorherige 7 Tage Umsatz", _format_currency(result.previous_7_total))
    col3.metric("Change (%)", f"{change_percent:+.2f}%")
    col4.metric("Status", status_label)

    st.subheader("Chart")
    chart_df = result.daily_revenue.set_index("date")[["revenue", "rolling_7_avg"]]
    st.line_chart(chart_df)
    chart_image_path = "temp_chart.png"

    try:
        _save_chart_png(chart_df, chart_image_path)
    except Exception as exc:
        st.warning(f"Chart konnte nicht als PNG gespeichert werden: {exc}")
        chart_image_path = ""

    if status_label == "Critical":
        st.error("Fruehwarnung: Deutlicher Abwaertstrend erkannt.")
    elif status_label == "Watch":
        st.warning("Achtung: Leichter Negativtrend erkannt.")
    else:
        st.success("Trend stabil oder positiv.")

    st.subheader("Executive Summary")
    st.write(executive_summary)

    st.subheader("Interpretation & Recommendations")
    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")

    store_name = Path(uploaded_file.name).stem if uploaded_file else "Shopify Store"
    if st.button("Generate PDF Report"):
        if not chart_image_path:
            st.error("PDF konnte nicht erstellt werden, weil kein Chart-PNG vorliegt.")
        else:
            try:
                from utils.pdf_report import generate_sentra_pdf
            except ModuleNotFoundError:
                st.error("PDF-Export benoetigt reportlab. Bitte requirements.txt installieren.")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_pdf = f"sentra_report_{timestamp}.pdf"
            try:
                generate_sentra_pdf(
                    output_path=output_pdf,
                    store_name=store_name,
                    last_7_revenue=result.last_7_total,
                    previous_7_revenue=result.previous_7_total,
                    change_percent=change_percent,
                    status_label=status_label,
                    executive_summary=executive_summary,
                    recommendations=recommendations,
                    chart_image_path=chart_image_path,
                )
            except Exception as exc:
                st.error(f"PDF-Export fehlgeschlagen: {exc}")
            else:
                st.session_state["sentra_pdf_path"] = output_pdf
                st.success(f"PDF gespeichert: {output_pdf}")

    pdf_path = st.session_state.get("sentra_pdf_path")
    if pdf_path:
        pdf_file = Path(str(pdf_path))
        if pdf_file.exists():
            st.download_button(
                "Download PDF Report",
                data=pdf_file.read_bytes(),
                file_name=pdf_file.name,
                mime="application/pdf",
            )
        else:
            st.session_state.pop("sentra_pdf_path", None)

    st.caption(f"Warnscore: {result.warn_score:.1f}")
    st.caption(result.summary_text)

if __name__ == "__main__":
    main()
