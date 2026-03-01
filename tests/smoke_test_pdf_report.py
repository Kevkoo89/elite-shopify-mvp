from __future__ import annotations

import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.pdf_report import format_currency_eur
from utils.pdf_report import generate_sentra_pdf
from utils.pdf_report import _currency  # noqa: PLC2701


def _page_count(pdf_text: str) -> int:
    return len(re.findall(r"/Type\s*/Page\b", pdf_text))


def _extract_page_streams(pdf_text: str) -> list[str]:
    object_pattern = re.compile(r"(\d+)\s+0\s+obj(.*?)endobj", re.DOTALL)
    objects = {int(obj_id): body for obj_id, body in object_pattern.findall(pdf_text)}

    pages: list[tuple[int, str]] = []
    for obj_id, body in objects.items():
        if "/Type /Page" in body and "/Parent" in body:
            pages.append((obj_id, body))
    pages.sort(key=lambda item: item[0])

    page_streams: list[str] = []
    for _, page_body in pages:
        content_ids: list[int] = []
        single_match = re.search(r"/Contents\s+(\d+)\s+0\s+R", page_body)
        array_match = re.search(r"/Contents\s*\[(.*?)\]", page_body, re.DOTALL)
        if single_match:
            content_ids = [int(single_match.group(1))]
        elif array_match:
            content_ids = [int(value) for value in re.findall(r"(\d+)\s+0\s+R", array_match.group(1))]

        streams: list[str] = []
        for content_id in content_ids:
            content_body = objects.get(content_id, "")
            stream_matches = re.findall(r"stream\r?\n(.*?)\r?\nendstream", content_body, re.DOTALL)
            streams.extend(stream_matches)
        page_streams.append("\n".join(streams))

    return page_streams


def main() -> None:
    recommendations = [
        "Bezahlte Werbekanäle prüfen und schwache Kampagnen pausieren.",
        "Conversion-Funnel prüfen (Landingpage, Checkout, Zahlungsfehler).",
        "Preislogik, Rabattlogik und Warenbestand der Topseller validieren.",
    ]

    with TemporaryDirectory() as tmp:
        output_pdf = Path(tmp) / "sentra_smoke.pdf"
        chart_png = Path(tmp) / "chart_missing.png"
        long_summary = " ".join(
            [
                "Der Umsatzrückgang ist deutlich sichtbar und wird im Wochenvergleich bestätigt."
                for _ in range(8)
            ]
        )

        generate_sentra_pdf(
            output_path=str(output_pdf),
            store_name="test_shopify_admin",
            last_7_revenue=17480.25,
            previous_7_revenue=23610.10,
            change_percent=-25.96,
            status_label="Critical",
            executive_summary=long_summary,
            recommendations=recommendations,
            chart_image_path=str(chart_png),
            language="de",
            warn_score=72.4,
        )

        assert output_pdf.exists(), "PDF wurde nicht erzeugt."
        raw = output_pdf.read_text(encoding="latin-1", errors="ignore")

        amount, currency = format_currency_eur(1195.0)
        assert amount == "1.195,00", "Betragsformat ist nicht DE-konform."
        assert currency == "EUR", "Währungscode unerwartet."
        assert _currency(1195.0) == "1.195,00\u00A0EUR", "Währungsformat ohne NBSP."
        assert " E) Tj T* (UR" not in raw, "EUR wurde über Zeilen getrennt."
        assert "(E) Tj T* (UR)" not in raw, "EUR wurde über Zeilen getrennt."
        assert "1.195,00EUR" not in raw, "EUR klebt ohne Abstand an der Zahl."

        for token in (
            "Berichtsdatum",
            "Kurzfazit",
            "Kennzahlen",
            "Diagramm",
            "Interpretation & Empfehlungen",
            "KPIs erkl",
            "N",
            "Checks",
            "Shop",
            "Erstellt von Sentra",
            "Kevron Dynamics",
            "Seite",
            "Was bedeutet der Status",
            "Hinweis zur Datenbasis",
        ):
            assert token in raw, f"DE-Token fehlt im PDF-Inhalt: {token}"

        pages = _page_count(raw)
        assert pages == 2, f"PDF sollte genau 2 Seiten enthalten, gefunden: {pages}"

        page_streams = _extract_page_streams(raw)
        assert len(page_streams) >= 2, "PDF-Inhalt von Seite 2 konnte nicht gelesen werden."

        page2 = page_streams[1]
        assert "Interpretation & Empfehlungen" in page2, "Section 4 fehlt auf Seite 2."
        assert "KPIs erkl" in page2, "Section 5 fehlt auf Seite 2."
        assert "Checks" in page2, "Section 6 fehlt auf Seite 2."
        assert len(page2.strip()) > 500, "Seite 2 wirkt fast leer."

        bullet_tokens = ("Bezahlte Werbe", "Conversion-Funnel", "Preislogik, Rabattlogik")
        bullet_hits_page2 = sum(1 for token in bullet_tokens if token in page2)
        assert bullet_hits_page2 >= 3, "Seite 2 enthält zu wenige Empfehlungspunkte."

    print("smoke_test_pdf_report: OK")


if __name__ == "__main__":
    main()
