from __future__ import annotations

import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.pdf_report import generate_sentra_pdf


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
        "Bezahlte Werbekanaele pruefen und schwache Kampagnen pausieren.",
        "Conversion-Funnel pruefen (Landingpage, Checkout, Zahlungsfehler).",
        "Preislogik, Rabattlogik und Warenbestand der Topseller validieren.",
    ]

    with TemporaryDirectory() as tmp:
        output_pdf = Path(tmp) / "sentra_smoke.pdf"
        chart_png = Path(tmp) / "chart_missing.png"
        long_summary = " ".join(
            [
                "Der Umsatzrueckgang ist deutlich sichtbar und wird im Wochenvergleich bestaetigt."
                for _ in range(28)
            ]
        )

        generate_sentra_pdf(
            output_path=str(output_pdf),
            store_name="Demo Shop",
            last_7_revenue=17480.25,
            previous_7_revenue=23610.10,
            change_percent=-25.96,
            status_label="Critical",
            executive_summary=long_summary,
            recommendations=recommendations,
            chart_image_path=str(chart_png),
        )

        assert output_pdf.exists(), "PDF wurde nicht erzeugt."
        raw = output_pdf.read_text(encoding="latin-1", errors="ignore")

        for token in (
            "Berichtsdatum",
            "Kurzfazit",
            "Kennzahlen",
            "Diagramm",
            "Interpretation & Empfehlungen",
            "Shop",
        ):
            assert token in raw, f"DE-Token fehlt im PDF-Inhalt: {token}"

        pages = _page_count(raw)
        assert pages >= 1, "PDF enthaelt keine Seiten."

        page_streams = _extract_page_streams(raw)
        if pages == 2 and len(page_streams) >= 2:
            page1 = page_streams[0]
            page2 = page_streams[1]
            section4_page1 = "Interpretation & Empfehlungen" in page1
            section4_page2 = "Interpretation & Empfehlungen" in page2
            bullet_hits_page2 = sum(
                1 for item in recommendations if item.split(".")[0][:24] in page2
            )

            assert section4_page2, "Section 4 soll bei Seitenumbruch sauber auf Seite 2 starten."
            assert (not section4_page1) or bullet_hits_page2 >= 2, (
                "Orphan erkannt: Seite 2 enthaelt nur einen Rest-Bullet von Section 4."
            )

    print("smoke_test_pdf_report: OK")


if __name__ == "__main__":
    main()
