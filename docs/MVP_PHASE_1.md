# MVP Phase 1 – Shopify Revenue Early Warning

## Kontext
Neues Projekt: elite-shopify-mvp  
Ziel: Minimaler funktionsfähiger MVP zur Validierung des Frühwarnsystems.

Diese Phase ist strikt begrenzt.  
Kein Overengineering. Kein Ausbau alter Enterprise-Struktur.

---

## Arbeitsregeln

- Niemals auf `main` arbeiten.
- Nur im Branch `feature/mvp-init`.
- Minimal-Diff.
- Keine neuen Subsysteme.
- Keine Security-/Autonomie-Module.
- Keine Tests in dieser Phase.
- Keine Governance-Dokumente.
- Keine Refactors alter Struktur.

---

## Ziel dieser Iteration

### 1. Streamlit UI
- Titel: "Shopify Revenue Early Warning"
- CSV Upload
- Button: "Analyse starten"
- 3 KPI Felder
- Chart Bereich
- Text-Ausgabe

### 2. Plugin-Struktur
Erstellen:

plugins/shopify_analyzer/
    __init__.py
    mapper.py
    analyzer.py

Nur Skeleton + klare Klassenstruktur:
- ShopifyCSVMapper
- ShopifyAnalyzer

Keine komplexe Logik.

---

### 3. Dummy-Analyse

- CSV mit pandas laden
- Datum + Umsatz Spalten erkennen
- Daily Revenue aggregieren
- Rolling 7 Day Average berechnen
- Vergleich letzte 7 Tage vs vorherige 7 Tage
- Warnscore (0–100, Prozentvergleich)

---

### 4. Visualisierung

- Umsatz + Rolling Average
- Wenn Warnscore > 60 → roter Hinweis anzeigen

---

## Deliverables

- Lauffähige Streamlit App
- Liste aller neu erstellten/geänderten Dateien
- Exakter Startbefehl
- Kurzbeschreibung der Warnscore-Logik

---

## Explizit NICHT Bestandteil dieser Phase

- PDF-Export
- Auth
- API
- Deployment
- Multi-Tenant
- Enterprise-Architektur
- Performance-Optimierung