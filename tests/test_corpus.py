from pathlib import Path

import pytest

from aiact_rag.corpus import parse_html


def test_unit_counts(units):
    kinds = [u.kind for u in units]
    assert kinds.count("recital") == 180
    assert kinds.count("article") == 113
    assert kinds.count("annex") == 13


def test_no_empty_paragraphs(units):
    for u in units:
        assert u.paragraphs, u.id
        for p in u.paragraphs:
            assert p.text.strip(), p.id


def test_article_structure(units_by_id):
    art5 = units_by_id["art_5"]
    assert art5.title == "Prohibited AI practices"
    assert art5.chapter == "II"
    assert [p.label for p in art5.paragraphs][:2] == ["Article 5(1)", "Article 5(2)"]
    # enumerated points are flattened into "(x) text" lines
    assert "\n(f) the placing on the market" in art5.paragraphs[0].text
    art6 = units_by_id["art_6"]
    assert art6.section == "1" and "high-risk" in art6.section_title


def test_markup_edge_cases(units_by_id):
    # exponents survive as ^, footnote markers are dropped
    assert "greater than 10^25." in units_by_id["art_51"].text
    # Annex I items are bare <span>s in table cells, not <p>
    assert "on medical devices" in units_by_id["anx_I"].text
    assert "Section B" in units_by_id["anx_I"].text


def test_heading_breadcrumb(units_by_id):
    h = units_by_id["art_9"].heading
    assert h.startswith("EU AI Act") and "Chapter III" in h and "Article 9 – Risk management system" in h


RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "aiact.html"


@pytest.mark.skipif(not RAW.exists(), reason="raw EUR-Lex HTML not downloaded (run `aiact fetch`)")
def test_parse_is_deterministic(units):
    reparsed = parse_html(RAW.read_text(encoding="utf-8"))
    assert [u.model_dump() for u in reparsed] == [u.model_dump() for u in units]
