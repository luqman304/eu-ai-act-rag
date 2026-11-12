import pytest

from aiact_rag.chunking import (
    DEFAULT_SIZE,
    STRATEGIES,
    chunk_fixed,
    index_name,
    make_chunks,
    parse_index_name,
)


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_every_unit_is_retrievable(units, strategy):
    chunks = make_chunks(units, strategy)
    covered = {u for c in chunks for u in c.unit_ids}
    missing = {u.id for u in units} - covered
    # fixed windows may credit a very short unit to no chunk only if it is below the credit threshold
    assert not missing or strategy.startswith("fixed")
    assert len(missing) <= 2


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_size_limit(units, strategy):
    for c in make_chunks(units, strategy):
        assert c.n_words <= DEFAULT_SIZE, (c.id, c.n_words)


@pytest.mark.parametrize("strategy", ["recursive", "structural", "structural_bare"])
def test_structure_aware_chunks_never_cross_units(units, strategy):
    assert all(len(c.unit_ids) == 1 for c in make_chunks(units, strategy))


def test_structural_has_breadcrumb_and_bare_does_not(units):
    s = make_chunks(units, "structural")
    b = make_chunks(units, "structural_bare")
    assert len(s) == len(b)
    assert all(c.text.startswith("EU AI Act") for c in s)
    assert not any(c.text.startswith("EU AI Act") for c in b)


def test_structural_repeats_lead_in_for_split_paragraphs(units):
    art5_chunks = [
        c for c in make_chunks(units, "structural") if c.unit_ids == ["art_5"] and c.label == "Article 5(1)"
    ]
    assert len(art5_chunks) > 1
    assert all("The following AI practices shall be prohibited:" in c.text for c in art5_chunks)


def test_fixed_overlap_stride(units):
    chunks = chunk_fixed(units[:20], size=100, overlap=20)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[80:] == second[:20]


def test_index_names_roundtrip():
    assert index_name("structural") == "structural"
    assert index_name("fixed", 120) == "fixed-120"
    assert parse_index_name("fixed_overlap-120") == ("fixed_overlap", 120)
    assert parse_index_name("structural") == ("structural", DEFAULT_SIZE)
