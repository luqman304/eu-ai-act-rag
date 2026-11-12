from collections import Counter

from aiact_rag.eval.dataset import load_gold, validate_gold


def test_gold_labels_are_grounded_in_the_official_text(settings, units_by_id):
    items = load_gold(settings.gold_path)
    assert validate_gold(items, units_by_id) == []


def test_gold_set_composition(settings):
    items = load_gold(settings.gold_path)
    assert len(items) >= 80
    styles = Counter(i.style for i in items if i.answerable)
    assert styles["paraphrase"] >= 30 and styles["multi"] >= 8
    assert sum(not i.answerable for i in items) >= 4
