"""Gold evaluation set: questions labelled with the provisions that answer them."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..corpus.models import Unit


class GoldItem(BaseModel):
    id: str
    question: str
    gold_units: list[str]
    reference_answer: str
    evidence: list[str] = []  # verbatim spans that must occur in the gold units (checked in tests)
    category: str
    style: Literal["lexical", "paraphrase", "multi"]
    answerable: bool = True


def load_gold(path: Path) -> list[GoldItem]:
    with path.open(encoding="utf-8") as f:
        return [GoldItem.model_validate_json(line) for line in f if line.strip()]


def validate_gold(items: list[GoldItem], units: dict[str, Unit]) -> list[str]:
    """Return a list of problems (empty when every label is grounded in the official text)."""
    norm = lambda s: re.sub(r"\s+", " ", s)  # noqa: E731
    problems: list[str] = []
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            problems.append(f"{it.id}: duplicate id")
        seen.add(it.id)
        if it.answerable != bool(it.gold_units):
            problems.append(f"{it.id}: answerable={it.answerable} but gold_units={it.gold_units}")
        missing = [u for u in it.gold_units if u not in units]
        if missing:
            problems.append(f"{it.id}: unknown units {missing}")
            continue
        text = norm(" ".join(units[u].text for u in it.gold_units))
        problems += [f"{it.id}: evidence not found: {e!r}" for e in it.evidence if norm(e) not in text]
        if it.answerable and not it.evidence:
            problems.append(f"{it.id}: answerable item without evidence span")
    return problems
