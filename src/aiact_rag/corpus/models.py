"""Typed representation of the legal text.

A *unit* is the smallest citable provision we evaluate retrieval against: one recital, one article or one
annex. Units are split into *paragraphs* (Article 5(1), Annex III point 4, ...) that the structural
chunker uses as its atoms. Gold labels in the evaluation set reference unit ids (``art_5``, ``anx_III``),
which keeps recall@k comparable across chunking strategies that produce very different chunks.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

UnitKind = Literal["recital", "article", "annex"]

EURLEX_BASE = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"


class Paragraph(BaseModel):
    id: str  # e.g. "art_5.1", "anx_III.4", "art_3.0"
    label: str  # e.g. "Article 5(1)", "Annex III, point 4"
    text: str


class Unit(BaseModel):
    id: str  # ELI id from EUR-Lex: "rct_27", "art_5", "anx_III"
    kind: UnitKind
    number: str
    title: str = ""
    chapter: str | None = None
    chapter_title: str | None = None
    section: str | None = None
    section_title: str | None = None
    paragraphs: list[Paragraph]

    @property
    def label(self) -> str:
        if self.kind == "article":
            return f"Article {self.number}"
        if self.kind == "annex":
            return f"Annex {self.number}"
        return f"Recital ({self.number})"

    @property
    def heading(self) -> str:
        """Human-readable breadcrumb used as contextual prefix for structural chunks."""
        parts = ["EU AI Act (Regulation (EU) 2024/1689)"]
        if self.chapter:
            parts.append(
                f"Chapter {self.chapter}" + (f" – {self.chapter_title}" if self.chapter_title else "")
            )
        if self.section:
            parts.append(
                f"Section {self.section}" + (f" – {self.section_title}" if self.section_title else "")
            )
        parts.append(self.label + (f" – {self.title}" if self.title else ""))
        return " › ".join(parts)

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)

    @property
    def url(self) -> str:
        return f"{EURLEX_BASE}#{self.id}"
