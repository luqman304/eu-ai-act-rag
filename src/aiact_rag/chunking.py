"""Chunking strategies compared in the evaluation.

All strategies operate on the same parsed corpus and emit :class:`Chunk` objects carrying the ids of the
legal units (articles / annexes / recitals) whose text they contain. That attribution is what lets
recall@k be measured at the *provision* level, so strategies with very different chunk boundaries are
scored against the same gold labels.

* ``fixed``         – the whole regulation as one word stream, cut into fixed windows (no overlap).
                      Boundaries ignore structure; a window can straddle two articles.
* ``fixed_overlap`` – same, with overlapping windows (stride = size - overlap).
* ``recursive``     – per-unit recursive splitting on newline → sentence → word, then greedy merging
                      (the LangChain ``RecursiveCharacterTextSplitter`` recipe, in words). Never crosses
                      a unit boundary, but carries no headings.
* ``structural``    – packs whole paragraphs (Article 5(1), Annex III point 4, ...) up to the size limit,
                      splits oversized paragraphs on enumerated points while repeating the lead-in
                      sentence, and prefixes every chunk with its breadcrumb (Chapter › Section ›
                      Article – title › paragraph label).
* ``structural_bare`` – ablation: identical boundaries to ``structural`` but without the breadcrumb,
                      isolating how much of the gain comes from boundaries vs. contextual headers.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from pydantic import BaseModel

from .corpus.models import Paragraph, Unit

DEFAULT_SIZE = 250  # words; ≈330 BPE tokens, comfortably under the 512-token limit of bge-small
DEFAULT_OVERLAP = 50
MIN_CREDIT_WORDS = 25  # a chunk "contains" a unit if it holds ≥ this many of its words (or all of them)

_SENT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(‘“])")


class Chunk(BaseModel):
    id: str
    strategy: str
    text: str
    unit_ids: list[str]
    kinds: list[str]  # unit kinds covered: "recital" | "article" | "annex"
    paragraph_ids: list[str]
    label: str
    n_words: int


def _words(text: str) -> list[str]:
    return text.split()


def _label_span(paragraphs: list[Paragraph]) -> str:
    if not paragraphs:
        return ""
    first, last = paragraphs[0].label, paragraphs[-1].label
    return first if first == last else f"{first} – {last}"


def chunk_fixed(units: list[Unit], size: int = DEFAULT_SIZE, overlap: int = 0) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    tokens: list[tuple[str, int, int]] = []  # (word, unit index, paragraph index)
    para_index: list[Paragraph] = []
    unit_len: list[int] = []
    for ui, unit in enumerate(units):
        n = 0
        for para in unit.paragraphs:
            pi = len(para_index)
            para_index.append(para)
            ws = _words(para.text)
            tokens.extend((w, ui, pi) for w in ws)
            n += len(ws)
        unit_len.append(n)

    strategy = "fixed" if overlap == 0 else "fixed_overlap"
    stride = size - overlap
    chunks: list[Chunk] = []
    for start in range(0, len(tokens), stride):
        window = tokens[start : start + size]
        if not window:
            break
        counts: dict[int, int] = {}
        for _, ui, _ in window:
            counts[ui] = counts.get(ui, 0) + 1
        credited = [ui for ui, c in counts.items() if c >= min(MIN_CREDIT_WORDS, unit_len[ui])]
        pids = list(dict.fromkeys(pi for _, _, pi in window))
        paras = [para_index[p] for p in pids]
        chunks.append(
            Chunk(
                id=f"{strategy}:{len(chunks)}",
                strategy=strategy,
                text=" ".join(w for w, _, _ in window),
                unit_ids=[units[ui].id for ui in credited],
                kinds=sorted({units[ui].kind for ui in credited}),
                paragraph_ids=[p.id for p in paras],
                label=_label_span(paras),
                n_words=len(window),
            )
        )
        if start + size >= len(tokens):
            break
    return chunks


def _recursive_split(text: str, size: int, seps: tuple[Callable[[str], list[str]], ...]) -> list[str]:
    if len(_words(text)) <= size:
        return [text]
    if not seps:
        ws = _words(text)
        return [" ".join(ws[i : i + size]) for i in range(0, len(ws), size)]
    split, rest = seps[0], seps[1:]
    pieces = [p for p in split(text) if p.strip()]
    if len(pieces) <= 1:
        return _recursive_split(text, size, rest)
    out: list[str] = []
    for p in pieces:
        out.extend(_recursive_split(p, size, rest))
    return out


def _merge(pieces: list[str], size: int, joiner: str = "\n") -> list[str]:
    merged: list[str] = []
    buf: list[str] = []
    n = 0
    for p in pieces:
        k = len(_words(p))
        if buf and n + k > size:
            merged.append(joiner.join(buf))
            buf, n = [], 0
        buf.append(p)
        n += k
    if buf:
        merged.append(joiner.join(buf))
    return merged


_RECURSIVE_SEPS: tuple[Callable[[str], list[str]], ...] = (
    lambda t: t.split("\n"),
    lambda t: _SENT.split(t),
)


def chunk_recursive(units: list[Unit], size: int = DEFAULT_SIZE) -> list[Chunk]:
    chunks: list[Chunk] = []
    for unit in units:
        pieces = _recursive_split(unit.text, size, _RECURSIVE_SEPS)
        for text in _merge(pieces, size):
            paras = [p for p in unit.paragraphs if _overlaps(p.text, text)] or unit.paragraphs[:1]
            chunks.append(
                Chunk(
                    id=f"recursive:{len(chunks)}",
                    strategy="recursive",
                    text=text,
                    unit_ids=[unit.id],
                    kinds=[unit.kind],
                    paragraph_ids=[p.id for p in paras],
                    label=_label_span(paras),
                    n_words=len(_words(text)),
                )
            )
    return chunks


def _overlaps(paragraph_text: str, chunk_text: str) -> bool:
    probe = " ".join(_words(paragraph_text)[:12])
    return probe in " ".join(_words(chunk_text)) or " ".join(_words(chunk_text)[:12]) in " ".join(
        _words(paragraph_text)
    )


def _split_paragraph(para: Paragraph, budget: int) -> list[str]:
    """Split one oversized paragraph on its lines (enumerated points), repeating the lead-in line."""
    lines = para.text.split("\n")
    lead = lines[0] if len(lines) > 1 and lines[0].rstrip().endswith(":") else None
    body = lines[1:] if lead else lines
    lead_words = len(_words(lead)) if lead else 0
    room = max(budget - lead_words, budget // 2)
    pieces: list[str] = []
    for line in body:
        pieces.extend(_recursive_split(line, room, (lambda t: _SENT.split(t),)))
    groups = _merge(pieces, room)
    if lead:
        groups = [f"{lead}\n{g}" for g in groups]
    return groups


def chunk_structural(units: list[Unit], size: int = DEFAULT_SIZE, headings: bool = True) -> list[Chunk]:
    chunks: list[Chunk] = []
    strategy = "structural" if headings else "structural_bare"

    def emit(unit: Unit, paras: list[Paragraph], body: str) -> None:
        header = f"{unit.heading}\n{_label_span(paras)}"
        chunks.append(
            Chunk(
                id=f"{strategy}:{len(chunks)}",
                strategy=strategy,
                text=f"{header}\n{body}" if headings else body,
                unit_ids=[unit.id],
                kinds=[unit.kind],
                paragraph_ids=[p.id for p in paras],
                label=_label_span(paras),
                n_words=len(_words(body)),
            )
        )

    for unit in units:
        # Same packing for both variants, so the bare ablation isolates the effect of the breadcrumb.
        budget = size - min(len(_words(unit.heading)), size // 4)
        buf: list[Paragraph] = []
        n = 0
        for para in unit.paragraphs:
            k = len(_words(para.text))
            if k > budget:
                if buf:
                    emit(unit, buf, "\n".join(p.text for p in buf))
                    buf, n = [], 0
                for part in _split_paragraph(para, budget):
                    emit(unit, [para], part)
                continue
            if buf and n + k > budget:
                emit(unit, buf, "\n".join(p.text for p in buf))
                buf, n = [], 0
            buf.append(para)
            n += k
        if buf:
            emit(unit, buf, "\n".join(p.text for p in buf))
    return chunks


STRATEGIES: dict[str, Callable[..., list[Chunk]]] = {
    "fixed": lambda units, size=DEFAULT_SIZE: chunk_fixed(units, size=size, overlap=0),
    "fixed_overlap": lambda units, size=DEFAULT_SIZE: chunk_fixed(units, size=size, overlap=size // 5),
    "recursive": chunk_recursive,
    "structural": chunk_structural,
    "structural_bare": lambda units, size=DEFAULT_SIZE: chunk_structural(units, size=size, headings=False),
}


def index_name(strategy: str, size: int = DEFAULT_SIZE) -> str:
    """Collection key for a (strategy, size) pair: ``structural`` or ``structural-120``."""
    return strategy if size == DEFAULT_SIZE else f"{strategy}-{size}"


def parse_index_name(name: str) -> tuple[str, int]:
    strategy, _, size = name.partition("-")
    return strategy, int(size) if size else DEFAULT_SIZE


def make_chunks(units: Iterable[Unit], strategy: str, size: int = DEFAULT_SIZE) -> list[Chunk]:
    try:
        fn = STRATEGIES[strategy]
    except KeyError as e:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}") from e
    return fn(list(units), size=size)
