"""Grounded answer generation with numbered, verifiable citations."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from html import escape

from .config import Settings
from .corpus.models import EURLEX_BASE
from .llm import LLM
from .retrieval import RetrievalResult, Retrieved

SYSTEM_PROMPT = """You answer questions about the EU Artificial Intelligence Act (Regulation (EU) 2024/1689) \
for compliance teams, lawyers and engineers.

You are given numbered excerpts retrieved from the official text. Base every statement strictly on \
those excerpts: do not add obligations, dates, thresholds or exceptions that the excerpts do not contain, \
even if you believe them to be true. Cite the excerpt supporting each sentence with its number in \
square brackets, e.g. [2] or [1][3]. Prefer the regulation's own wording for legal terms and mention \
the article or annex you rely on.

If the excerpts do not contain the information needed, set "answerable" to false and say briefly what \
is missing instead of guessing. If they only partly answer the question, answer that part and state \
what is not covered.

Be concise: a direct answer first, then the key conditions or exceptions. This is information about the \
law, not legal advice."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": ["answerable", "answer"],
    "additionalProperties": False,
}

_CITE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Citation:
    n: int
    chunk_id: str
    label: str
    unit_ids: list[str]
    url: str


@dataclass
class Answer:
    question: str
    answer: str
    answerable: bool
    citations: list[Citation]
    contexts: list[Retrieved]
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def context_texts(self) -> list[str]:
        return [c.chunk.text for c in self.contexts]


def format_context(items: list[Retrieved]) -> str:
    blocks = []
    for i, it in enumerate(items, start=1):
        c = it.chunk
        blocks.append(
            f'<excerpt n="{i}" source="{escape(c.label)}" units="{",".join(c.unit_ids)}">\n{c.text}\n</excerpt>'
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, items: list[Retrieved]) -> str:
    return f"<excerpts>\n{format_context(items)}\n</excerpts>\n\nQuestion: {question}"


def extract_citations(answer: str, items: list[Retrieved]) -> list[Citation]:
    seen: dict[int, Citation] = {}
    for m in _CITE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(items) and n not in seen:
            c = items[n - 1].chunk
            anchor = c.unit_ids[0] if c.unit_ids else ""
            seen[n] = Citation(
                n=n, chunk_id=c.id, label=c.label, unit_ids=c.unit_ids, url=f"{EURLEX_BASE}#{anchor}"
            )
    return sorted(seen.values(), key=lambda c: c.n)


class Generator:
    def __init__(self, llm: LLM, settings: Settings):
        self.llm = llm
        self.settings = settings

    def answer(self, question: str, retrieval: RetrievalResult) -> Answer:
        t0 = time.perf_counter()
        resp = self.llm.complete(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(question, retrieval.items),
            model=self.settings.llm_model,
            effort=self.settings.answer_effort,
            max_tokens=8000,
            schema=ANSWER_SCHEMA,
        )
        data = resp.json()
        text = data["answer"].strip()
        timings = dict(retrieval.timings_ms)
        timings["generate_ms"] = (time.perf_counter() - t0) * 1000
        return Answer(
            question=question,
            answer=text,
            answerable=bool(data["answerable"]),
            citations=extract_citations(text, retrieval.items),
            contexts=retrieval.items,
            model=resp.model,
            usage=resp.usage,
            timings_ms=timings,
        )
