"""Two-stage retrieval: first-stage search in Qdrant, optional cross-encoder reranking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .chunking import Chunk
from .config import Settings
from .embeddings import CrossEncoder
from .store import ChunkIndex


@dataclass(frozen=True)
class RetrievalConfig:
    strategy: str = "structural"
    mode: str = "hybrid"
    rerank: bool = True
    k: int = 6
    candidates: int = 40
    rerank_model: str | None = None  # None → settings.rerank_model
    recitals: bool = True  # False → only operative text (articles + annexes) is searched

    @property
    def name(self) -> str:
        rr = f"+rerank({(self.rerank_model or 'default').split('/')[-1]})" if self.rerank else ""
        rc = "" if self.recitals else "-recitals"
        return f"{self.strategy}/{self.mode}{rr}{rc}"


@dataclass
class Retrieved:
    chunk: Chunk
    rank: int
    first_stage_score: float
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    query: str
    config: RetrievalConfig
    items: list[Retrieved]
    timings_ms: dict[str, float] = field(default_factory=dict)

    def ranked_units(self) -> list[str]:
        """Unit ids in rank order, de-duplicated (first occurrence wins)."""
        seen: dict[str, None] = {}
        for it in self.items:
            for u in it.chunk.unit_ids:
                seen.setdefault(u, None)
        return list(seen)


class Retriever:
    def __init__(self, index: ChunkIndex, settings: Settings):
        self.index = index
        self.settings = settings
        self._rerankers: dict[str, CrossEncoder] = {}

    def reranker(self, model: str | None = None) -> CrossEncoder:
        name = model or self.settings.rerank_model
        if name not in self._rerankers:
            self._rerankers[name] = CrossEncoder(name, self.settings.model_cache_dir)
        return self._rerankers[name]

    def retrieve(self, query: str, cfg: RetrievalConfig) -> RetrievalResult:
        t0 = time.perf_counter()
        limit = max(cfg.candidates, cfg.k) if cfg.rerank else cfg.k
        hits = self.index.search(cfg.strategy, query, cfg.mode, limit, recitals=cfg.recitals)
        t1 = time.perf_counter()
        timings = {"search_ms": (t1 - t0) * 1000}
        if cfg.rerank and hits:
            scores = self.reranker(cfg.rerank_model).score(query, [h.chunk.text for h in hits])
            order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)[: cfg.k]
            items = [
                Retrieved(
                    chunk=hits[i].chunk, rank=r, first_stage_score=hits[i].score, rerank_score=scores[i]
                )
                for r, i in enumerate(order, start=1)
            ]
            timings["rerank_ms"] = (time.perf_counter() - t1) * 1000
        else:
            items = [
                Retrieved(chunk=h.chunk, rank=r, first_stage_score=h.score) for r, h in enumerate(hits, 1)
            ]
        return RetrievalResult(query=query, config=cfg, items=items[: cfg.k], timings_ms=timings)
