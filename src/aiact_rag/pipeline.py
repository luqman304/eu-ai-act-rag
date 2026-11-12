"""Wires the components together; shared by the API, the CLI and the evaluation harness."""

from __future__ import annotations

import logging
from functools import cached_property

from .chunking import DEFAULT_SIZE, index_name, make_chunks
from .config import Settings
from .corpus import Unit, load_units
from .embeddings import DenseEncoder, SparseEncoder
from .generation import Answer, Generator
from .llm import LLM
from .retrieval import RetrievalConfig, RetrievalResult, Retriever
from .store import ChunkIndex, make_client

log = logging.getLogger(__name__)


class RAGService:
    def __init__(self, settings: Settings, llm: LLM | None = None):
        self.settings = settings
        self._llm = llm

    @cached_property
    def client(self):
        return make_client(self.settings)

    @cached_property
    def dense(self) -> DenseEncoder:
        return DenseEncoder(self.settings.dense_model, self.settings.model_cache_dir)

    @cached_property
    def sparse(self) -> SparseEncoder:
        return SparseEncoder(self.settings.sparse_model, self.settings.model_cache_dir)

    @cached_property
    def index(self) -> ChunkIndex:
        return ChunkIndex(self.client, self.settings, self.dense, self.sparse)

    @cached_property
    def retriever(self) -> Retriever:
        return Retriever(self.index, self.settings)

    @cached_property
    def llm(self) -> LLM:
        return self._llm or LLM(self.settings)

    @cached_property
    def generator(self) -> Generator:
        return Generator(self.llm, self.settings)

    @cached_property
    def units(self) -> dict[str, Unit]:
        return {u.id: u for u in load_units(self.settings.corpus_path)}

    def default_config(self, **overrides) -> RetrievalConfig:
        base = {
            "strategy": self.settings.default_strategy,
            "mode": self.settings.default_mode,
            "rerank": self.settings.default_rerank,
            "k": self.settings.default_k,
            "candidates": self.settings.rerank_candidates,
            "recitals": self.settings.default_recitals,
        }
        base.update({k: v for k, v in overrides.items() if v is not None})
        return RetrievalConfig(**base)

    def ingest(
        self, strategies: list[str], sizes: list[int] | None = None, skip_existing: bool = False
    ) -> dict[str, int]:
        """Chunk + embed + index. Returns {index name: number of chunks}."""
        units = list(self.units.values())
        if not self.settings.include_recitals:
            units = [u for u in units if u.kind != "recital"]
        existing = self.index.available() if skip_existing else {}
        out = {}
        for size in sizes or [DEFAULT_SIZE]:
            for strategy in strategies:
                name = index_name(strategy, size)
                if existing.get(name):
                    out[name] = existing[name]
                    continue
                chunks = make_chunks(units, strategy, size=size)
                out[name] = self.index.build(name, chunks)
        return out

    def retrieve(self, question: str, cfg: RetrievalConfig) -> RetrievalResult:
        return self.retriever.retrieve(question, cfg)

    def ask(self, question: str, cfg: RetrievalConfig) -> Answer:
        return self.generator.answer(question, self.retrieve(question, cfg))
