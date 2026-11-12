from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrievalOptions(BaseModel):
    strategy: str | None = Field(
        None, description="Index name, e.g. 'structural'. Defaults to the server config."
    )
    mode: Literal["dense", "sparse", "hybrid"] | None = None
    rerank: bool | None = None
    recitals: bool | None = Field(None, description="Include recitals (non-binding preamble) in the search.")
    k: int | None = Field(None, ge=1, le=20)


class SearchRequest(RetrievalOptions):
    query: str = Field(..., min_length=3, max_length=2000)


class AskRequest(RetrievalOptions):
    question: str = Field(..., min_length=3, max_length=2000)


class ChunkOut(BaseModel):
    rank: int
    chunk_id: str
    label: str
    unit_ids: list[str]
    text: str
    score: float
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    config: str
    results: list[ChunkOut]
    timings_ms: dict[str, float]


class CitationOut(BaseModel):
    n: int
    label: str
    unit_ids: list[str]
    url: str


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    citations: list[CitationOut]
    contexts: list[ChunkOut]
    config: str
    model: str
    usage: dict[str, int]
    timings_ms: dict[str, float]
    disclaimer: str = "Information about the text of Regulation (EU) 2024/1689, not legal advice."


class ParagraphOut(BaseModel):
    id: str
    label: str
    text: str


class ProvisionResponse(BaseModel):
    id: str
    kind: str
    label: str
    title: str
    heading: str
    url: str
    paragraphs: list[ParagraphOut]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    collections: dict[str, int]
    default_index_ready: bool
    llm_configured: bool
    version: str
