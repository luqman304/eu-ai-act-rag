"""Runtime configuration, read from environment variables (or a .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ChunkStrategy = Literal["fixed", "fixed_overlap", "recursive", "structural", "structural_bare"]
RetrievalMode = Literal["dense", "sparse", "hybrid"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "results"
    corpus_url: str = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"
    include_recitals: bool = True

    # "http://qdrant:6333" in docker-compose; ":memory:" or a local path for embedded mode.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_prefix: str = "aiact"

    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "BAAI/bge-reranker-base"  # MiniLM-L6 never helped (see REPORT.md)
    model_cache_dir: Path | None = None  # FASTEMBED cache; baked into the Docker image

    default_strategy: str = "structural"  # index name: strategy or strategy-<size>
    default_mode: RetrievalMode = "hybrid"
    default_rerank: bool = False  # bge-reranker: +1.8 recall@5 (n.s.) for ~6 s on CPU
    default_k: int = 6
    default_recitals: bool = False  # recitals crowd out binding text: recall@1 40.5 -> 76.8
    rerank_candidates: int = 40

    anthropic_api_key: str | None = Field(default=None, repr=False)  # read from env or .env
    anthropic_base_url: str | None = None
    llm_model: str = "claude-opus-5-5"
    judge_model: str = "claude-opus-5-5"
    answer_effort: Effort = "low"
    judge_effort: Effort = "medium"
    llm_fallbacks: bool = True  # server-side refusal fallback (beta)
    llm_cache: bool = True  # on-disk response cache: makes eval re-runs free and reproducible
    llm_max_concurrency: int = 8

    api_token: str | None = Field(default=None, description="If set, required as 'Authorization: Bearer'.")
    cors_origins: list[str] = ["*"]

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "corpus" / "aiact.jsonl"

    @property
    def gold_path(self) -> Path:
        return self.data_dir / "eval" / "gold.jsonl"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def collection(self, strategy: str) -> str:
        return f"{self.collection_prefix}__{strategy}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
