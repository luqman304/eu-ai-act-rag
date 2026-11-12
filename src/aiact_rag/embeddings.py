"""Local ONNX models via fastembed: dense bi-encoder, BM25 sparse encoder and a cross-encoder reranker.

Everything runs on CPU without an API key, so indexing and the whole retrieval evaluation are free and
deterministic. Models are loaded lazily and shared process-wide.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

# BGE v1.5 retrieval instruction for short queries (https://huggingface.co/BAAI/bge-small-en-v1.5)
_QUERY_PREFIX = {"BAAI/bge": "Represent this sentence for searching relevant passages: "}

_lock = threading.Lock()


def _cache(cache_dir: Path | None) -> dict:
    return {"cache_dir": str(cache_dir)} if cache_dir else {}


@lru_cache(maxsize=4)
def _dense(model: str, cache_dir: Path | None):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model, **_cache(cache_dir))


@lru_cache(maxsize=4)
def _sparse(model: str, cache_dir: Path | None):
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=model, **_cache(cache_dir))


@lru_cache(maxsize=4)
def _cross(model: str, cache_dir: Path | None):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model, **_cache(cache_dir))


class DenseEncoder:
    def __init__(self, model: str, cache_dir: Path | None = None):
        self.model_name = model
        self._model = _dense(model, cache_dir)
        self.prefix = next((p for k, p in _QUERY_PREFIX.items() if model.startswith(k)), "")

    @property
    def dim(self) -> int:
        return len(self.embed_queries(["dimension probe"])[0])

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
        with _lock:
            return list(self._model.passage_embed(texts, batch_size=batch_size))

    def embed_queries(self, texts: list[str]) -> list[np.ndarray]:
        with _lock:
            return list(self._model.embed([self.prefix + t for t in texts]))


class SparseEncoder:
    def __init__(self, model: str, cache_dir: Path | None = None):
        self.model_name = model
        self._model = _sparse(model, cache_dir)

    def embed_documents(self, texts: list[str]) -> list:
        with _lock:
            return list(self._model.passage_embed(texts))

    def embed_queries(self, texts: list[str]) -> list:
        with _lock:
            return list(self._model.query_embed(texts))


class CrossEncoder:
    def __init__(self, model: str, cache_dir: Path | None = None):
        self.model_name = model
        self._model = _cross(model, cache_dir)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        with _lock:
            return [float(s) for s in self._model.rerank(query, documents, batch_size=32)]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)
