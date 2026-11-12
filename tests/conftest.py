"""Shared fixtures. Tests run offline: hashing encoders stand in for the ONNX models and a scripted LLM
stands in for Claude, so the suite needs neither model downloads nor an API key."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np
import pytest

from aiact_rag.config import Settings, get_settings
from aiact_rag.corpus import load_units
from aiact_rag.llm import LLM, LLMResponse
from aiact_rag.pipeline import RAGService

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _bucket(tok: str, n: int) -> int:
    return int(hashlib.md5(tok.encode()).hexdigest(), 16) % n


class HashDense:
    model_name = "hash-dense"
    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in _tokens(text):
            v[_bucket(t, self.dim)] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v + 1e-6

    def embed_documents(self, texts, batch_size=32):
        return [self._vec(t) for t in texts]

    def embed_queries(self, texts):
        return [self._vec(t) for t in texts]


@dataclass
class _Sparse:
    indices: np.ndarray
    values: np.ndarray


class HashSparse:
    model_name = "hash-sparse"

    def _vec(self, text: str) -> _Sparse:
        counts: dict[int, float] = {}
        for t in _tokens(text):
            i = _bucket(t, 1 << 20)
            counts[i] = counts.get(i, 0.0) + 1.0
        idx = np.array(sorted(counts), dtype=np.int64)
        return _Sparse(indices=idx, values=np.array([counts[i] for i in idx], dtype=np.float32))

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_queries(self, texts):
        return [self._vec(t) for t in texts]


class ScriptedLLM(LLM):
    """Returns schema-shaped JSON. Answers cite excerpt [1]; judges call every claim supported."""

    def __init__(self, settings: Settings):
        super().__init__(settings.model_copy(update={"llm_cache": False}))
        self.calls: list[dict] = []

    def complete(self, *, system, user, model, effort, max_tokens=8000, schema=None):
        self.calls.append({"system": system, "user": user, "schema": schema})
        props = set((schema or {}).get("properties", {}))
        if props == {"answerable", "answer"}:
            unanswerable = "GDPR" in user
            data = {
                "answerable": not unanswerable,
                "answer": "Not covered." if unanswerable else "It is prohibited [1].",
            }
        elif "claims" in props:
            data = {
                "claims": [{"claim": "It is prohibited.", "verdict": "supported", "evidence": "prohibited"}]
            }
        elif "questions" in props:
            data = {
                "questions": ["Is it prohibited?", "Is this practice banned?", "Is it allowed?"],
                "noncommittal": False,
            }
        elif "points" in props:
            data = {
                "points": [
                    {"point": "prohibited", "status": "covered"},
                    {"point": "exception", "status": "missing"},
                ]
            }
        else:
            data = {}
        return LLMResponse(
            text=json.dumps(data), model="scripted", usage={"input_tokens": 10, "output_tokens": 5}
        )


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings().model_copy(
        update={"qdrant_url": ":memory:", "default_rerank": False, "llm_cache": False}
    )


@pytest.fixture(scope="session")
def units(settings):
    return load_units(settings.corpus_path)


@pytest.fixture(scope="session")
def units_by_id(units):
    return {u.id: u for u in units}


@pytest.fixture()
def service(settings) -> RAGService:
    svc = RAGService(settings, llm=ScriptedLLM(settings))
    svc.__dict__["dense"] = HashDense()
    svc.__dict__["sparse"] = HashSparse()
    return svc


@pytest.fixture()
def indexed_service(service) -> RAGService:
    service.ingest(["structural"])
    return service
