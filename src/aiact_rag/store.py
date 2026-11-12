"""Qdrant index: one collection per chunking strategy, with a named dense vector and a BM25 sparse vector."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from .chunking import Chunk
from .config import Settings
from .embeddings import DenseEncoder, SparseEncoder

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"


def make_client(settings: Settings) -> QdrantClient:
    url = settings.qdrant_url
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if url.startswith(("http://", "https://")):
        return QdrantClient(url=url, api_key=settings.qdrant_api_key, timeout=30)
    return QdrantClient(path=url)  # embedded on-disk mode


@dataclass
class Hit:
    chunk: Chunk
    score: float


class ChunkIndex:
    def __init__(self, client: QdrantClient, settings: Settings, dense: DenseEncoder, sparse: SparseEncoder):
        self.client = client
        self.settings = settings
        self.dense = dense
        self.sparse = sparse

    def build(self, strategy: str, chunks: list[Chunk], batch_size: int = 64, recreate: bool = True) -> int:
        name = self.settings.collection(strategy)
        if recreate and self.client.collection_exists(name):
            self.client.delete_collection(name)
        if not self.client.collection_exists(name):
            self.client.create_collection(
                name,
                vectors_config={
                    DENSE: models.VectorParams(size=self.dense.dim, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            with warnings.catch_warnings():  # local mode ignores payload indexes and warns about it
                warnings.simplefilter("ignore")
                for field in ("unit_ids", "kinds"):
                    self.client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.text for c in batch]
            dvecs = self.dense.embed_documents(texts)
            svecs = self.sparse.embed_documents(texts)
            points = [
                models.PointStruct(
                    id=start + i,
                    vector={
                        DENSE: dv.tolist(),
                        SPARSE: models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
                    },
                    payload=c.model_dump(),
                )
                for i, (c, dv, sv) in enumerate(zip(batch, dvecs, svecs, strict=True))
            ]
            self.client.upsert(name, points=points, wait=True)
        log.info("indexed %d chunks into %s", len(chunks), name)
        return len(chunks)

    def available(self) -> dict[str, int]:
        out: dict[str, int] = {}
        prefix = f"{self.settings.collection_prefix}__"
        for c in self.client.get_collections().collections:
            if c.name.startswith(prefix):
                out[c.name.removeprefix(prefix)] = self.client.count(c.name, exact=True).count
        return out

    def search(self, strategy: str, query: str, mode: str, limit: int, recitals: bool = True) -> list[Hit]:
        name = self.settings.collection(strategy)
        flt = None
        if not recitals:
            flt = models.Filter(
                must_not=[models.FieldCondition(key="kinds", match=models.MatchValue(value="recital"))]
            )
        common = {"limit": limit, "with_payload": True, "query_filter": flt}
        if mode == "dense":
            res = self.client.query_points(name, query=self._dense_q(query), using=DENSE, **common)
        elif mode == "sparse":
            res = self.client.query_points(name, query=self._sparse_q(query), using=SPARSE, **common)
        elif mode == "hybrid":
            pre = max(limit, 50)
            res = self.client.query_points(
                name,
                prefetch=[
                    models.Prefetch(query=self._dense_q(query), using=DENSE, limit=pre, filter=flt),
                    models.Prefetch(query=self._sparse_q(query), using=SPARSE, limit=pre, filter=flt),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                **common,
            )
        else:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        return [Hit(chunk=Chunk.model_validate(p.payload), score=float(p.score)) for p in res.points]

    def _dense_q(self, query: str) -> list[float]:
        return self.dense.embed_queries([query])[0].tolist()

    def _sparse_q(self, query: str) -> models.SparseVector:
        sv = self.sparse.embed_queries([query])[0]
        return models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
