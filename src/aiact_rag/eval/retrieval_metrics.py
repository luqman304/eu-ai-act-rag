"""Provision-level retrieval metrics.

Inputs are the ranked top-k *chunks* (as lists of unit ids each chunk contains) and the gold unit ids.
Measuring at the chunk cut-off is what matters for RAG: the generator sees exactly k chunks, so
recall@k answers "how much of the needed law reached the model?".
"""

from __future__ import annotations

import math
from collections.abc import Sequence

ChunkUnits = Sequence[Sequence[str]]


def recall_at_k(ranked: ChunkUnits, gold: Sequence[str], k: int) -> float:
    if not gold:
        raise ValueError("recall is undefined without gold units")
    found = {u for chunk in ranked[:k] for u in chunk}
    return len(found & set(gold)) / len(set(gold))


def hit_at_k(ranked: ChunkUnits, gold: Sequence[str], k: int) -> float:
    g = set(gold)
    return float(any(g & set(chunk) for chunk in ranked[:k]))


def precision_at_k(ranked: ChunkUnits, gold: Sequence[str], k: int) -> float:
    """Share of the top-k chunks that contain at least one gold unit (a.k.a. context precision)."""
    top = ranked[:k]
    if not top:
        return 0.0
    g = set(gold)
    return sum(1 for chunk in top if g & set(chunk)) / len(top)


def reciprocal_rank(ranked: ChunkUnits, gold: Sequence[str], k: int | None = None) -> float:
    g = set(gold)
    for i, chunk in enumerate(ranked[:k] if k else ranked, start=1):
        if g & set(chunk):
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: ChunkUnits, gold: Sequence[str], k: int) -> float:
    """nDCG where a chunk's gain is the number of *new* gold units it brings (redundant chunks earn 0)."""
    g = set(gold)
    seen: set[str] = set()
    dcg = 0.0
    for i, chunk in enumerate(ranked[:k], start=1):
        new = (g & set(chunk)) - seen
        seen |= new
        dcg += len(new) / math.log2(i + 1)
    # Ideal: one gold unit per position at the top.
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(g), k) + 1))
    return dcg / idcg if idcg else 0.0


def all_metrics(
    ranked: ChunkUnits, gold: Sequence[str], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, gold, k)
    out["hit@5"] = hit_at_k(ranked, gold, 5)
    out["precision@5"] = precision_at_k(ranked, gold, 5)
    out["mrr@10"] = reciprocal_rank(ranked, gold, 10)
    out["ndcg@10"] = ndcg_at_k(ranked, gold, 10)
    return out
