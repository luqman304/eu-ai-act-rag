import math

import pytest

from aiact_rag.eval.retrieval_metrics import (
    all_metrics,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from aiact_rag.eval.stats import mean_ci, paired_delta_ci

RANKED = [["rct_1"], ["art_5"], ["art_5"], ["art_6", "art_7"], ["anx_III"]]


def test_recall():
    assert recall_at_k(RANKED, ["art_5"], 1) == 0
    assert recall_at_k(RANKED, ["art_5"], 2) == 1
    assert recall_at_k(RANKED, ["art_6", "anx_III"], 4) == 0.5
    assert recall_at_k(RANKED, ["art_6", "anx_III"], 5) == 1
    with pytest.raises(ValueError):
        recall_at_k(RANKED, [], 5)


def test_hit_precision_rr():
    assert hit_at_k(RANKED, ["art_7"], 3) == 0 and hit_at_k(RANKED, ["art_7"], 4) == 1
    assert precision_at_k(RANKED, ["art_5"], 5) == pytest.approx(2 / 5)
    assert reciprocal_rank(RANKED, ["art_6"]) == pytest.approx(1 / 4)
    assert reciprocal_rank(RANKED, ["art_99"]) == 0


def test_ndcg_ignores_redundant_chunks():
    # second art_5 chunk adds no new gold unit, so it must not inflate DCG
    got = ndcg_at_k(RANKED, ["art_5"], 10)
    assert got == pytest.approx((1 / math.log2(3)) / 1.0)
    assert ndcg_at_k([["a"], ["b"]], ["a", "b"], 2) == pytest.approx(1.0)


def test_all_metrics_keys():
    m = all_metrics(RANKED, ["art_5"])
    assert set(m) == {
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "hit@5",
        "precision@5",
        "mrr@10",
        "ndcg@10",
    }


def test_bootstrap():
    m, lo, hi = mean_ci([1, 0, 1, 1, 0, 1, 1, 1])
    assert lo <= m <= hi and m == pytest.approx(0.75)
    d, lo, hi, p = paired_delta_ci([0] * 30, [1] * 30)
    assert d == 1 and lo == 1 and p == 0
    d, *_, p = paired_delta_ci([0, 1] * 15, [0, 1] * 15)
    assert d == 0 and p == 1
