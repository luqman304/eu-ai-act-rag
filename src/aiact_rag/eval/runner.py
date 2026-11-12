"""Experiment runners: retrieval grid (no LLM, free) and end-to-end generation eval (LLM-judged)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from ..pipeline import RAGService
from ..retrieval import RetrievalConfig
from .dataset import GoldItem
from .generation_metrics import Judge
from .retrieval_metrics import all_metrics

log = logging.getLogger(__name__)

KS = (1, 3, 5, 10)

# List prices (USD / 1M input, output tokens), used only for the cost line printed after a run.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def retrieval_grid(
    strategies: Iterable[str],
    modes: Iterable[str],
    rerankers: Iterable[str | None],
    recitals: Iterable[bool] = (True,),
    candidates: int = 40,
) -> list[RetrievalConfig]:
    return [
        RetrievalConfig(
            strategy=s,
            mode=m,
            rerank=r is not None,
            rerank_model=r,
            k=max(KS),
            candidates=candidates,
            recitals=rc,
        )
        for s, m, r, rc in product(strategies, modes, rerankers, recitals)
    ]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def _config_dict(cfg: RetrievalConfig) -> dict[str, Any]:
    return {**asdict(cfg), "name": cfg.name}


def run_retrieval_eval(
    svc: RAGService,
    items: list[GoldItem],
    configs: list[RetrievalConfig],
    out_dir: Path,
    tag: str | None = None,
) -> dict[str, Any]:
    items = [it for it in items if it.answerable]
    runs = []
    for cfg in configs:
        t0 = time.perf_counter()
        per_item = []
        for it in items:
            res = svc.retrieve(it.question, cfg)
            ranked = [r.chunk.unit_ids for r in res.items]
            per_item.append(
                {
                    "id": it.id,
                    "style": it.style,
                    "category": it.category,
                    "gold": it.gold_units,
                    "retrieved": [
                        {"chunk": r.chunk.id, "label": r.chunk.label, "units": r.chunk.unit_ids}
                        for r in res.items
                    ],
                    "latency_ms": sum(res.timings_ms.values()),
                    **all_metrics(ranked, it.gold_units, KS),
                }
            )
        elapsed = time.perf_counter() - t0
        log.info("%-55s %.1fs", cfg.name, elapsed)
        runs.append({"config": _config_dict(cfg), "items": per_item})
    payload = {
        "kind": "retrieval",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_items": len(items),
        "models": {"dense": svc.settings.dense_model, "sparse": svc.settings.sparse_model},
        "runs": runs,
    }
    payload["tag"] = tag
    _write(out_dir / (f"retrieval_{tag}.json" if tag else "retrieval.json"), payload)
    return payload


def _eval_one(svc: RAGService, judge: Judge, cfg: RetrievalConfig, it: GoldItem) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": it.id,
        "style": it.style,
        "category": it.category,
        "answerable_gold": it.answerable,
    }
    try:
        res = svc.retrieve(it.question, cfg)
        ans = svc.generator.answer(it.question, res)
        scores = judge.score(ans, it.gold_units, it.reference_answer, it.answerable)
        ranked = [r.chunk.unit_ids for r in res.items]
        row.update(
            {
                "answer": ans.answer,
                "answerable_pred": ans.answerable,
                "citations": [c.label for c in ans.citations],
                "contexts": [r.chunk.label for r in res.items],
                "usage": ans.usage,
                "latency_ms": sum(ans.timings_ms.values()),
                **(
                    {f"recall@{cfg.k}": all_metrics(ranked, it.gold_units, (cfg.k,))[f"recall@{cfg.k}"]}
                    if it.gold_units
                    else {}
                ),
                **asdict(scores),
            }
        )
    except Exception as e:  # keep the run going; failures are reported, not hidden
        log.exception("item %s failed", it.id)
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def run_generation_eval(
    svc: RAGService,
    items: list[GoldItem],
    configs: list[RetrievalConfig],
    out_dir: Path,
    concurrency: int = 8,
) -> dict[str, Any]:
    judge = Judge(svc.llm, svc.settings, svc.dense)
    price = PRICES.get(svc.settings.llm_model, (float("nan"), float("nan")))
    runs = []
    for cfg in configs:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(lambda it, c=cfg: _eval_one(svc, judge, c, it), items))
        tokens_in = sum(r.get("usage", {}).get("input_tokens", 0) for r in rows)
        tokens_out = sum(r.get("usage", {}).get("output_tokens", 0) for r in rows)
        log.info(
            "%-55s %.0fs, answer tokens in/out %d/%d (~$%.2f excl. judge)",
            cfg.name,
            time.perf_counter() - t0,
            tokens_in,
            tokens_out,
            tokens_in / 1e6 * price[0] + tokens_out / 1e6 * price[1],
        )
        runs.append({"config": _config_dict(cfg), "items": rows})
    payload = {
        "kind": "generation",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_items": len(items),
        "models": {"generator": svc.settings.llm_model, "judge": svc.settings.judge_model},
        "efforts": {"answer": svc.settings.answer_effort, "judge": svc.settings.judge_effort},
        "runs": runs,
    }
    _write(out_dir / "generation.json", payload)
    return payload
