"""Command-line interface: `aiact --help`."""

from __future__ import annotations

import json
import logging
import sys
from typing import Annotated

import typer

from .config import get_settings

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="EU AI Act RAG: ingest, query and evaluate."
)
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation harness.")
app.add_typer(eval_app, name="eval")

ALL_STRATEGIES = ["fixed", "fixed_overlap", "recursive", "structural_bare", "structural"]


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpx2", "httpcore", "anthropic", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _service():
    from .pipeline import RAGService

    return RAGService(get_settings())


@app.command()
def fetch(
    force: Annotated[bool, typer.Option(help="Re-download even if the HTML is cached.")] = False,
) -> None:
    """Download the regulation from EUR-Lex and parse it into data/corpus/aiact.jsonl."""
    from .corpus.fetch import build_corpus, download

    _setup_logging()
    s = get_settings()
    raw = s.data_dir / "raw" / "aiact.html"
    if force or not raw.exists():
        download(s.corpus_url, raw)
    n = build_corpus(s, raw)
    typer.echo(f"parsed {n} units → {s.corpus_path}")


@app.command()
def ingest(
    strategies: Annotated[
        str, typer.Option(help="Comma-separated chunking strategies, or 'all'.")
    ] = "structural",
    sizes: Annotated[str, typer.Option(help="Comma-separated chunk sizes in words.")] = "250",
    skip_existing: Annotated[bool, typer.Option(help="Keep collections that already exist.")] = False,
) -> None:
    """Chunk, embed and index the corpus into Qdrant (one collection per strategy/size)."""
    _setup_logging()
    names = ALL_STRATEGIES if strategies == "all" else _csv(strategies)
    out = _service().ingest(names, [int(x) for x in _csv(sizes)], skip_existing=skip_existing)
    for name, n in out.items():
        typer.echo(f"{name:24s} {n:5d} chunks")


@app.command()
def search(
    query: str,
    strategy: str | None = None,
    mode: str | None = None,
    rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
    k: int = 6,
) -> None:
    """Retrieve chunks without calling the LLM."""
    svc = _service()
    res = svc.retrieve(query, svc.default_config(strategy=strategy, mode=mode, rerank=rerank, k=k))
    for it in res.items:
        score = it.rerank_score if it.rerank_score is not None else it.first_stage_score
        typer.echo(f"{it.rank:2d}. [{score:7.3f}] {it.chunk.label:40s} {it.chunk.text[:110]!r}")
    typer.echo(f"timings: { {k: round(v, 1) for k, v in res.timings_ms.items()} }")


@app.command()
def ask(question: str, strategy: str | None = None, k: int | None = None) -> None:
    """Answer a question with citations."""
    _setup_logging()
    svc = _service()
    ans = svc.ask(question, svc.default_config(strategy=strategy, k=k))
    typer.echo(ans.answer)
    typer.echo("")
    for c in ans.citations:
        typer.echo(f"[{c.n}] {c.label}  {c.url}")


@eval_app.command("check-gold")
def check_gold() -> None:
    """Verify every gold label against the official text (evidence spans must be present)."""
    from .eval.dataset import load_gold, validate_gold

    s = get_settings()
    svc = _service()
    items = load_gold(s.gold_path)
    problems = validate_gold(items, svc.units)
    for p in problems:
        typer.echo(p, err=True)
    typer.echo(f"{len(items)} items, {len(problems)} problems")
    raise typer.Exit(1 if problems else 0)


@eval_app.command("retrieval")
def eval_retrieval(
    strategies: Annotated[str, typer.Option(help="Index names (comma-separated) or 'all'.")] = "all",
    modes: str = "dense,sparse,hybrid",
    rerankers: Annotated[
        str, typer.Option(help="'none' and/or cross-encoder model names.")
    ] = "none,Xenova/ms-marco-MiniLM-L-6-v2",
    recitals: Annotated[str, typer.Option(help="'yes', 'no' or 'yes,no'.")] = "yes",
    candidates: int = 40,
    baseline: Annotated[str | None, typer.Option(help="Config name for paired deltas.")] = None,
    out: Annotated[str | None, typer.Option(help="Output directory (default results/).")] = None,
    ingest_missing: bool = True,
    tag: Annotated[str | None, typer.Option(help="Write results/retrieval_<tag>.json (ablations).")] = None,
) -> None:
    """Run the retrieval grid (no LLM calls) and write results/retrieval[_TAG].json."""
    from pathlib import Path

    from .chunking import parse_index_name
    from .eval.dataset import load_gold
    from .eval.runner import retrieval_grid, run_retrieval_eval

    _setup_logging()
    s = get_settings()
    svc = _service()
    names = ALL_STRATEGIES if strategies == "all" else _csv(strategies)
    if ingest_missing:
        for name in names:
            strat, size = parse_index_name(name)
            svc.ingest([strat], [size], skip_existing=True)
    rr = [None if r == "none" else r for r in _csv(rerankers)]
    rc = [r == "yes" for r in _csv(recitals)]
    configs = retrieval_grid(names, _csv(modes), rr, rc, candidates=candidates)
    out_dir = Path(out) if out else s.results_dir
    payload = run_retrieval_eval(svc, load_gold(s.gold_path), configs, out_dir, tag=tag)
    from .eval.report import retrieval_tables

    typer.echo(retrieval_tables(payload, baseline))


@eval_app.command("generation")
def eval_generation(
    configs: Annotated[
        str,
        typer.Option(help="Semicolon-separated configs 'strategy/mode/rerank(yes|no)[/recitals(yes|no)]'."),
    ] = "fixed/dense/no;structural/hybrid/yes",
    k: int = 6,
    limit: Annotated[
        int | None, typer.Option(help="Evaluate only the first N questions (smoke test).")
    ] = None,
    concurrency: int | None = None,
    out: str | None = None,
) -> None:
    """End-to-end eval: generate answers with Claude and score them with an LLM judge."""
    from pathlib import Path

    from .chunking import parse_index_name
    from .eval.dataset import load_gold
    from .eval.report import generation_table
    from .eval.runner import run_generation_eval
    from .retrieval import RetrievalConfig

    _setup_logging()
    s = get_settings()
    svc = _service()
    cfgs = []
    for spec in _csv(configs.replace(";", ",")):
        parts = spec.split("/")
        strat, mode, rerank = parts[0], parts[1], parts[2] == "yes"
        rec = parts[3] != "no" if len(parts) > 3 else True
        strategy, size = parse_index_name(strat)
        svc.ingest([strategy], [size], skip_existing=True)
        cfgs.append(
            RetrievalConfig(
                strategy=strat, mode=mode, rerank=rerank, k=k, candidates=s.rerank_candidates, recitals=rec
            )
        )
    items = load_gold(s.gold_path)[:limit] if limit else load_gold(s.gold_path)
    payload = run_generation_eval(
        svc,
        items,
        cfgs,
        Path(out) if out else s.results_dir,
        concurrency=concurrency or s.llm_max_concurrency,
    )
    typer.echo(generation_table(payload))
    from .eval.runner import PRICES

    for model, (tin, tout) in svc.llm.usage_by_model.items():
        pin, pout = PRICES.get(model, (float("nan"), float("nan")))
        cost = tin / 1e6 * pin + tout / 1e6 * pout
        typer.echo(f"{model}: {tin:,} input + {tout:,} output tokens (uncached) ≈ ${cost:.2f}")


@eval_app.command("report")
def eval_report(
    baseline: Annotated[str | None, typer.Option(help="Retrieval config name used as baseline.")] = None,
) -> None:
    """Regenerate results/REPORT.md from the JSON results."""
    from .eval.report import generation_failures, generation_table, load, retrieval_tables, style_breakdown

    s = get_settings()
    ret = load(s.results_dir / "retrieval.json")
    gen = load(s.results_dir / "generation.json")
    parts = ["# Evaluation report\n", "_Generated by `aiact eval report` from `results/*.json`._\n"]
    if ret:
        parts += ["## Retrieval\n", retrieval_tables(ret, baseline), ""]
        top = sorted(ret["runs"], key=lambda r: -sum(i["recall@5"] for i in r["items"]))[:3]
        names = [ret["runs"][0]["config"]["name"]] + [r["config"]["name"] for r in top]
        parts += ["### Recall@5 by question style\n", style_breakdown(ret, list(dict.fromkeys(names))), ""]
    for extra in sorted(s.results_dir.glob("retrieval_*.json")):
        payload = load(extra)
        parts += [f"## Ablation: {payload.get('tag') or extra.stem}\n", retrieval_tables(payload), ""]
    if gen:
        parts += [
            "## Generation\n",
            generation_table(gen),
            "",
            "### Unsupported claims (sample)\n",
            generation_failures(gen),
        ]
    path = s.results_dir / "REPORT.md"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    typer.echo(f"wrote {path}")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, workers: int = 1) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("aiact_rag.api.app:app", host=host, port=port, workers=workers, log_level="info")


@app.command()
def config() -> None:
    """Print the effective configuration (secrets redacted)."""
    data = get_settings().model_dump(mode="json")
    for key in ("api_token", "qdrant_api_key", "anthropic_api_key"):
        if data.get(key):
            data[key] = "***"
    typer.echo(json.dumps(data, indent=2))


if __name__ == "__main__":
    app()
