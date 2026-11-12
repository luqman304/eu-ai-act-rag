"""End-to-end plumbing of both runners and the report, with the scripted LLM."""

from aiact_rag.eval.dataset import load_gold
from aiact_rag.eval.report import generation_table, retrieval_tables, style_breakdown
from aiact_rag.eval.runner import retrieval_grid, run_generation_eval, run_retrieval_eval
from aiact_rag.retrieval import RetrievalConfig


def test_retrieval_runner_and_report(indexed_service, settings, tmp_path):
    items = load_gold(settings.gold_path)[:12]
    configs = retrieval_grid(["structural"], ["dense", "hybrid"], [None], [True, False])
    payload = run_retrieval_eval(indexed_service, items, configs, tmp_path)
    assert (tmp_path / "retrieval.json").exists()
    assert len(payload["runs"]) == 4
    row = payload["runs"][0]["items"][0]
    assert 0 <= row["recall@5"] <= 1 and row["retrieved"]
    md = retrieval_tables(payload, baseline="structural/dense")
    assert "| structural | hybrid |" in md and "Δ vs baseline" in md
    assert "paraphrase" in style_breakdown(payload, ["structural/hybrid"])


def test_generation_runner_and_report(indexed_service, settings, tmp_path):
    items = [i for i in load_gold(settings.gold_path) if i.id in {"q002", "q019", "q085"}]
    cfg = RetrievalConfig(strategy="structural", mode="hybrid", rerank=False, k=4)
    payload = run_generation_eval(indexed_service, items, [cfg], tmp_path, concurrency=2)
    rows = {r["id"]: r for r in payload["runs"][0]["items"]}
    assert not any("error" in r for r in rows.values())
    assert rows["q002"]["faithfulness"] == 1.0 and rows["q002"]["correctness"] == 0.5
    assert rows["q085"]["abstention_correct"] is True  # scripted LLM abstains on the GDPR question
    md = generation_table(payload)
    assert "1/1" in md  # abstention accuracy on the unanswerable item
