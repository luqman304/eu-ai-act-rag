import pytest
from fastapi.testclient import TestClient

from aiact_rag.api.app import create_app
from aiact_rag.eval.generation_metrics import Judge
from aiact_rag.generation import extract_citations
from aiact_rag.retrieval import RetrievalConfig


@pytest.fixture()
def client(indexed_service, settings, monkeypatch):
    monkeypatch.setenv("AIACT_WARMUP", "0")
    app = create_app(settings, service=indexed_service)
    with TestClient(app) as c:
        yield c


def test_hybrid_retrieval_finds_the_provision(indexed_service):
    cfg = RetrievalConfig(strategy="structural", mode="hybrid", rerank=False, k=5)
    res = indexed_service.retrieve(
        "untargeted scraping of facial images from the internet or CCTV footage", cfg
    )
    assert "art_5" in res.ranked_units()


def test_recital_filter(indexed_service):
    cfg = RetrievalConfig(strategy="structural", mode="dense", rerank=False, k=10, recitals=False)
    res = indexed_service.retrieve("emotion recognition at the workplace", cfg)
    assert res.items and not any(u.startswith("rct_") for u in res.ranked_units())


def test_citation_extraction(indexed_service):
    res = indexed_service.retrieve(
        "prohibited practices", RetrievalConfig(strategy="structural", rerank=False, k=3)
    )
    cites = extract_citations("A [1]. B [3][1]. Bogus [9].", res.items)
    assert [c.n for c in cites] == [1, 3]
    assert cites[0].url.startswith("https://eur-lex.europa.eu/")


def test_judge_scoring(indexed_service, settings):
    svc = indexed_service
    cfg = RetrievalConfig(strategy="structural", rerank=False, k=3)
    ans = svc.ask("Is social scoring prohibited?", cfg)
    judge = Judge(svc.llm, settings, svc.dense)
    s = judge.score(ans, ["art_5"], "Social scoring is prohibited, with exceptions.", answerable=True)
    assert s.faithfulness == 1.0
    assert s.correctness == 0.5
    assert 0 < s.answer_relevance <= 1
    unans = svc.ask("What is the GDPR fine?", cfg)
    s2 = judge.score(unans, [], "Not covered.", answerable=False)
    assert s2.abstained and s2.abstention_correct and s2.faithfulness is None


def test_api_endpoints(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["default_index_ready"]

    r = client.post("/v1/search", json={"query": "AI literacy of staff", "k": 3, "mode": "hybrid"})
    assert r.status_code == 200 and len(r.json()["results"]) == 3

    r = client.post("/v1/ask", json={"question": "Is social scoring prohibited?"})
    body = r.json()
    assert r.status_code == 200 and body["answerable"] and body["citations"][0]["n"] == 1
    assert r.headers["x-request-id"]

    assert client.get("/v1/provisions/art_5").json()["title"] == "Prohibited AI practices"
    assert client.get("/v1/provisions/art_999").status_code == 404
    assert client.post("/v1/ask", json={"question": "x"}).status_code == 422  # too short
    assert client.post("/v1/search", json={"query": "abc", "strategy": "nope"}).status_code == 400
    assert client.get("/").status_code == 200


def test_api_auth(indexed_service, settings, monkeypatch):
    monkeypatch.setenv("AIACT_WARMUP", "0")
    app = create_app(settings.model_copy(update={"api_token": "s3cret"}), service=indexed_service)
    with TestClient(app) as c:
        assert c.post("/v1/search", json={"query": "AI literacy"}).status_code == 401
        ok = c.post("/v1/search", json={"query": "AI literacy"}, headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        assert c.get("/readyz").status_code == 200  # probes stay unauthenticated
