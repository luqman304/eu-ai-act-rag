"""HTTP API.

GET  /healthz                 liveness
GET  /readyz                  readiness: Qdrant reachable and default index populated
POST /v1/search               retrieval only (no LLM)
POST /v1/ask                  grounded answer with citations
GET  /v1/provisions/{id}      full text of an article / annex / recital (e.g. art_5, anx_III)
GET  /                        minimal web UI
"""

from __future__ import annotations

import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .. import __version__
from ..config import Settings, get_settings
from ..llm import LLMError, LLMRefusal
from ..pipeline import RAGService
from ..retrieval import Retrieved
from .schemas import (
    AskRequest,
    AskResponse,
    ChunkOut,
    CitationOut,
    HealthResponse,
    ParagraphOut,
    ProvisionResponse,
    RetrievalOptions,
    SearchRequest,
    SearchResponse,
)

log = logging.getLogger("aiact_rag.api")
STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.svc = service or RAGService(settings)
        if os.getenv("AIACT_WARMUP", "1") == "1":
            try:  # load ONNX models up-front so the first request is not slow
                svc = app.state.svc
                svc.dense.embed_queries(["warm up"])
                svc.sparse.embed_queries(["warm up"])
                if settings.default_rerank:
                    svc.retriever.reranker().score("warm up", ["warm up"])
            except Exception:  # pragma: no cover - readiness reports the problem
                log.exception("warm-up failed")
        yield

    app = FastAPI(
        title="EU AI Act RAG",
        version=__version__,
        description="Grounded Q&A over Regulation (EU) 2024/1689 with verifiable citations.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
    )

    bearer = HTTPBearer(auto_error=False)

    def auth(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if settings.api_token is None:
            return
        if creds is None or not secrets.compare_digest(creds.credentials, settings.api_token):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    def svc_dep(request: Request) -> RAGService:
        return request.app.state.svc

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["x-request-id"] = rid
        log.info("%s %s %d %.0fms rid=%s", request.method, request.url.path, response.status_code, ms, rid)
        return response

    @app.exception_handler(LLMRefusal)
    async def _refusal(_: Request, exc: LLMRefusal):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(LLMError)
    async def _llm_error(_: Request, exc: LLMError):
        return JSONResponse(status_code=502, content={"detail": f"LLM backend error: {exc}"})

    def _cfg(svc: RAGService, opts: RetrievalOptions):
        cfg = svc.default_config(
            strategy=opts.strategy, mode=opts.mode, rerank=opts.rerank, k=opts.k, recitals=opts.recitals
        )
        available = svc.index.available()
        if not available.get(cfg.strategy):
            raise HTTPException(
                status_code=400,
                detail=f"index {cfg.strategy!r} is not available; indexed: {sorted(k for k, v in available.items() if v)}",
            )
        return cfg

    def _chunk_out(it: Retrieved) -> ChunkOut:
        return ChunkOut(
            rank=it.rank,
            chunk_id=it.chunk.id,
            label=it.chunk.label,
            unit_ids=it.chunk.unit_ids,
            text=it.chunk.text,
            score=it.first_stage_score,
            rerank_score=it.rerank_score,
        )

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", response_model=HealthResponse)
    def readyz(svc: RAGService = Depends(svc_dep)):
        try:
            cols = svc.index.available()
        except Exception as e:
            log.warning("qdrant not reachable: %s", e)
            cols = {}
        ready = bool(cols.get(settings.default_strategy))
        body = HealthResponse(
            status="ok" if ready else "degraded",
            collections=cols,
            default_index_ready=ready,
            llm_configured=bool(
                settings.anthropic_api_key
                or os.getenv("ANTHROPIC_API_KEY")
                or os.getenv("ANTHROPIC_AUTH_TOKEN")
            ),
            version=__version__,
        )
        return JSONResponse(status_code=200 if ready else 503, content=body.model_dump())

    @app.post("/v1/search", response_model=SearchResponse, dependencies=[Depends(auth)])
    def search(req: SearchRequest, svc: RAGService = Depends(svc_dep)) -> SearchResponse:
        cfg = _cfg(svc, req)
        res = svc.retrieve(req.query, cfg)
        return SearchResponse(
            config=cfg.name, results=[_chunk_out(i) for i in res.items], timings_ms=res.timings_ms
        )

    @app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(auth)])
    def ask(req: AskRequest, svc: RAGService = Depends(svc_dep)) -> AskResponse:
        cfg = _cfg(svc, req)
        ans = svc.ask(req.question, cfg)
        return AskResponse(
            answer=ans.answer,
            answerable=ans.answerable,
            citations=[
                CitationOut(n=c.n, label=c.label, unit_ids=c.unit_ids, url=c.url) for c in ans.citations
            ],
            contexts=[_chunk_out(i) for i in ans.contexts],
            config=cfg.name,
            model=ans.model,
            usage=ans.usage,
            timings_ms=ans.timings_ms,
        )

    @app.get("/v1/provisions/{unit_id}", response_model=ProvisionResponse, dependencies=[Depends(auth)])
    def provision(unit_id: str, svc: RAGService = Depends(svc_dep)) -> ProvisionResponse:
        u = svc.units.get(unit_id)
        if u is None:
            raise HTTPException(
                status_code=404, detail=f"unknown provision {unit_id!r} (try art_5, anx_III, rct_27)"
            )
        return ProvisionResponse(
            id=u.id,
            kind=u.kind,
            label=u.label,
            title=u.title,
            heading=u.heading,
            url=u.url,
            paragraphs=[ParagraphOut(**p.model_dump()) for p in u.paragraphs],
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
