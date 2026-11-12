# syntax=docker/dockerfile:1.7
# Build stage: dependencies and models
FROM python:3.12-slim-bookworm AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Bake the ONNX models into the image so containers start offline and fast.
ENV MODEL_CACHE_DIR=/models
RUN python - <<'EOF'
from aiact_rag.config import get_settings
from aiact_rag.embeddings import CrossEncoder, DenseEncoder, SparseEncoder
s = get_settings()
DenseEncoder(s.dense_model, s.model_cache_dir).embed_queries(["warm"])
SparseEncoder(s.sparse_model, s.model_cache_dir).embed_queries(["warm"])
CrossEncoder(s.rerank_model, s.model_cache_dir).score("warm", ["warm"])
EOF

# Runtime stage
FROM python:3.12-slim-bookworm
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/models \
    DATA_DIR=/app/data \
    RESULTS_DIR=/app/results \
    QDRANT_URL=http://qdrant:6333 \
    HF_HUB_OFFLINE=1
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=app:app /models /models
COPY data/corpus ./data/corpus
COPY data/eval ./data/eval
COPY results ./results
RUN mkdir -p data/cache && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/healthz').status != 200)"
CMD ["uvicorn", "aiact_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
