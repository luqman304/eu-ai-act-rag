.PHONY: install test lint up down ingest eval-retrieval eval-generation eval-generation-cheap report

install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests

up:
	docker compose up -d --build api

down:
	docker compose down

ingest:
	.venv/bin/aiact ingest --strategies all

# Full retrieval grid: 5 chunkers x 3 retrieval modes x {no rerank, MiniLM cross-encoder}. No API key needed.
eval-retrieval:
	.venv/bin/aiact eval retrieval --strategies all --modes dense,sparse,hybrid \
	  --rerankers none,Xenova/ms-marco-MiniLM-L-6-v2 --baseline fixed/dense

# End-to-end eval: baseline vs shipped config, Claude as generator and judge (needs ANTHROPIC_API_KEY).
eval-generation:
	.venv/bin/aiact eval generation --configs "fixed/dense/no;structural/hybrid/no/no"

# Same eval on the cheapest model (about $0.30 for all 88 questions).
eval-generation-cheap:
	LLM_MODEL=claude-haiku-5-5 JUDGE_MODEL=claude-haiku-5-5 JUDGE_EFFORT=low $(MAKE) eval-generation

report:
	.venv/bin/aiact eval report --baseline fixed/dense
