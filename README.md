# EU AI Act RAG

**Grounded question answering over the EU Artificial Intelligence Act, with an evaluation harness that decides
how the system is built.**

Ask a question in plain English ("Can my employer use AI to infer how I feel at work?") and get a short answer
in which every sentence cites the exact article, paragraph or annex point it comes from, linked to the official
text on EUR-Lex. If the regulation does not answer the question, the system says so instead of guessing.

The retrieval pipeline was not picked by intuition. I built a labelled test set and measured 30+ retrieval
configurations (5 chunking strategies × 3 search modes × reranking on/off, plus ablations on recitals, chunk size
and reranker model). The shipped configuration is the one the numbers chose, and the README reports how much each
decision was worth, with confidence intervals.

`Python 3.12` · `FastAPI` · `Qdrant` (dense + BM25 hybrid search) · `fastembed` ONNX models · `Claude`
(answer generation and LLM-as-judge) · `Docker Compose` · `pytest` · `GitHub Actions`

---

## Contents

1. [Headline results](#headline-results)
2. [Architecture](#architecture)
3. [Design decisions](#design-decisions)
4. [Evaluation methodology](#evaluation-methodology)
5. [Full results](#full-results)
6. [Quick start](#quick-start)
7. [API reference](#api-reference)
8. [CLI reference](#cli-reference)
9. [Configuration](#configuration)
10. [Reproducing the evaluation](#reproducing-the-evaluation)
11. [Testing and CI](#testing-and-ci)
12. [Project layout](#project-layout)
13. [Limitations](#limitations)
14. [Next steps](#next-steps)

---

## Headline results

### Retrieval

84 answerable questions, labelled at provision level (for example `art_6` + `anx_III`). Recall is the share of
gold provisions that appear in the top-k retrieved chunks.

| configuration | recall@1 | recall@5 | recall@10 | MRR@10 | latency p50 |
|---|---:|---:|---:|---:|---:|
| baseline: fixed 250-word windows, dense search | 23.2 | 69.6 | 76.8 | 0.44 | 6 ms |
| structural chunks, dense search | 35.1 | 75.6 | 83.3 | 0.54 | 6 ms |
| structural chunks, hybrid search (dense + BM25, RRF) | 40.5 | 86.9 | 89.3 | 0.62 | 16 ms |
| **+ recitals excluded (shipped)** | **76.8** | **89.9** | **95.8** | **0.86** | **22 ms** |
| + bge-reranker-base (tested, not shipped) | 76.8 | 91.7 | 95.8 | 0.86 | 6.5 s |

**Shipped vs baseline: +20.2 recall@5, 95% CI [+11.9, +29.2].** Recall@1 more than triples (23.2 → 76.8), so the
most relevant provision is usually the first excerpt the model reads.

### Answer quality

All 88 questions (84 answerable + 4 out-of-scope), answered end to end and scored by an LLM judge. Run on Claude
Haiku 5.5 at low effort for both generator and judge; the whole run cost about $0.30.

| configuration | faithfulness | answer relevance | correctness | citation precision | out-of-scope declined | answerable wrongly declined |
|---|---:|---:|---:|---:|---:|---:|
| baseline: fixed, dense | 0.944 | 0.713 | 0.681 | 0.340 | 4/4 | 9/84 |
| **shipped: structural, hybrid, no recitals** | **0.964** | 0.731 | **0.851** | **0.665** | 4/4 | **5/84** |

| metric | change | 95% CI |
|---|---:|---|
| correctness | **+17.0 points** | [+10.1, +23.8] |
| citation precision | **+31.7 points** | [+24.9, +38.5] |
| faithfulness | +2.1 points | [+0.1, +4.2] |
| answer relevance | +1.8 points | not significant |

**What this shows:** the model stays faithful to its context in both setups, so weak retrieval produces
*incomplete* answers rather than *invented* ones. Better retrieval therefore shows up mainly as correctness and
citation quality, which is exactly where the gains are.

### Key findings

1. **Structure-aware chunking wins in every search mode.** Splitting on the regulation's own paragraphs and points
   beats fixed windows, overlapping windows and recursive (LangChain-style) splitting.
2. **A breadcrumb header is worth +6.5 recall@5 on its own** (95% CI [+0.6, +13.1]). Measured by comparing two
   chunkers with identical boundaries, one with the header and one without.
3. **Recitals crowd out binding law.** The 180 recitals (the non-binding preamble) took about half of every top-5.
   Excluding them nearly doubled recall@1 (40.5 → 76.8).
4. **Reranking depends on chunking.** A small cross-encoder (MiniLM-L6) *cut* recall@5 by 17.3 points on fixed
   windows. A stronger one (bge-reranker-base) helps structural chunks only slightly (+1.8, CI [−4.2, +7.1]) and
   costs ~6 s per query on CPU, so it is available but off by default.
5. **Hybrid search matters most on paraphrased questions.** Dense-only retrieval misses exact legal terms; BM25
   misses paraphrases; reciprocal-rank fusion gets both.

---

## Architecture

```
                      ┌─────────────────────────────────────────────────────────────┐
  EUR-Lex XHTML ────► │ parser       306 units: 180 recitals · 113 articles · 13    │
  (official text)     │              annexes, each split into numbered paragraphs   │
                      └───────────────────────────┬─────────────────────────────────┘
                                                  ▼
                      ┌─────────────────────────────────────────────────────────────┐
                      │ chunker      fixed · fixed_overlap · recursive ·            │
                      │              structural · structural_bare                   │
                      │              every chunk records which provisions it holds  │
                      └───────────────────────────┬─────────────────────────────────┘
                                                  ▼
                      ┌─────────────────────────────────────────────────────────────┐
                      │ Qdrant       one collection per chunker/size                │
                      │              named dense vector (bge-small-en-v1.5)         │
                      │              + sparse BM25 vector (IDF-weighted)            │
                      └───────────────────────────┬─────────────────────────────────┘
                                                  ▼
   question ──► first stage: dense | BM25 | hybrid (RRF fusion inside Qdrant), 40 candidates,
                optional payload filter to drop recitals
                                                  ▼
                optional cross-encoder rerank (bge-reranker-base) ──► top-k excerpts
                                                  ▼
                Claude, structured output {answerable, answer} with [n] citations
                                                  ▼
                citations resolved to provisions ──► EUR-Lex deep links
```

The same `RAGService` object backs the HTTP API, the CLI and the evaluation harness, so what is evaluated is
exactly what is served.

### Components

| module | responsibility |
|---|---|
| `corpus/fetch.py`, `corpus/parse.py` | Download the Official Journal XHTML and parse it into recitals, articles and annexes, each with numbered paragraphs, a heading breadcrumb and a EUR-Lex anchor URL. |
| `chunking.py` | Five chunking strategies. Every chunk records the provisions (`unit_ids`) and paragraphs it contains, which is what makes recall comparable across chunkers. |
| `embeddings.py` | Thin wrappers over `fastembed` ONNX models: dense (bge-small-en-v1.5 with its query instruction), sparse (BM25) and cross-encoders. No PyTorch, so the Docker image stays small and CPU-only. |
| `store.py` | Qdrant collections with a named dense vector and an IDF-modified sparse vector; dense, sparse and hybrid (server-side RRF) search; recital filter. Works against a server, an on-disk embedded store, or `:memory:` for tests. |
| `retrieval.py` | Two-stage retrieval: first-stage search for 40 candidates, optional cross-encoder rerank, cut to k. Records per-stage timings. |
| `generation.py`, `llm.py` | Prompting, structured output, citation extraction; Anthropic client with refusal fallback, error mapping, token accounting and an SQLite response cache. |
| `pipeline.py` | `RAGService`: lazily builds the components and exposes `ingest`, `retrieve` and `ask`. |
| `eval/` | Gold-set loader and validator, retrieval metrics, LLM judge, runners, paired-bootstrap statistics and Markdown report generation. |
| `api/` | FastAPI app, request/response schemas and a small single-page UI. |
| `cli.py` | `aiact` command: fetch, ingest, search, ask, serve, config, eval. |

---

## Design decisions

### Parsing the official text, not a PDF

EUR-Lex publishes the Official Journal as XHTML with stable ELI identifiers: `art_5`, `anx_III`, `rct_27`, and
numbered paragraph containers (`005.001` = Article 5(1)). Parsing that structure gives exact provision boundaries
and deep links for free. Edge cases the parser handles, each covered by a test:

- **Enumerations** are laid out as two-column tables (`(a)` | text). They are flattened to `(a) …` lines so the
  structure survives as plain text.
- **Footnote markers** (`<a><span class="oj-super">`) are dropped; **genuine superscripts** are kept as exponents,
  so Article 51's compute threshold reads `10^25` FLOPs rather than `1025`.
- **Annex I** stores its items as bare `<span>`s in table cells rather than `<p>`s. The first parser version
  silently produced an empty Annex I; the gold-set evidence check (below) is what caught it.

### Five chunkers, one budget

All chunkers use the same 250-word budget (≈330 tokens, well under bge-small's 512-token limit), so the comparison
is about *where* text is cut, not how much.

| strategy | boundaries | context header |
|---|---|---|
| `fixed` | 250-word windows over the whole regulation; windows can straddle articles | no |
| `fixed_overlap` | same, with 50-word overlap | no |
| `recursive` | per provision: split on newline → sentence → word, then greedily merge (the LangChain recipe) | no |
| `structural_bare` | packs whole paragraphs; oversized paragraphs are split on their enumerated points, repeating the lead-in sentence ("The following AI practices shall be prohibited:") in each piece | no |
| `structural` | same boundaries as `structural_bare` | yes: `Chapter III – High-risk AI systems › Section 1 › Article 6 – Classification rules… › Article 6(3)` |

`structural_bare` exists purely as an ablation: it isolates the value of the header from the value of the
boundaries.

### Hybrid search inside Qdrant

Each collection holds a named dense vector and a sparse BM25 vector with Qdrant's IDF modifier. Hybrid search
issues both as prefetches and fuses them with reciprocal-rank fusion in a single Qdrant query, so there is no
client-side merging and no score normalisation to tune. Legal text is full of exact terms ("provider",
"deployer", "post-remote biometric identification") that dense models blur and BM25 nails, while users ask in
paraphrase, which BM25 misses. Fusion gets both.

### Recitals off by default

Recitals explain the law's intent but impose no obligations. They are long, numerous (180 of 306 units) and use
the same vocabulary as the articles, so they dominate the top ranks. The shipped default filters them out with a
Qdrant payload filter; users can switch them back on per request (`"recitals": true`) when they want the
legislator's reasoning.

### Answers that can be checked

- The model receives numbered `<excerpt>` blocks and must cite every sentence as `[n]`.
- **Structured output** (`{"answerable": bool, "answer": str}` enforced by JSON schema) makes abstention a field
  rather than a phrase to pattern-match.
- Citations are resolved back to provisions and EUR-Lex anchors, so every claim is one click from the source.
- The system prompt forbids adding dates, thresholds or exceptions not present in the excerpts, even if the model
  "knows" them. That is what keeps faithfulness high when retrieval is weak.

### Production concerns

- **Refusal handling:** the server-side fallback is enabled by default; a refusal maps to HTTP 422, upstream
  errors to 502, never a bare 500.
- **Response cache:** an SQLite cache keyed on the full request makes evaluation re-runs free and reproducible.
- **Cost visibility:** token usage is returned per request and summed per model after each eval run.
- **Ops:** bearer-token auth, request IDs on every response, liveness (`/healthz`) and readiness (`/readyz`)
  probes, model warm-up at start-up, CORS configuration.
- **Container:** multi-stage build, ONNX models baked into the image so containers start offline, non-root user,
  built-in healthcheck. Compose runs Qdrant, a one-shot idempotent indexer, then the API.

---

## Evaluation methodology

### Gold set (`data/eval/gold.jsonl`)

88 hand-written questions: 84 answerable and 4 out-of-scope (GDPR fines, notified-body prices, US agencies,
parliamentary vote counts) that the system should decline.

Each item has:

| field | purpose |
|---|---|
| `question` | what a user would ask |
| `gold_units` | the provisions that answer it, e.g. `["art_6", "anx_III"]` |
| `reference_answer` | a human-written answer, used for the correctness metric |
| `evidence` | verbatim spans that must occur in the gold provisions |
| `style` | `lexical` (30), `paraphrase` (46) or `multi` (8) |

Three choices make the set trustworthy:

- **Labels are provisions, not chunks.** Chunk IDs differ for every chunker, so chunk-level labels could not
  compare them. Every chunk records which provisions it contains; a chunk counts as containing a provision if it
  holds at least 25 of that provision's words (or all of it, if shorter). This stops a fixed window from getting
  credit for a provision because one stray sentence spilled into it.
- **Labels are machine-checked.** `aiact eval check-gold` verifies every evidence span against the parsed text of
  its gold provisions, and CI fails if any are missing. This guards against mislabelling and against parser
  regressions (it caught the empty Annex I).
- **Difficulty is mixed on purpose.** `lexical` questions reuse the Act's wording; `paraphrase` questions are how
  practitioners actually ask; `multi` questions need two provisions together (Article 6 + Annex III for
  high-risk classification).

### Metrics

| metric | definition | needs an LLM |
|---|---|---|
| recall@k | share of gold provisions present in the top-k chunks | no |
| hit@5 | at least one gold provision in the top 5 | no |
| precision@5 | share of top-5 chunks containing a gold provision | no |
| MRR@10 | reciprocal rank of the first relevant chunk | no |
| nDCG@10 | gain only for *new* gold provisions, so a second chunk of the same article earns nothing | no |
| faithfulness | RAGAS-style: the judge splits the answer into atomic claims and marks each as supported or unsupported *by the excerpts alone* | judge |
| answer relevance | RAGAS-style: the judge writes 3 questions the answer would answer; score is the mean cosine similarity to the real question (bge-small); 0 if the answer is non-committal | judge |
| correctness | the judge splits the reference answer into key points: covered 1, partial 0.5, missing or contradicted 0 | judge |
| abstention | out-of-scope questions should get `answerable=false`; false abstentions are counted on answerable ones | no |
| citation precision | share of cited excerpts that contain a gold provision | no |

### Statistics

With 84 questions, small differences are noise. Every comparison is reported as a **paired bootstrap 95%
confidence interval** (5,000 resamples, fixed seed 1689 so reports are reproducible). Paired means both
configurations are scored on the same resampled questions, which removes question difficulty from the variance.
Per-question results, including the judge's individual claim verdicts, are saved in the result JSON so any score
can be audited.

---

## Full results

The complete tables, including recall@1/3/5/10, nDCG, precision and latency for every configuration and every
ablation, are in [`results/REPORT.md`](results/REPORT.md), generated from the JSON files in `results/`.

### Recall@5 across the main grid

| chunker | dense | sparse (BM25) | hybrid | dense + MiniLM rerank | hybrid + MiniLM rerank |
|---|---:|---:|---:|---:|---:|
| `fixed` | 69.6 | 75.0 | 79.8 | 60.7 | 62.5 |
| `fixed_overlap` | 63.1 | 78.0 | 75.6 | 65.5 | 66.7 |
| `recursive` | 69.0 | 76.8 | 82.7 | 74.4 | 76.8 |
| `structural_bare` | 69.0 | 81.5 | 80.4 | 75.6 | 75.6 |
| `structural` | 75.6 | 79.2 | **86.9** | 84.5 | **86.9** |

### Recall@5 by question style

| configuration | lexical (n=30) | paraphrase (n=46) | multi (n=8) |
|---|---:|---:|---:|
| baseline: fixed, dense | 80.0 | 65.2 | 56.2 |
| structural, hybrid | 86.7 | 89.1 | 75.0 |

The biggest gain is on paraphrased questions (+23.9), the ones real users ask.

### Chunk-size ablation (hybrid search, recitals included)

| size (words) | fixed | structural |
|---:|---:|---:|
| 120 | 80.4 | 88.7 |
| 175 | 76.8 | 87.5 |
| 250 | 79.8 | 86.9 |
| 350 | 74.4 | 83.3 |

Structural chunking wins at every size. Differences between sizes are inside the confidence intervals, so 250
words was kept: it gives the generator more context per excerpt at no measurable recall cost.

---

## Quick start

### Requirements

- Docker with Compose v2, **or** Python 3.12
- An Anthropic API key for `/v1/ask` and the answer-quality eval. Retrieval, search and the retrieval eval
  work without one.

### Run with Docker

```bash
cp .env.example .env
```

Add your `ANTHROPIC_API_KEY` to `.env`, then:

```bash
docker compose up -d --build api
```

This starts Qdrant, runs the one-shot indexer (chunk, embed and index the serving collection; it skips
collections that already exist), then starts the API. Open http://localhost:8000 for the UI and
http://localhost:8000/docs for the interactive OpenAPI docs.

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question": "Is AI used to screen job applications high-risk?"}' | jq '.answer, .citations'
```

### Run locally (no Docker)

```bash
make install
```

```bash
source .venv/bin/activate
```

Qdrant can run embedded on disk, so no server is needed:

```bash
export QDRANT_URL=data/qdrant
```

```bash
aiact ingest
```

```bash
aiact search "emotion recognition at work"
```

```bash
aiact ask "When is a general-purpose AI model presumed to have systemic risk?"
```

```bash
aiact serve
```

The parsed corpus is included in the repository. To rebuild it from EUR-Lex, run `aiact fetch --force`.

---

## API reference

| method and path | description |
|---|---|
| `POST /v1/ask` | Retrieve and answer, with citations |
| `POST /v1/search` | Retrieval only, no LLM call |
| `GET /v1/provisions/{id}` | Full text of a provision: `art_5`, `anx_III`, `rct_44`, … |
| `GET /healthz` | Liveness |
| `GET /readyz` | Readiness: Qdrant reachable, default index populated, LLM configured |
| `GET /` | Web UI |

### `POST /v1/ask`

Request (every field except `question` is optional and falls back to the server defaults):

```json
{
  "question": "Which AI practices are prohibited?",
  "k": 6,
  "strategy": "structural",
  "mode": "hybrid",
  "rerank": false,
  "recitals": false
}
```

Response:

```json
{
  "answer": "Article 5(1) prohibits ... [1][2]",
  "answerable": true,
  "citations": [
    {"n": 1, "label": "Article 5(1)", "unit_ids": ["art_5"],
     "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_5"}
  ],
  "contexts": [{"rank": 1, "chunk_id": "...", "label": "Article 5(1)", "unit_ids": ["art_5"],
                "text": "...", "score": 0.032, "rerank_score": null}],
  "config": "structural/hybrid-recitals",
  "model": "claude-opus-5-5",
  "usage": {"input_tokens": 2310, "output_tokens": 180},
  "timings_ms": {"search_ms": 22.1, "generate_ms": 2400.5},
  "disclaimer": "Information about the text of Regulation (EU) 2024/1689, not legal advice."
}
```

### Errors

| status | when |
|---|---|
| 400 | the requested index has not been ingested |
| 401 | `API_TOKEN` is set and the bearer token is missing or wrong |
| 404 | unknown provision id |
| 422 | invalid request (e.g. question shorter than 3 characters), or the model declined the request |
| 502 | upstream LLM error or no API key configured |
| 503 | `/readyz` only: Qdrant unreachable or default index empty |

Set `API_TOKEN` to require `Authorization: Bearer <token>` on all `/v1/*` routes. Health probes stay
unauthenticated so orchestrators can reach them. Every response carries an `X-Request-ID` header.

---

## CLI reference

```
aiact fetch [--force]                        download and parse the regulation
aiact ingest [--strategies S] [--sizes N]    chunk, embed and index ('all' for every chunker)
aiact search "query" [--strategy] [--mode] [--rerank/--no-rerank] [-k]
aiact ask "question" [--strategy] [-k]
aiact serve [--host] [--port] [--workers]
aiact config                                 print effective settings, secrets masked

aiact eval check-gold                        verify every gold label against the text
aiact eval retrieval [--strategies] [--modes] [--rerankers] [--recitals yes,no] [--baseline] [--tag]
aiact eval generation [--configs "strategy/mode/rerank[/recitals];..."] [--limit N]
aiact eval report [--baseline]               regenerate results/REPORT.md
```

Index names are `strategy` or `strategy-size`, e.g. `structural-120`. Generation configs use yes/no flags:
`structural/hybrid/no/no` is structural chunks, hybrid search, no rerank, recitals excluded.

---

## Configuration

All settings are environment variables (or entries in `.env`). Defaults shown.

| variable | default | meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | required for answers and the answer-quality eval |
| `API_TOKEN` | – | if set, required as a bearer token on `/v1/*` |
| `QDRANT_URL` | `http://localhost:6333` | server URL, a local path for embedded mode, or `:memory:` |
| `QDRANT_API_KEY` | – | for Qdrant Cloud |
| `LLM_MODEL` | `claude-opus-5-5` | answer model |
| `JUDGE_MODEL` | `claude-opus-5-5` | evaluation judge |
| `ANSWER_EFFORT` / `JUDGE_EFFORT` | `low` / `medium` | reasoning effort per role |
| `LLM_FALLBACKS` | `true` | server-side refusal fallback |
| `LLM_CACHE` | `true` | SQLite response cache in `data/cache/` |
| `LLM_MAX_CONCURRENCY` | `8` | parallel calls during evaluation |
| `DEFAULT_STRATEGY` | `structural` | index served by default |
| `DEFAULT_MODE` | `hybrid` | `dense`, `sparse` or `hybrid` |
| `DEFAULT_RERANK` | `false` | cross-encoder reranking |
| `DEFAULT_RECITALS` | `false` | include recitals in search |
| `DEFAULT_K` | `6` | excerpts passed to the model |
| `RERANK_CANDIDATES` | `40` | first-stage candidates before reranking |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | cross-encoder |
| `CORS_ORIGINS` | `["*"]` | allowed origins |
| `AIACT_WARMUP` | on | load models at start-up |

---

## Reproducing the evaluation

Every number in this README comes from files in `results/` and can be regenerated.

```bash
make eval-retrieval
```

Runs the 30-configuration grid. No API key needed; about 20 minutes on a laptop CPU.

```bash
make eval-generation-cheap
```

Runs the answer-quality eval on Claude Haiku 5.5 (about $0.30). Use `make eval-generation` for the Opus judge.
Responses are cached, so re-running costs nothing.

```bash
make report
```

Regenerates `results/REPORT.md`. Ablations write separate files, for example:

```bash
aiact eval retrieval --strategies structural --modes hybrid --recitals yes,no --tag recitals
```

With Docker:

```bash
docker compose run --rm eval eval retrieval
```

---

## Testing and CI

```bash
make test
```

```bash
make lint
```

38 tests run fully offline in about 5 seconds. They use deterministic hash-based encoders in place of the ONNX
models, a scripted LLM in place of Claude, and in-memory Qdrant, so they need no network, no model downloads and no
API key. Coverage includes:

- parser edge cases (enumerations, footnotes, exponents, Annex I)
- chunker invariants (every unit is retrievable, size limits, structural chunks never cross provisions, header and lead-in handling)
- metric correctness (nDCG ignores duplicate provisions, MRR, recall at the cut-off)
- gold-set integrity (every evidence span occurs in its gold provisions)
- the API end to end (auth, validation, error mapping, health probes)
- the evaluation runners and report generation

GitHub Actions runs lint, formatting checks, the test suite and the gold-set check on pushes to `main` and on pull
requests, then builds the Docker image.

---

## Project layout

```
src/aiact_rag/
  config.py            settings (env / .env)
  corpus/              EUR-Lex download and parser → data/corpus/aiact.jsonl
  chunking.py          five chunkers with provision attribution
  embeddings.py        dense, BM25 and cross-encoder wrappers (fastembed / ONNX)
  store.py             Qdrant collections, dense / sparse / hybrid-RRF search, recital filter
  retrieval.py         two-stage retrieval and reranking
  generation.py        prompt, structured output, citation extraction
  llm.py               Anthropic client, refusal fallback, response cache, token accounting
  pipeline.py          RAGService used by API, CLI and evaluation
  eval/                gold set, metrics, judge, runners, bootstrap statistics, report
  api/                 FastAPI app, schemas, static UI
  cli.py               `aiact` command
data/corpus/           parsed regulation (306 units)
data/eval/gold.jsonl   88 labelled questions
results/               result JSON for every run + REPORT.md
tests/                 offline test suite
Dockerfile             multi-stage, models baked in, non-root
docker-compose.yml     qdrant + indexer + api (+ eval profile)
```

---

## Limitations

- **One annotator.** The 84 answerable questions were written and labelled by one person; confidence intervals are
  reported for that reason. Recitals are never gold labels even when they discuss the topic, which slightly
  penalises configurations that surface them.
- **LLM-judged metrics.** Faithfulness, relevance and correctness come from an LLM judge. The prompts and schemas
  are in `eval/generation_metrics.py` and every verdict is saved. The published run used the cheapest judge, which
  sometimes counts statements like "the excerpts do not cover X" as unsupported claims, so faithfulness is, if
  anything, understated. Rerun with `JUDGE_MODEL=claude-opus-5-5` for a stricter judge.
- **Definitions are hard.** Most remaining false refusals ask for a definition. Article 3 holds 68 definitions in
  one paragraph, so its chunks look alike to the retriever.
- **Scope.** English text only (other EUR-Lex languages work by changing `CORPUS_URL`). Answers describe the text of
  the Regulation; they do not cover delegated acts, guidelines or case law, and they are not legal advice.

## Next steps

- Split Article 3 into one chunk per definition and re-measure the false-refusal rate.
- Rerun the answer-quality eval with Opus as judge to tighten the faithfulness estimate.
- Add a second annotator for a subset of the gold set and report inter-annotator agreement.
- Try a GPU-backed reranker, where the +1.8 recall@5 would come at an acceptable latency.

---

Source text: © European Union, https://eur-lex.europa.eu. Reuse authorised under Commission Decision 2011/833/EU.
