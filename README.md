# ai-rag-app

[![CI](https://github.com/aidaploco/ai-rag-app/actions/workflows/ci.yml/badge.svg)](https://github.com/aidaploco/ai-rag-app/actions/workflows/ci.yml)

A retrieval-augmented generation (RAG) **SDK** and a small **FastAPI service** built on it. The SDK puts embeddings, vector stores, rerankers and hybrid search behind small interfaces, so you can swap a store or add a retrieval stage without touching the rest, and it ships the tools to **measure** what each change does: recall, MRR and nDCG, paired significance tests, and a runner for a public benchmark (BEIR SciFact). The service ingests text, Markdown, HTML, PDF and DOCX files and answers questions over them.

> **This is an engineering showcase, not a hardened product.** The service has no authentication, the demo corpus in `app/eval/` is synthetic, and several SDK components (OpenSearch, FAISS, hybrid search, reranking) are not wired into the service. See [Known limitations](#known-limitations).

> **Where to read next:** [How it works](#how-it-works), then [What it measured](#what-it-measured) and the full write-up with caveats in [`app/eval/README.md`](app/eval/README.md).

## The one big idea

**Swap the part, then measure the change.** Every retrieval component sits behind an interface (`Embedder`, `Chatter`, `Retriever`, `Indexer`, `Reranker`), and the results quoted below come from recorded runs in [`app/eval/results/`](app/eval/results/). Negative and inconclusive results are recorded too: hybrid search lost on the small synthetic set and was statistically indistinguishable from dense retrieval on SciFact, and that is what the repo says.

## How it works

```mermaid
flowchart LR
    UP["POST /ingest<br/>txt, md, html, pdf, docx"] --> PA["parse to text blocks<br/>(PDF: one per page)"]
    PA --> CK["split into chunks<br/>blank-line split (default)<br/>or token-capped chunker"]
    CK --> DD["dedup by content hash"]
    DD --> EM["embed (OpenAI)"]
    EM --> ST[("vector store<br/>Qdrant or pgvector")]
    Q["POST /generate"] --> RT["retrieve top-k<br/>dense, optional filters"]
    ST --> RT
    RT --> GEN["chat model answers<br/>from the retrieved context"]
    GEN --> OUT["answer, context, token usage"]
```

- **The provider seam.** The SDK defines the interfaces; concrete code lives in `providers/`: OpenAI for embeddings and chat, and Qdrant, pgvector, OpenSearch and FAISS as vector stores. One contract suite runs against all four stores (Qdrant's in-memory client, FAISS in-process, and real Postgres and OpenSearch service containers in CI), including that every score equals an independently computed cosine. Import-linter enforces that vendor SDKs stay inside `providers/` and `tokenizer.py`.
- **Retrieval stages compose.** `HybridRetriever` (reciprocal rank fusion over BM25 and dense) and `RerankingRetriever` (over-fetch, then an LLM reorders) are themselves retrievers, so `RagWorkflow` and the eval harness take them unchanged.
- **Content-addressed ingestion.** A chunk's ID is a hash of its text, so re-uploading a file is a no-op (a second upload reports every paragraph skipped). Chunks carry `chunk_idx`, and PDF chunks carry their real `page_nr`.
- **A chunker with a hard token cap.** The optional `CHUNKING=recursive` mode splits by paragraph, then sentence, then token window, counting tokens with the embedding model's own tokenizer. Every chunk is an exact substring of the input and within the cap; the cap was fuzz-tested against the real tokenizer, which found and fixed a case where a trimmed piece came out one token over the cap on multi-byte text before it shipped.
- **Resilience and tracing.** The OpenAI embedding and chat calls retry on rate limits and honour `Retry-After`, and the main paths emit OpenTelemetry spans. The SDK also has throttling primitives, but no provider uses them yet.
- **Evaluation is part of the SDK.** `rag_sdk.eval` has recall@k, MRR@k and nDCG@k, anchor-containment scoring for comparing chunkers at equal token budgets, a paired bootstrap, a BEIR loader and a disk embedding cache.

What the **service** wires up and what is **SDK-only**:

| | In the service | SDK only |
|---|---|---|
| Vector store | Qdrant (default), pgvector (`VECTOR_STORE=pgvector`) | OpenSearch (dense and BM25), FAISS |
| Retrieval | dense, with metadata filters | hybrid (BM25 + dense, RRF), LLM reranking |
| Chunking | blank-line split (default), token-capped chunker (opt-in) | |

## What it measured

All of this is from recorded runs. The SciFact settings were declared in code before its first scored run; two adjustments made to the synthetic evals along the way are listed in [`app/eval/README.md`](app/eval/README.md). The small synthetic corpus (90 paragraphs, 27 queries) is for regression checks; the SciFact numbers are the ones to quote.

| Question | Result |
|---|---|
| Is the benchmark pipeline right? | On BEIR SciFact (5,183 abstracts, 300 test queries) BM25 scores **0.661** nDCG@10 against the published 0.665, and `pytrec_eval`, the TREC evaluator BEIR uses, agrees with the harness to four decimals. |
| Dense retrieval (`text-embedding-3-small`) | **0.731** nDCG@10, recall@100 0.970. |
| Does LLM reranking help? | Reranking the dense top 20 with `gpt-4o-mini` gives about **0.77** (three runs of the same configuration scored 0.767, 0.770 and 0.772), a gain of about **+0.04** whose 95% interval excludes zero. A 300-query run sends about 1.3M prompt tokens (roughly 4,300 per query) and takes about a second per query. |
| Does hybrid search help? | **Not distinguishable from dense** on SciFact: 0.719 against 0.731, interval -0.040 to +0.014 (recall@100 rose from 0.970 to 0.980). On the 27-query synthetic set it lost overall: it fixed all three identifier-style misses and hurt 9 of 12 paraphrase queries. One configuration (RRF constant 60; depth 20 on the synthetic set, 100 on SciFact), not tuned afterwards. |
| Does the chunker help? | On synthetic documents of about 450 tokens with no blank lines, scored by whether the retrieved chunks contain the answer phrase, 64-token chunks found the answer on **14 more of 27 queries** than the default split at a 512-token retrieval budget and were worse on none. Against naive fixed windows, and for overlap, the differences were within noise. The 256/32 defaults used in the Quickstart were not evaluated. |
| Do the stores agree? | Qdrant, pgvector (exact and HNSW) returned identical top-10 lists on 27 of 27 queries over identical vectors. |

The full tables, costs and caveats, and how to reproduce each, are in [`app/eval/README.md`](app/eval/README.md).

## Quickstart

Requires Docker, and an OpenAI API key (embeddings and chat). For development, Python 3.11+ with [`uv`](https://docs.astral.sh/uv/).

```bash
echo "OPENAI_API_KEY=sk-..." >> .env
docker compose up --build        # Qdrant on :6333, the API on :8000 (docs at /docs)
```

```bash
# Upload a file (txt, md, html, pdf or docx). A second upload of the same file skips every paragraph.
curl -s -F "file=@app/eval/corpus.txt" localhost:8000/ingest
# {"pages_ingested":90,"pages_skipped":0}

curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"query":"How often should I descale my espresso machine?"}'
```

The answer for that query was `"You should descale your espresso machine every three months."` with five retrieved chunks, the top one scored 0.605 and starting *"Descale the boiler every three months using a citric-acid solution"*, and 193 tokens used. Add `"filters": {"filename": "corpus.txt"}` to restrict a query to one file.

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | Upload a file (`file`, optional `filename`); returns how many paragraphs were ingested and skipped. 415 for an unsupported or non-UTF-8 type, 413 over 10 MiB, 422 for a corrupt or text-free file |
| `POST /generate` | `{"query": ..., "filters": {...}}`; returns the answer, the retrieved context and token usage |
| `POST /` | Upload a file and ask a question in one call (`file`, `query`). The question searches the whole collection, not only the uploaded file |

**Use Postgres instead of Qdrant, and the token-capped chunker:**

```bash
cat >> .env <<'EOF'
VECTOR_STORE=pgvector
POSTGRES_DSN=postgresql://postgres:postgres@postgres:5432/postgres
CHUNKING=recursive
CHUNK_MAX_TOKENS=256
CHUNK_OVERLAP_TOKENS=32
EOF
docker compose --profile pgvector up --build
```

Changing `CHUNKING` on an existing collection gives new chunk IDs and leaves the old chunks in place, so use a new collection or re-ingest. Switching the store keeps the same IDs and simply starts empty.

## Tests, CI and evals

```bash
uv sync
uv run pytest rag-sdk/tests && uv run pytest app/tests    # some cases skip without the services below
uv run ruff format --check rag-sdk app && uv run ruff check rag-sdk app
uv run mypy rag-sdk/rag_sdk                                # types (SDK)
uv run lint-imports                                        # the architecture contracts
```

The Postgres and OpenSearch tests need a throwaway instance: set `PGVECTOR_TEST_DSN` and `OPENSEARCH_TEST_URL` (for example `postgresql://postgres:pw@localhost:5432/postgres` and `http://localhost:9200`). Without them those cases skip locally and **fail in CI**, so they cannot silently stop running. Point them only at disposable databases.

CI ([`ci.yml`](.github/workflows/ci.yml)) runs lint, types and import-linter; the full suite on Python 3.11 (the declared floor) and 3.13 (what the image runs) against real Postgres and OpenSearch containers, with coverage floors of 95%; a gitleaks scan of the full history; and a Docker build whose image is smoke-tested with no network (the tokenizer vocabulary is baked in at build time). One fail-closed job, `ci-success`, is the single check to require on `main`. Actions are pinned to commit SHAs, and a test fails if a pin, the Python matrix or the gate drifts. Dependabot opens weekly grouped updates.

The retrieval evals make real OpenAI calls and cost (small) money, so they are manual. Each runner is documented at the top of its file and in [`app/eval/README.md`](app/eval/README.md): for example `app/eval/run_beir_eval.py` for SciFact, and a manual workflow, [`evals.yml`](.github/workflows/evals.yml), re-runs the synthetic baseline and fails if a metric regresses (it needs an `OPENAI_API_KEY` repository secret).

## Configuration

Settings load from environment variables or `.env` ([`config.py`](app/rag_app/config.py) is authoritative; [`.env.example`](.env.example) lists them).

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | required | Embeddings and chat |
| `EMBEDDING_MODEL` / `CHAT_MODEL` | `text-embedding-3-small` / `gpt-4o-mini` | Models |
| `EMBEDDING_DIMENSIONS` | `1536` | Vector size of the collection. Leave it at the model's native size; see the limitation below |
| `VECTOR_STORE` | `qdrant` | `qdrant` or `pgvector` |
| `QDRANT_URL` / `QDRANT_API_KEY` / `QDRANT_COLLECTION` | `http://localhost:6333` / unset / `documents` | Qdrant (a blank API key counts as unset) |
| `POSTGRES_DSN` / `PGVECTOR_TABLE` | required for pgvector / `documents` | Postgres connection and table |
| `CHUNKING` | `paragraph` | `paragraph` (blank-line split) or `recursive` (token-capped) |
| `CHUNK_MAX_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `256` / `32` | Chunk cap and overlap (whole sentences; the overlap must be smaller than the cap) |

## Repository layout

```
rag-sdk/rag_sdk/
  providers/    the interfaces, plus OpenAI, Qdrant, pgvector, OpenSearch, FAISS and the LLM reranker
  workflow/     ingestion, RAG, reranking and hybrid (RRF) retrieval
  chunking.py   the token-capped, sentence-aware chunker (pure; takes a token counter)
  tokenizer.py  exact OpenAI token counts (tiktoken)
  eval/         metrics, the harness, anchor scoring, paired bootstrap, BEIR loader, embedding cache
  core/         retries, throttling, tracing
rag-sdk/tests/  unit tests and the shared vector-store contract suite
app/rag_app/    the FastAPI service: routes, parsing (txt/md/html/pdf/docx), ingestion and RAG services
app/tests/      service tests, plus tests that keep CI, the Docker image and the gate aligned
app/eval/       eval runners, the synthetic corpus and golden set, recorded results, a regression check
.github/        CI, a manual eval workflow, Dependabot
Dockerfile, docker-compose.yml
```

## Known limitations

- **No authentication.** Every endpoint is open. Do not expose the service as it is.
- **The service uses only part of the SDK.** OpenSearch, FAISS, hybrid search and reranking are tested and measured but not wired into the service.
- **`EMBEDDING_DIMENSIONS` is not sent to the embeddings API.** Only the model's native size works; any other value mismatches the collection. (A fix is planned.)
- **The OpenAI embedder batches by input count, not by tokens.** Providers also cap tokens per request, so a very large single upload can be rejected. The eval tooling has a token-aware batcher that the ingest path does not use yet.
- **Deduplication is by content only.** A paragraph already stored from another file is skipped and keeps the first file's `filename`, so filename filters miss it. After a document is edited, existing chunks keep their first-ingest `chunk_idx`.
- **`POST /` searches the whole collection,** not just the file uploaded with the question, so its answer can come from other documents.
- **Parsing has limits.** No OCR (scanned PDFs return 422); Markdown is read as plain text, not parsed; DOCX body paragraphs only (tables, headers and footers are skipped); text and HTML must be UTF-8; uploads are capped at 10 MiB. A text block with no blank lines is one chunk unless `CHUNKING=recursive` is on.
- **The evaluations are small.** The synthetic set is 90 paragraphs and 27 queries; SciFact is short abstracts indexed one per document. Results on long documents or other query styles may differ, and hybrid search in particular was one untuned configuration. FAISS here is an exact in-process index, not approximate search at scale, and OpenSearch rejects a metadata key whose type changes between documents.
- **OpenAI only** for embeddings and chat. The Docker image runs as root, and compose uses `qdrant/qdrant:latest`. Not load-tested.

## Development conventions

- **Python 3.11** is the declared floor, tested alongside 3.13 (what the image runs). Managed with `uv`; formatting and linting with `ruff`; types with `mypy` (SDK); architecture with `import-linter`.
- **Measure before claiming.** Declare an evaluation's settings before running it, record the result whichever way it goes, and compare arms with a paired test when the query count is small.
- **A new vector store must pass the shared contract suite**, including the cosine-score check, before it counts as a provider.
- **Fail loudly.** A missing key, a bad table name, a wrong vector size or an unreadable file is an error, not a silent fallback.
- Changes land through pull requests that pass `ci-success`.

## License

No license is granted. The source is published so it can be read and reviewed; please do not copy, redistribute, or reuse it without permission. All rights reserved.
