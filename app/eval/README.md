# Retrieval eval

A small, synthetic benchmark for comparing retrieval changes (reranking, hybrid search, chunking).

- `corpus.txt`: 90 synthetic paragraphs in 12 topics. Each topic has near-topic siblings; every
  identifier has two near-miss siblings (e.g. `E-4471` vs `E-4417`, `E-4174`) and every
  paraphrase target has one same-vocabulary distractor. Nothing project-specific. Chosen by that
  rule, not by looking at which queries missed.
- `golden.jsonl`: 27 queries (12 / 9 / 6) tagged `paraphrase`, `identifier` (rare tokens) or `multi` (several
  relevant paragraphs). Labels are anchor phrases resolved to content IDs by the runner, so they
  survive edits to the corpus as long as each anchor matches exactly one paragraph.
- `run_eval.py`: creates a **scratch** collection, indexes the corpus, scores the golden set, and
  deletes the collection. It never reads `.env`, never sends a Qdrant API key, and refuses to use
  a collection name that already exists.
- `results/dense_baseline.json`: the recorded dense-only baseline (see below).
- `run_hybrid_eval.py`: dense vs BM25 vs hybrid (client-side RRF, `HybridRetriever`) on a scratch
  OpenSearch index, vectors embedded once. Configuration (RRF constant 60, per-leg depth 20,
  default analyzer) is fixed in the code before the run. It aborts unless the dense arm
  reproduces `dense_baseline.json`. Result: `results/hybrid_vs_dense.json`.
- `run_backend_compare.py`: loads the same vectors (embedded once) into Qdrant and pgvector (exact
  scan, and HNSW with iterative scan) and scores all three. It aborts unless Qdrant and
  pgvector-exact reproduce `dense_baseline.json` per query. Both stores must be throwaway: it
  creates uniquely named scratch objects and drops them. Result: `results/backend_compare.json`.
- `run_rerank_eval.py`: runs dense and dense + `LLMReranker` on one scratch collection with cached
  query embeddings. It first checks the dense arm reproduces `dense_baseline.json` per query and
  aborts if not. The rerank arm runs several times (LLM output varies); the spread, token cost,
  latency, fallback count and per-query rank movement go to `results/llm_rerank.json`.

```
OPENAI_API_KEY=... uv run python app/eval/run_eval.py --qdrant-url http://localhost:6333
```

## Reading the baseline

`results/dense_baseline.json` holds the aggregates (overall and per kind), `per_query` rows with the
rank of the first relevant paragraph, and `saturated_metrics`: any metric at or above 0.97 cannot
show an improvement on this corpus. Compare a change against the record by diffing those fields,
not by eyeballing the aggregates; with 27 queries, one query moving is about 0.04.

Caveats: `recall@1` cannot reach 1.0 for `multi` queries (three relevant each). `--dimensions` is
not exposed because the embedder does not yet send `dimensions` to the API.
The corpus is still small, so treat it as a regression check, not a benchmark.

## Reading the rerank record

`results/llm_rerank.json` holds `dense`, `rerank_overall_spread` (min/max/mean across runs), and
per-run `movement_vs_dense` (queries improved / unchanged / worse by first-relevant rank), token
counts, latency and `unparseable_replies_kept_original_order` (a fallback scores exactly like
dense, so it must be zero to trust a run). A delta inside the run spread is noise.

Limits: this measures *reordering* only. `retrieve_k=20` is a fifth of this corpus, so every
relevant paragraph was already a candidate; it says nothing about candidate recall on a larger
corpus. Token and latency figures are for one-sentence paragraphs and scale with chunk size times
`retrieve_k`. After reranking almost no headroom is left here, so a later change (e.g. hybrid
search) can only be judged against the dense arm, or on a larger, harder corpus.

## Reading the backend comparison

`results/backend_compare.json` records store versions, per-arm metrics, per-query rows and
`queries_with_identical_top10_list` (how many queries returned the same top-10 as Qdrant). At 90
rows an HNSW index is effectively exact, so this shows the stores agree on identical vectors, not
how HNSW trades recall for speed at scale; that needs a much larger corpus.

## Reading the hybrid result

`results/hybrid_vs_dense.json` has per-arm and per-kind scores, `hybrid_vs_dense_movement`
(improved / unchanged / worse queries by kind) and every query where any metric fell. On this
corpus unweighted RRF is **worse overall than dense**: it fixes every identifier miss (BM25 alone
is perfect there: MRR 1.0 vs dense 0.80) but regresses 9 of 12 paraphrase queries.

Mechanism, from the record: with RRF constant 60 and depth 20, a paragraph returned by both legs
always outranks one returned by a single leg (2/80 beats 1/61). Each leg returns up to 22% of this
90-paragraph corpus, and the default analyzer keeps stop words ("the", "what", "my"), so BM25
matches many paragraphs on filler words; a lot of the overlap with the dense leg is chance. For 7
of the 10 queries that got worse, no relevant paragraph was in the lexical top 10 at all (lexical
paraphrase recall is 0.33 at 5 and 0.50 at 10), so the lexical leg only added noise. A small
corpus, a deep depth and equal weights all amplify this; it is not a general verdict on hybrid
search. The configuration was not tuned afterwards. Untried options, each a new recorded
configuration: weighted RRF, routing identifier-like queries to BM25, a stop-word analyzer, a
smaller depth.
