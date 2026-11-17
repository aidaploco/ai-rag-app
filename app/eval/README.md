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
- `run_chunking_eval.py`: compares chunking strategies on long, blank-line-free documents derived
  from the corpus (18 paragraphs joined by spaces, about 450 tokens each), scored by anchor
  containment at equal token budgets (`rag_sdk.eval.anchors`). In-memory Qdrant; arms, budgets and
  depth are declared in code before the run. Result: `results/chunking.json`.
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
counts, latency and the fallback counts (`empty_ranking_replies` and `unparseable_replies` in runs made
after the logging fix; the committed `llm_rerank.json` predates it and has a single
`unparseable_replies_kept_original_order` field, which was 0). A fallback keeps the retrieval order and
scores exactly like dense, so check these before trusting a run. A delta inside the run spread is
noise.

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

## Reading the chunking result

Labels here are *anchors* (phrases that answer a query), not content IDs, because chunking changes
chunk text and so changes IDs. A chunk finds an anchor if it contains it. Arms are compared at
equal token budgets (256, 512, 1024 tokens of retrieved context): only whole chunks that fit count,
so large chunks cannot win by returning a whole document.

What the record shows (5 documents, 27 queries, 36 distinct anchors, one run):

- **Any chunking beats the current blank-line split on this kind of text.** These documents have
  no blank lines, so the default makes one ~450-token chunk per document. At a 512-token budget
  `recursive_64` found the answer on 14 more queries than the default and was worse on none. (At
  256 tokens the gap is 25 queries, but that one is true by construction: the default's ~450-token
  chunk cannot fit a 256-token budget, so it scores 0.)
- **Recursive vs naive fixed windows is a small difference.** `recursive_64` was better on 5
  queries and worse on 1 at 256 tokens (2 and 1 at 512). Fixed windows cut 1 of 36 anchors in half
  at a boundary; the recursive chunker cut none. With 27 queries that is a hint, not a result.
- **Overlap showed no demonstrated benefit** (better on 2, worse on 1 at 256; 1 and 1 at 512).
  Overlap is whole sentences only, so on this corpus it needs a budget of at least one paragraph.
- **Smaller chunks (64 tokens) did better than 128** at the small budgets, as expected: more
  chunks fit.
- **Do not compare MRR across chunk sizes.** The recorded `mrr` is the rank of the first chunk
  containing an anchor and ignores size, so the 5 huge chunks of the default score about as well as
  anything. The budgeted recall is the comparison to use.

Limits: the documents are synthetic and built by concatenation; anchors are short phrases, so
"contains the anchor" is weaker than "contains the full answer"; the corpus is tiny. The app
default is unchanged (`CHUNKING=paragraph`); `recursive` is opt-in.

## Public benchmark: BEIR SciFact

`run_beir_eval.py` evaluates on a public dataset, **BEIR SciFact (test split)**: 5,183
scientific abstracts, 300 claim queries, 339 relevance judgements. The data is
downloaded at runtime into `app/eval/.cache` (git-ignored, SHA-256 checked) and is **never committed**;
check the dataset's own licence before redistributing anything derived from it. Cite: Thakur et al.,
*BEIR* (NeurIPS Datasets and Benchmarks 2021, arXiv:2104.08663), and Wadden et al., *Fact or Fiction: Verifying Scientific Claims* (EMNLP 2020), which introduced SciFact.

Every setting is declared in the runner before the first scored run: one chunk per document
(`"<title>. <text>"`), `text-embedding-3-small` at 1536 dimensions, unweighted RRF with k=60 and depth 100,
the default OpenSearch analyzer. Results are in `results/beir_scifact.json`, per-query rows keyed by query ID.

| Arm | nDCG@10 | recall@10 | recall@100 | MRR@10 |
|---|---|---|---|---|
| Dense, exact (brute-force cosine) | 0.731 | 0.8576 | 0.97 | 0.6979 |
| Dense, OpenSearch HNSW | 0.731 | 0.8576 | 0.97 | 0.6979 |
| BM25, OpenSearch | 0.6606 | 0.7843 | 0.8826 | 0.6283 |
| Hybrid: BM25 + dense, unweighted RRF | 0.7186 | 0.8346 | 0.98 | 0.6887 |
| Dense top-20 + LLM rerank, run 1 | 0.7697 | 0.8763 | — | 0.7389 |
| Dense top-20 + LLM rerank, run 2 | 0.7723 | 0.8763 | — | 0.7433 |

(The rerank arm returns 10 results, so it has no recall@100.)

**How far to trust it**

- *Calibration.* The BM25 arm scored 0.6606 against the published 0.665 (BEIR paper, Table 2), a difference of
  -0.004. All arms were also scored with `pytrec_eval`, the TREC evaluator BEIR uses, and match to four
  decimals (`crosscheck_pytrec_eval` in the record).
- *Approximate vs exact.* OpenSearch's HNSW dense arm matches brute-force dense to four decimals
  (mean top-10 overlap 0.995). Re-building the index changes a handful of per-query rows (4 to 6 of 300
  between builds) and not the aggregates; BM25 and exact dense reproduce exactly.
- *Is a gap real?* With 300 queries, a point or two is often noise, so `compare_arms.py` runs a paired bootstrap on
  per-query nDCG@10 (`results/beir_scifact_comparisons.json`):
  - BM25 minus dense: -0.070 (95% CI -0.111 to -0.030; better on 56, worse on 90, tied on 154). Dense is better, significantly.
  - Hybrid minus dense: -0.012 (95% CI -0.040 to +0.013; better on 52, worse on 49, tied on 199). **Not distinguishable from dense**; on this
    dataset unweighted fusion neither helped nor measurably hurt (it raised recall@100 from 0.97 to 0.98).
  - Rerank minus dense: run 1 +0.039 (95% CI +0.016 to +0.061; better on 55, worse on 23, tied on 222); run 2 +0.041 (95% CI +0.019 to +0.064; better on 55, worse on 24, tied on 221). A real
    improvement. An earlier run of the same configuration, made before the logging fix below, scored 0.7674.

**Costs and caveats**

- One-time embedding pass: 5,483 texts, 1,665,861 tokens, 23 sequential requests (about 3 cents at an
  assumed price of $0.02 per million tokens; verify). The vectors are cached on disk, so later runs
  send none.
- Rerank: 300 calls and 1,280,797 prompt tokens per run with `gpt-4o-mini` at temperature 0 (about 20 cents per run at an
  assumed price), 0.991 s mean latency per call.
- `empty_ranking_replies` counts replies containing a well-formed array with no usable passage number
  (39 and 38 of 300 queries in the two runs; the model found nothing that supports the claim). The
  retrieval order was kept for those, so they score exactly like dense. `unparseable_replies` (0 in
  both runs) counts replies with no array at all. An earlier run of the same configuration counted
  both as "unparseable"; a diagnostic run on 60 queries found 7 such replies, and the 6 that were
  printed were all a literal `[]`, which is why the reranker now logs the two cases separately.
- In the record's `crosscheck_pytrec_eval`, the rerank arms' `recall_100` equals `recall_10` because those
  arms return only 10 results; it is not comparable with the other arms' recall@100.
- Reranking only reorders the top 20, so it cannot improve recall beyond what dense retrieved there.
- SciFact queries are claims, not questions, and abstracts are short. Results on long documents, other query
  styles or a larger corpus may differ. Nothing here was tuned against the test split.
