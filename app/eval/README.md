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
