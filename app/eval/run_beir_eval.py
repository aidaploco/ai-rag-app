"""Retrieval evaluation on a public benchmark: BEIR SciFact (test split).

    OPENAI_API_KEY=... uv run --with pytrec-eval-terrier python app/eval/run_beir_eval.py \\
        --opensearch-url http://localhost:9200 [--rerank]

OpenSearch must be a THROWAWAY node: the runner creates a randomly named scratch index and drops it.
The dataset is downloaded at runtime into app/eval/.cache (git-ignored, never committed), and its
SHA-256 is checked. Check the dataset's own licence before redistributing anything derived from it.

Everything that affects the numbers is declared below, before any scored run. Arms:
  dense_exact       brute-force cosine over the cached vectors (the ground truth for "dense")
  dense_opensearch  OpenSearch HNSW kNN (approximate); top-10 overlap with dense_exact recorded
  bm25_opensearch   OpenSearch BM25, default analyzer
  hybrid_rrf        unweighted RRF of dense_opensearch + bm25_opensearch. Same fusion constant
                    (k=60) as the earlier synthetic eval, but depth 100 instead of 20 because
                    recall@100 is scored; scale and depth both differ from that eval
  dense_rerank_runN (--rerank, N runs) LLM listwise rerank of the top-20 dense_opensearch results

Scores come from rag_sdk.eval and are cross-checked with pytrec_eval (the TREC evaluator BEIR uses)
when it is installed. The BM25 arm is calibrated against the published figure.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import statistics
import sys
import urllib.request
import uuid
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch
from rag_app.schema import AppSchemaExtension, DocumentContext
from rag_sdk.eval.beir import BeirDataset, load_beir
from rag_sdk.eval.embedding_cache import CachedEmbedder, EmbeddingCache
from rag_sdk.eval.harness import GoldenQuery, evaluate
from rag_sdk.providers.base import Retriever, SearchResult, SearchResults
from rag_sdk.providers.llm_reranker import LLMReranker
from rag_sdk.providers.openai import OpenAIChatter, OpenAIEmbedder
from rag_sdk.providers.opensearch import OpenSearchIndexer
from rag_sdk.tokenizer import openai_token_counter
from rag_sdk.workflow.hybrid import HybridRetriever
from rag_sdk.workflow.ingestion import IngestionWorkflow
from rag_sdk.workflow.reranking import RerankingRetriever
from rag_sdk.workflow.schema_extension import content_id_for
from run_rerank_eval import RERANKER_LOGGER, LevelCounter, MeteredChatter

HERE = Path(__file__).parent
CACHE = HERE / ".cache"

# ---- declared before the first scored run: do not tune against results -------------------------
DATASET = "BEIR/SciFact"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
SPLIT = "test"
TEXT_FORMAT = '"<title>. <text>" per document, one chunk each (no chunking)'
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
KS = (10, 100)  # nDCG@10 is the headline; recall@100 is the standard BEIR second figure
RRF_K = 60
HYBRID_DEPTH = 100
RERANK_CANDIDATES = 20
RERANK_MODEL = "gpt-4o-mini"
LEXICAL = "OpenSearch BM25 (k1=1.2, b=0.75), default standard analyzer, single 'title. text' field"
PUBLISHED_BM25_NDCG10 = 0.665  # BEIR paper (arXiv 2104.08663) Table 2, SciFact, BM25
# -------------------------------------------------------------------------------------------------

SCIFACT_FILES = ("scifact/corpus.jsonl", "scifact/queries.jsonl", f"scifact/qrels/{SPLIT}.tsv")


def fetch_scifact(cache: Path) -> Path:
    """Download (once), verify the checksum, and extract only the three files we need."""
    target = cache / "scifact"
    if all((cache / name).exists() for name in SCIFACT_FILES):
        return target
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "scifact.zip"
    if not archive.exists():
        print(f"downloading {SCIFACT_URL}")
        urllib.request.urlretrieve(SCIFACT_URL, archive)  # noqa: S310 (fixed https URL)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SCIFACT_SHA256:
        archive.unlink()
        raise SystemExit(f"checksum mismatch for scifact.zip: {digest}")
    with zipfile.ZipFile(archive) as zf:
        for name in SCIFACT_FILES:  # fixed names only: nothing from the archive picks a path
            (cache / name).parent.mkdir(parents=True, exist_ok=True)
            (cache / name).write_bytes(zf.read(name))
    return target


class ExactRetriever(Retriever):
    """Brute-force cosine similarity over a matrix of vectors: the exact dense baseline."""

    def __init__(self, embedder: CachedEmbedder, matrix: np.ndarray, ids: list[str]) -> None:
        self._embedder = embedder
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._matrix = matrix / norms
        self._ids = ids

    async def retrieve(
        self, query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> SearchResults:
        q = np.asarray(await self._embedder.embed(query), dtype=np.float32)
        scores = self._matrix @ (q / np.linalg.norm(q))
        k = top_k or 10
        top = np.argsort(-scores, kind="stable")[:k]
        return [SearchResult("", float(scores[i]), {}, content_id=self._ids[i]) for i in top]


def per_query_rows(report: Any, qids: list[str]) -> dict[str, Any]:
    """Per-query rows keyed by query ID, so the committed record does not carry the dataset."""
    rows = {}
    for qid, row in zip(qids, report.per_query(), strict=True):
        rows[qid] = {k: v for k, v in row.items() if k not in ("query", "kind")}
    return rows


def pytrec_crosscheck(
    data: BeirDataset, runs: dict[str, dict[str, list[str]]], docid_by_cid: dict[str, str]
) -> dict[str, Any] | None:
    """Score each arm's rankings with pytrec_eval, by BEIR document ID, for comparison."""
    try:
        import pytrec_eval
    except ImportError:
        return None
    evaluator = pytrec_eval.RelevanceEvaluator(
        data.qrels, {"ndcg_cut_10", "recall_10", "recall_100"}
    )
    out = {}
    for arm, run in runs.items():
        trec_run = {
            qid: {docid_by_cid[c]: float(len(ranked) - i) for i, c in enumerate(ranked)}
            for qid, ranked in run.items()
        }
        scores = evaluator.evaluate(trec_run)
        out[arm] = {
            m: round(statistics.fmean(s[m] for s in scores.values()), 4)
            for m in ("ndcg_cut_10", "recall_10", "recall_100")
        }
    return out


def estimate_rerank_tokens(queries: list[str], candidates: list[list[str]], count) -> int:
    """Prompt tokens the reranker will send, built the same way LLMReranker builds its prompt."""
    total = 0
    for query, passages in zip(queries, candidates, strict=True):
        body = "\n\n".join(f"[{i}] {p[:1000]}" for i, p in enumerate(passages, start=1))
        total += count(f"Query: {query}\n\nPassages:\n{body}") + 80  # ~system prompt
    return total


async def main(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    data = load_beir(fetch_scifact(args.cache), SPLIT)
    docids = list(data.corpus)
    texts = [data.corpus[d] for d in docids]
    cids = [content_id_for(t) for t in texts]
    if len(set(cids)) != len(cids):
        raise SystemExit("documents with identical text collapse to one ID; handle before running")
    docid_by_cid = dict(zip(cids, docids, strict=True))
    cid_by_docid = dict(zip(docids, cids, strict=True))
    qids = list(data.queries)
    golden = [
        GoldenQuery(
            data.queries[q], frozenset(cid_by_docid[d] for d in data.qrels[q]), kind="scifact"
        )
        for q in qids
    ]

    count = openai_token_counter(EMBEDDING_MODEL)
    openai_client = AsyncOpenAI(api_key=api_key)
    cache = EmbeddingCache(args.cache / "vectors.sqlite", EMBEDDING_MODEL, DIMENSIONS)
    embedder = CachedEmbedder(
        OpenAIEmbedder(openai_client, model=EMBEDDING_MODEL, dimensions=DIMENSIONS), cache, count
    )
    index = f"eval_beir_{uuid.uuid4().hex[:10]}"
    client = AsyncOpenSearch(hosts=[args.opensearch_url], timeout=180)
    try:
        version = httpx.get(args.opensearch_url).json()["version"]["number"]
        vectors = np.asarray(await embedder.embed_chunks(texts), dtype=np.float32)
        await embedder.embed_chunks(list(data.queries.values()))
        embedded = (embedder.texts_embedded, embedder.tokens_embedded, embedder.requests)
        print(
            f"embeddings sent this run: {embedded[0]} texts, {embedded[1]:,} tokens, "
            f"{embedded[2]} requests"
        )

        indexer = OpenSearchIndexer(client, embedder, index)
        workflow = IngestionWorkflow(embedder, indexer, index, AppSchemaExtension())
        try:
            await workflow.ensure_index_exists()
            await workflow.ingest(DocumentContext(filename=DATASET), texts)
            await client.indices.refresh(index=index)

            exact = ExactRetriever(embedder, vectors, cids)
            dense = indexer.create_searcher(index)
            bm25 = indexer.create_lexical_searcher(index)
            hybrid = HybridRetriever([dense, bm25], rrf_k=RRF_K, depth=HYBRID_DEPTH, top_n=max(KS))
            arms: dict[str, Retriever] = {
                "dense_exact": exact,
                "dense_opensearch": dense,
                "bm25_opensearch": bm25,
                "hybrid_rrf": hybrid,
            }
            reports = {name: await evaluate(r, golden, ks=KS) for name, r in arms.items()}

            overlap = statistics.fmean(
                len(set(a.retrieved[:10]) & set(b.retrieved[:10])) / 10
                for a, b in zip(
                    reports["dense_exact"].queries, reports["dense_opensearch"].queries, strict=True
                )
            )

            rerank_info: dict[str, Any] | None = None
            if args.rerank:
                top = [
                    [data.corpus[docid_by_cid[c]] for c in q.retrieved[:RERANK_CANDIDATES]]
                    for q in reports["dense_opensearch"].queries
                ]
                est_tokens = estimate_rerank_tokens([g.query for g in golden], top, count)
                est_usd = est_tokens / 1e6 * args.price_per_mtok_input
                print(
                    f"rerank estimate: {est_tokens:,} prompt tokens ~ ${est_usd:.2f} "
                    f"(assumed ${args.price_per_mtok_input}/M input tokens; verify)"
                )
                if est_usd * args.rerank_runs > args.max_rerank_usd:
                    raise SystemExit(
                        f"estimate x {args.rerank_runs} runs exceeds --max-rerank-usd "
                        f"{args.max_rerank_usd}"
                    )
                reranker_logger = logging.getLogger(RERANKER_LOGGER)
                reranker_logger.setLevel(logging.INFO)
                run_info = []
                for run_no in range(1, args.rerank_runs + 1):
                    chatter = MeteredChatter(
                        OpenAIChatter(
                            openai_client, model=RERANK_MODEL, call_options={"temperature": 0}
                        )
                    )
                    empties = LevelCounter(logging.INFO)
                    unparseable = LevelCounter(logging.WARNING)
                    reranker_logger.addHandler(empties)
                    reranker_logger.addHandler(unparseable)
                    reranked = RerankingRetriever(
                        dense, LLMReranker(chatter), retrieve_k=RERANK_CANDIDATES, top_n=KS[0]
                    )
                    reports[f"dense_rerank_run{run_no}"] = await evaluate(
                        reranked, golden, ks=(KS[0],)
                    )
                    reranker_logger.removeHandler(empties)
                    reranker_logger.removeHandler(unparseable)
                    run_info.append(
                        {
                            "run": run_no,
                            "chat_calls": len(chatter.latencies),
                            "prompt_tokens": chatter.prompt_tokens,
                            "completion_tokens": chatter.completion_tokens,
                            "latency_s_mean": round(statistics.fmean(chatter.latencies), 3),
                            # Well-formed but empty ranking: the model found nothing relevant, and
                            # the retrieval order was kept (scores exactly like dense).
                            "empty_ranking_replies": empties.count,
                            "unparseable_replies": unparseable.count,
                        }
                    )
                rerank_info = {
                    "model": RERANK_MODEL,
                    "temperature": 0,
                    "candidates": RERANK_CANDIDATES,
                    "estimated_prompt_tokens_per_run": est_tokens,
                    "runs": run_info,
                }
        finally:
            await client.indices.delete(index=index, ignore_unavailable=True)

        runs = {
            name: {qid: q.retrieved for qid, q in zip(qids, rep.queries, strict=True)}
            for name, rep in reports.items()
        }
        crosscheck = pytrec_crosscheck(data, runs, docid_by_cid)
        arms_out = {name: rep.summary()["overall"] for name, rep in reports.items()}
        bm25_ndcg = arms_out["bm25_opensearch"]["ndcg"]["10"]
        record = {
            "date": date.today().isoformat(),
            "dataset": {
                "name": DATASET,
                "split": SPLIT,
                "zip_sha256": SCIFACT_SHA256,
                "documents": len(docids),
                "queries": len(qids),
                "judged_pairs": sum(len(v) for v in data.qrels.values()),
                "documents_sharing_identical_text": 0,
                "note": "downloaded at runtime, not committed; check the dataset licence",
            },
            "declared_config": {
                "text_format": TEXT_FORMAT,
                "embedding_model": EMBEDDING_MODEL,
                "dimensions": DIMENSIONS,
                "ks": list(KS),
                "rrf_k": RRF_K,
                "hybrid_depth": HYBRID_DEPTH,
                "lexical": LEXICAL,
            },
            "opensearch_version": version,
            "embeddings_sent_this_run": {
                "texts": embedded[0],
                "tokens": embedded[1],
                "requests": embedded[2],
            },
            "calibration": {
                "bm25_published_ndcg10": PUBLISHED_BM25_NDCG10,
                "source": "BEIR paper (arXiv 2104.08663) Table 2, BM25 on SciFact",
                "bm25_measured_ndcg10": bm25_ndcg,
                "difference": round(bm25_ndcg - PUBLISHED_BM25_NDCG10, 4),
            },
            "dense_opensearch_vs_exact_mean_top10_overlap": round(overlap, 4),
            "arms": arms_out,
            "rerank": rerank_info,
            "crosscheck_pytrec_eval": crosscheck,
            "per_query": {name: per_query_rows(rep, qids) for name, rep in reports.items()},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")

        print(f"\n{DATASET} {SPLIT}: {len(qids)} queries, {len(docids)} documents")
        print(f"{'arm':20} {'nDCG@10':>8} {'R@10':>7} {'R@100':>7} {'MRR@10':>7}")
        for name, o in arms_out.items():
            r100 = o["recall"].get("100", "-")
            print(
                f"{name:20} {o['ndcg']['10']:>8} {o['recall']['10']:>7} "
                f"{r100!s:>7} {o['mrr']['10']:>7}"
            )
        print(
            f"BM25 calibration: measured {bm25_ndcg} vs published {PUBLISHED_BM25_NDCG10} "
            f"(difference {bm25_ndcg - PUBLISHED_BM25_NDCG10:+.3f})"
        )
        print(f"dense OpenSearch vs exact, mean top-10 overlap: {overlap:.3f}")
        print("pytrec_eval cross-check:", json.dumps(crosscheck) if crosscheck else "not installed")
        return 0
    finally:
        cache.close()
        await client.close()
        await openai_client.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--opensearch-url", required=True, help="THROWAWAY OpenSearch node")
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=HERE / "results" / "beir_scifact.json")
    p.add_argument(
        "--rerank", action="store_true", help="also run the LLM rerank arm (costs money)"
    )
    p.add_argument("--rerank-runs", type=int, default=1, help="repeat the rerank arm N times")
    p.add_argument("--max-rerank-usd", type=float, default=1.0)
    p.add_argument(
        "--price-per-mtok-input",
        type=float,
        default=0.15,
        help="ASSUMED gpt-4o-mini input $/M tokens; verify before relying on it",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
