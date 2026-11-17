"""Compare dense retrieval with dense + LLM rerank on the same scratch collection.

    OPENAI_API_KEY=... uv run python app/eval/run_rerank_eval.py --qdrant-url http://localhost:6333

Same safety rules as run_eval.py (scratch collection, no `.env`, no Qdrant key). Query embeddings
are cached so both arms see identical candidates. The dense arm is checked against
results/dense_baseline.json first; a mismatch aborts. The rerank arm runs several times because
LLM output varies, and the spread is recorded, since a delta inside it is noise.
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from rag_app.parsing import split_pages
from rag_app.schema import AppSchemaExtension, DocumentContext
from rag_sdk.eval.harness import GoldenQuery, evaluate
from rag_sdk.providers.base import Chatter, Embedder, GeneratedResponse, Prompt
from rag_sdk.providers.llm_reranker import LLMReranker
from rag_sdk.providers.openai import OpenAIChatter, OpenAIEmbedder
from rag_sdk.providers.qdrant import QdrantIndexer
from rag_sdk.workflow.ingestion import IngestionWorkflow
from rag_sdk.workflow.reranking import RerankingRetriever
from run_eval import DIMENSIONS, load_golden

HERE = Path(__file__).parent
KS = (1, 3, 5, 10)
RERANKER_LOGGER = "rag_sdk.providers.llm_reranker"


class CachingEmbedder(Embedder):
    """Caches single-text embeddings and counts texts actually sent to the provider."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}
        self.texts_embedded = 0

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            self.texts_embedded += 1
            self._cache[text] = await self._inner.embed(text)
        return self._cache[text]

    async def embed_chunks(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]:
        self.texts_embedded += len(texts)
        return await self._inner.embed_chunks(texts, batch_size)


class MeteredChatter(Chatter):
    """Records chat calls, tokens and per-call latency."""

    def __init__(self, inner: Chatter) -> None:
        self._inner = inner
        self.latencies: list[float] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def generate(self, prompt: Prompt) -> GeneratedResponse:
        start = time.monotonic()
        response = await self._inner.generate(prompt)
        self.latencies.append(time.monotonic() - start)
        self.prompt_tokens += response.token_usage.prompt_tokens
        self.completion_tokens += response.token_usage.completion_tokens
        return response


class LevelCounter(logging.Handler):
    """Counts reranker log records of exactly one level.

    INFO = a well-formed but empty ranking (the model found nothing relevant) and WARNING = an
    unparseable reply. Both keep the retrieval order, so both score exactly like no rerank at all.
    """

    def __init__(self, level: int) -> None:
        super().__init__(level=level)
        self._exact = level
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno == self._exact:
            self.count += 1


def rank_key(rank: int | None) -> float:
    return float("inf") if rank is None else float(rank)


def movement(dense_rows: list[dict[str, Any]], rerank_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count queries whose first-relevant rank improved / stayed / worsened, overall and by kind."""
    out: dict[str, dict[str, int]] = {}
    for d, r in zip(dense_rows, rerank_rows, strict=True):
        assert d["query"] == r["query"]
        a, b = rank_key(d["first_relevant_rank"]), rank_key(r["first_relevant_rank"])
        verdict = "improved" if b < a else "worse" if b > a else "unchanged"
        for scope in ("overall", d["kind"]):
            out.setdefault(scope, {"improved": 0, "unchanged": 0, "worse": 0})[verdict] += 1
    return out


def spread(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(statistics.fmean(values), 4),
    }


async def main(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    pages = split_pages(args.corpus.read_text())
    if len(set(pages)) != len(pages):
        raise SystemExit("corpus has duplicate paragraphs")
    golden: list[GoldenQuery] = load_golden(args.golden, pages)
    baseline = json.loads(args.baseline.read_text())

    qdrant = AsyncQdrantClient(url=args.qdrant_url)
    openai_client = AsyncOpenAI(api_key=api_key)
    empties = LevelCounter(logging.INFO)
    unparseable = LevelCounter(logging.WARNING)
    reranker_logger = logging.getLogger(RERANKER_LOGGER)
    reranker_logger.setLevel(logging.INFO)  # INFO records are not emitted unless this is set
    reranker_logger.addHandler(empties)
    reranker_logger.addHandler(unparseable)
    try:
        if await qdrant.collection_exists(args.collection):
            raise SystemExit(f"collection {args.collection!r} already exists; refusing to reuse it")
        server_version = httpx.get(args.qdrant_url).json().get("version", "unknown")

        embedder = CachingEmbedder(
            OpenAIEmbedder(openai_client, model=args.embedding_model, dimensions=DIMENSIONS)
        )
        indexer = QdrantIndexer(qdrant, embedder, args.collection)
        workflow = IngestionWorkflow(embedder, indexer, args.collection, AppSchemaExtension())
        try:
            await workflow.ensure_index_exists()
            await workflow.ingest(DocumentContext(filename="eval_corpus.txt"), pages)
            wanted = sorted({i for g in golden for i in g.relevant})
            if set(wanted) - await indexer.existing_content_ids(wanted):
                raise SystemExit("golden labels missing from the collection")

            dense = indexer.create_searcher(args.collection)
            dense_report = await evaluate(dense, golden, ks=KS)
            dense_rows = dense_report.per_query()

            # The dense arm must reproduce the recorded baseline, or the comparison is invalid.
            recorded = {r["query"]: r["first_relevant_rank"] for r in baseline["per_query"]}
            drift = [
                r["query"]
                for r in dense_rows
                if recorded.get(r["query"]) != r["first_relevant_rank"]
            ]
            if drift:
                raise SystemExit(
                    f"dense arm differs from baseline on {len(drift)} queries: {drift}"
                )

            runs = []
            for run_no in range(1, args.runs + 1):
                chatter = MeteredChatter(
                    OpenAIChatter(
                        openai_client, model=args.chat_model, call_options={"temperature": 0}
                    )
                )
                empties.count = unparseable.count = 0
                reranked = RerankingRetriever(
                    dense, LLMReranker(chatter), retrieve_k=args.retrieve_k, top_n=max(KS)
                )
                report = await evaluate(reranked, golden, ks=KS)
                rows = report.per_query()
                lat = sorted(chatter.latencies)
                runs.append(
                    {
                        "run": run_no,
                        "summary": report.summary(),
                        "chat_calls": len(lat),
                        "prompt_tokens": chatter.prompt_tokens,
                        "completion_tokens": chatter.completion_tokens,
                        "latency_s": {
                            "mean": round(statistics.fmean(lat), 3),
                            "p95": round(lat[int(0.95 * (len(lat) - 1))], 3),
                        },
                        "empty_ranking_replies": empties.count,
                        "unparseable_replies": unparseable.count,
                        "movement_vs_dense": movement(dense_rows, rows),
                        "per_query": rows,
                    }
                )
        finally:
            await qdrant.delete_collection(args.collection)

        def metric_spread(metric: str, k: int) -> dict[str, float]:
            return spread([r["summary"]["overall"][metric][str(k)] for r in runs])

        dense_summary = dense_report.summary()
        record = {
            "date": date.today().isoformat(),
            "qdrant_server_version": server_version,
            "embedding_model": args.embedding_model,
            "chat_model": args.chat_model,
            "temperature": 0,
            "retrieve_k": args.retrieve_k,
            "final_k_values": list(KS),
            "rerank_runs": args.runs,
            "corpus_paragraphs": len(pages),
            "golden_queries": len(golden),
            "openai_embedding_inputs": embedder.texts_embedded,
            "openai_chat_calls": sum(r["chat_calls"] for r in runs),
            "dense_matches_recorded_baseline": True,
            "dense": dense_summary,
            "rerank_overall_spread": {
                m: {str(k): metric_spread(m, k) for k in KS} for m in ("recall", "mrr", "ndcg")
            },
            "dense_per_query": dense_rows,
            "runs": runs,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")

        print(f"dense vs rerank (retrieve_k={args.retrieve_k}, {args.runs} runs)")
        for m in ("recall", "mrr", "ndcg"):
            for k in KS:
                d = dense_summary["overall"][m][str(k)]
                print(f"  {m}@{k}: dense {d}  rerank {record['rerank_overall_spread'][m][str(k)]}")
        for r in runs:
            print(
                f"  run {r['run']}: moved {r['movement_vs_dense']['overall']}, "
                f"tokens {r['prompt_tokens']}+{r['completion_tokens']}, "
                f"latency {r['latency_s']}, "
                f"empty rankings {r['empty_ranking_replies']}, "
                f"unparseable {r['unparseable_replies']}"
            )
        return 0
    finally:
        reranker_logger.removeHandler(empties)
        reranker_logger.removeHandler(unparseable)
        await qdrant.close()
        await openai_client.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--qdrant-url", required=True, help="Qdrant URL (no API key is ever sent)")
    p.add_argument("--collection", default="eval_rerank_scratch")
    p.add_argument("--corpus", type=Path, default=HERE / "corpus.txt")
    p.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    p.add_argument("--baseline", type=Path, default=HERE / "results" / "dense_baseline.json")
    p.add_argument("--out", type=Path, default=HERE / "results" / "llm_rerank.json")
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    p.add_argument("--chat-model", default="gpt-4o-mini")
    p.add_argument("--retrieve-k", type=int, default=20)
    p.add_argument("--runs", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
