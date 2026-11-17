"""Compare vector-store backends on identical vectors: Qdrant vs pgvector (exact and HNSW).

    OPENAI_API_KEY=... uv run python app/eval/run_backend_compare.py \\
        --qdrant-url http://localhost:6333 --postgres-dsn postgresql://postgres:pw@localhost:5432/postgres

Both stores must be THROWAWAY instances. The runner creates a randomly named scratch collection
and scratch tables and drops them afterwards. Texts are
embedded once and the same vectors are loaded into every backend, so any difference comes from
the store. The Qdrant and pgvector-exact arms must reproduce results/dense_baseline.json per
query, otherwise the run aborts.
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import asyncpg
import httpx
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from rag_app.parsing import split_pages
from rag_app.schema import AppSchemaExtension, DocumentContext
from rag_sdk.eval.harness import evaluate
from rag_sdk.providers.base import Embedder, Indexer, Retriever
from rag_sdk.providers.openai import OpenAIEmbedder
from rag_sdk.providers.pgvector import PgVectorIndexer
from rag_sdk.providers.qdrant import QdrantIndexer
from rag_sdk.workflow.ingestion import IngestionWorkflow
from run_eval import DIMENSIONS, load_golden

HERE = Path(__file__).parent
KS = (1, 3, 5, 10)


class MemoEmbedder(Embedder):
    """Embeds each distinct text once, whichever method asks, and counts provider inputs."""

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
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self.texts_embedded += len(missing)
            for text, vec in zip(missing, await self._inner.embed_chunks(missing, batch_size)):
                self._cache[text] = vec
        return [self._cache[t] for t in texts]


async def main(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    pages = split_pages(args.corpus.read_text())
    golden = load_golden(args.golden, pages)
    baseline = {
        r["query"]: r["first_relevant_rank"]
        for r in json.loads(args.baseline.read_text())["per_query"]
    }

    suffix = uuid.uuid4().hex[:10]
    collection, exact_table, hnsw_table = (
        f"eval_{suffix}",
        f"eval_exact_{suffix}",
        f"eval_hnsw_{suffix}",
    )
    qdrant = AsyncQdrantClient(url=args.qdrant_url)
    openai_client = AsyncOpenAI(api_key=api_key)
    pool = await asyncpg.create_pool(args.postgres_dsn, min_size=1, max_size=4)
    cleanup: list[Any] = []
    try:
        pg_version = await pool.fetchval("SHOW server_version")
        vector_version = await pool.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        qdrant_version = httpx.get(args.qdrant_url).json().get("version", "unknown")

        embedder = MemoEmbedder(
            OpenAIEmbedder(openai_client, model=args.embedding_model, dimensions=DIMENSIONS)
        )
        arms: dict[str, tuple[Indexer, str]] = {
            "qdrant": (QdrantIndexer(qdrant, embedder, collection), collection),
            "pgvector_exact": (PgVectorIndexer(pool, embedder, exact_table), exact_table),
            "pgvector_hnsw_iterative": (
                PgVectorIndexer(pool, embedder, hnsw_table, create_hnsw_index=True),
                hnsw_table,
            ),
        }
        cleanup.append(lambda: qdrant.delete_collection(collection))
        for table in (exact_table, hnsw_table):
            cleanup.append(lambda t=table: pool.execute(f'DROP TABLE IF EXISTS "{t}"'))

        retrievers: dict[str, Retriever] = {}
        for arm, (indexer, index) in arms.items():
            workflow = IngestionWorkflow(embedder, indexer, index, AppSchemaExtension())
            await workflow.ensure_index_exists()
            await workflow.ingest(DocumentContext(filename="eval_corpus.txt"), pages)
            retrievers[arm] = indexer.create_searcher(index)
        async with pool.acquire() as conn:
            for table in (exact_table, hnsw_table):
                await conn.execute(f'ANALYZE "{table}"')

        reports = {arm: await evaluate(r, golden, ks=KS) for arm, r in retrievers.items()}
        rows = {arm: rep.per_query() for arm, rep in reports.items()}

        for arm in ("qdrant", "pgvector_exact"):  # the comparison is only valid if these match
            drift = [
                r["query"] for r in rows[arm] if baseline[r["query"]] != r["first_relevant_rank"]
            ]
            if drift:
                raise SystemExit(f"{arm} differs from dense_baseline.json on {len(drift)} queries")

        retrieved = {arm: [q.retrieved for q in rep.queries] for arm, rep in reports.items()}
        agreement = {
            f"{arm}_vs_qdrant_identical_top10": sum(
                a == b for a, b in zip(retrieved[arm], retrieved["qdrant"])
            )
            for arm in ("pgvector_exact", "pgvector_hnsw_iterative")
        }
        record = {
            "date": date.today().isoformat(),
            "qdrant_server_version": qdrant_version,
            "postgres_version": pg_version,
            "pgvector_extension_version": vector_version,
            "embedding_model": args.embedding_model,
            "embedding_dimensions": DIMENSIONS,
            "corpus_paragraphs": len(pages),
            "golden_queries": len(golden),
            "openai_embedding_inputs": embedder.texts_embedded,
            "matches_recorded_baseline": ["qdrant", "pgvector_exact"],
            "queries_with_identical_top10_list": agreement,
            "arms": {arm: rep.summary() for arm, rep in reports.items()},
            "per_query": rows,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")

        print(
            f"{len(golden)} queries, {len(pages)} paragraphs, "
            f"embeddings sent: {embedder.texts_embedded}"
        )
        for arm, rep in reports.items():
            o = rep.summary()["overall"]
            print(
                f"  {arm:25} recall@5 {o['recall']['5']}  mrr@10 {o['mrr']['10']}  "
                f"ndcg@10 {o['ndcg']['10']}"
            )
        print(f"  identical top-10 lists vs qdrant: {agreement}")
        return 0
    finally:
        for step in reversed(cleanup):
            try:
                await step()
            except Exception as exc:  # keep cleaning the rest
                print(f"cleanup warning: {exc}", file=sys.stderr)
        await pool.close()
        await qdrant.close()
        await openai_client.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--qdrant-url", required=True, help="THROWAWAY Qdrant (no API key is sent)")
    p.add_argument("--postgres-dsn", required=True, help="THROWAWAY Postgres with pgvector")
    p.add_argument("--corpus", type=Path, default=HERE / "corpus.txt")
    p.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    p.add_argument("--baseline", type=Path, default=HERE / "results" / "dense_baseline.json")
    p.add_argument("--out", type=Path, default=HERE / "results" / "backend_compare.json")
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
