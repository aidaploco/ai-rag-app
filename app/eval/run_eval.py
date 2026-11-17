"""Run the retrieval eval against a *scratch* Qdrant collection and record the scores.

Usage (from the repo root):

    OPENAI_API_KEY=... uv run python app/eval/run_eval.py \\
        --qdrant-url http://localhost:6333 --out app/eval/results/dense_baseline.json

The runner never reads `.env` and never uses a Qdrant API key. It creates its own collection
(refusing to run if that name already exists), indexes the synthetic corpus, scores the golden
set, and deletes the collection again, so it cannot touch real data.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from rag_app.parsing import split_pages
from rag_app.schema import AppSchemaExtension, DocumentContext
from rag_sdk.eval.harness import METRICS, GoldenQuery, evaluate
from rag_sdk.providers.base import Embedder
from rag_sdk.providers.openai import OpenAIEmbedder
from rag_sdk.providers.qdrant import QdrantIndexer
from rag_sdk.workflow.ingestion import IngestionWorkflow
from rag_sdk.workflow.schema_extension import content_id_for

HERE = Path(__file__).parent
# Fixed at the model default: the embedder does not yet send `dimensions` to the API.
DIMENSIONS = 1536
CEILING = 0.97  # an overall mean at/above this leaves no room to show an improvement


class CountingEmbedder(Embedder):
    """Wraps an embedder and counts how many texts were sent to the provider."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self.texts_embedded = 0

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed(self, text: str) -> list[float]:
        self.texts_embedded += 1
        return await self._inner.embed(text)

    async def embed_chunks(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]:
        self.texts_embedded += len(texts)
        return await self._inner.embed_chunks(texts, batch_size)


def load_golden(path: Path, pages: list[str]) -> list[GoldenQuery]:
    """Resolve each anchor phrase to the content ID of the single paragraph containing it."""
    golden = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids = set()
        for anchor in row["relevant_anchors"]:
            matches = [p for p in pages if anchor in p]
            if len(matches) != 1:
                raise SystemExit(f"anchor {anchor!r} matches {len(matches)} paragraphs, need 1")
            ids.add(content_id_for(matches[0]))
        golden.append(GoldenQuery(row["query"], frozenset(ids), row.get("kind", "default")))
    return golden


async def main(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    pages = split_pages(args.corpus.read_text())
    if len(set(pages)) != len(pages):
        raise SystemExit("corpus has duplicate paragraphs; their IDs would collide")
    golden = load_golden(args.golden, pages)

    qdrant = AsyncQdrantClient(url=args.qdrant_url)
    openai_client = AsyncOpenAI(api_key=api_key)
    try:
        if await qdrant.collection_exists(args.collection):
            raise SystemExit(f"collection {args.collection!r} already exists; refusing to reuse it")
        server_version = httpx.get(args.qdrant_url).json().get("version", "unknown")

        embedder = CountingEmbedder(
            OpenAIEmbedder(openai_client, model=args.model, dimensions=DIMENSIONS)
        )
        indexer = QdrantIndexer(qdrant, embedder, args.collection)
        workflow = IngestionWorkflow(embedder, indexer, args.collection, AppSchemaExtension())
        try:
            await workflow.ensure_index_exists()
            await workflow.ingest(DocumentContext(filename="eval_corpus.txt"), pages)

            # A stale label would read as a retrieval miss, so check every label is indexed.
            wanted = {i for g in golden for i in g.relevant}
            missing = wanted - await indexer.existing_content_ids(sorted(wanted))
            if missing:
                raise SystemExit(f"{len(missing)} golden labels are not in the collection")

            retriever = indexer.create_searcher(args.collection)
            report = await evaluate(retriever, golden, ks=args.ks)
        finally:
            await qdrant.delete_collection(args.collection)

        summary = report.summary()

        def saturated_in(block: dict[str, dict[str, float]]) -> list[str]:
            return [
                f"{m}@{k}" for m, by_k in block.items() for k, v in by_k.items() if v >= CEILING
            ]

        saturated = {
            "overall": saturated_in(summary["overall"]),
            **{
                kind: saturated_in({m: block[m] for m in METRICS})
                for kind, block in summary["by_kind"].items()
            },
        }
        not_at_rank_1 = sum(1 for q in report.queries if q.first_relevant_rank != 1)
        record: dict[str, Any] = {
            "date": date.today().isoformat(),
            "retriever": "dense (QdrantRetriever, cosine)",
            "embedding_model": args.model,
            "embedding_dimensions": DIMENSIONS,
            "qdrant_server_version": server_version,
            "corpus_paragraphs": len(pages),
            "golden_queries": len(golden),
            "openai_embedding_inputs": embedder.texts_embedded,
            "queries_without_relevant_at_rank_1": not_at_rank_1,
            "saturated_metrics": saturated,
            **summary,
            "per_query": report.per_query(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        for scope, names in saturated.items():
            if names:
                print(f"NOTE: {scope}: {', '.join(names)} >= {CEILING} (no headroom)")
        print(f"queries without a relevant result at rank 1: {not_at_rank_1}/{len(golden)}")
        return 0
    finally:
        await qdrant.close()
        await openai_client.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--qdrant-url", required=True, help="Qdrant URL (no API key is ever sent)")
    p.add_argument("--collection", default="eval_scratch", help="scratch collection to create")
    p.add_argument("--corpus", type=Path, default=HERE / "corpus.txt")
    p.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    p.add_argument("--out", type=Path, default=HERE / "results" / "dense_baseline.json")
    p.add_argument("--model", default="text-embedding-3-small")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
