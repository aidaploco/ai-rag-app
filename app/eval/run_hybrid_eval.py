"""Dense vs BM25 vs hybrid (RRF) on a scratch OpenSearch index.

    OPENAI_API_KEY=... uv run python app/eval/run_hybrid_eval.py --opensearch-url http://localhost:9200

OpenSearch must be a THROWAWAY node. The runner creates a randomly named scratch index and drops it
afterwards. Texts are embedded once. The configuration below is fixed before running (changing it
and re-running to find a better number would make the result meaningless); the dense arm must
reproduce results/dense_baseline.json per query, otherwise the run aborts.
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

import httpx
from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch
from rag_app.parsing import split_pages
from rag_app.schema import AppSchemaExtension, DocumentContext
from rag_sdk.eval.harness import evaluate
from rag_sdk.providers.openai import OpenAIEmbedder
from rag_sdk.providers.opensearch import OpenSearchIndexer
from rag_sdk.workflow.hybrid import HybridRetriever
from rag_sdk.workflow.ingestion import IngestionWorkflow
from run_backend_compare import MemoEmbedder
from run_eval import DIMENSIONS, load_golden
from run_rerank_eval import movement

HERE = Path(__file__).parent
KS = (1, 3, 5, 10)
# Fixed before the run. Do not tune against the result.
RRF_K = 60
DEPTH = 20
ANALYZER = "standard (OpenSearch default for text fields), BM25 k1=1.2 b=0.75"


def decreases(
    base_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Queries where any metric at any k is lower than in ``base_rows``."""
    out = []
    for b, n in zip(base_rows, new_rows, strict=True):
        worse = [
            f"{m}@{k}: {b[m][k]} -> {n[m][k]}"
            for m in ("recall", "mrr", "ndcg")
            for k in b[m]
            if n[m][k] < b[m][k] - 1e-9
        ]
        if worse:
            out.append(
                {
                    "query": b["query"],
                    "kind": b["kind"],
                    "first_relevant_rank": [b["first_relevant_rank"], n["first_relevant_rank"]],
                    "decreases": worse,
                }
            )
    return out


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

    index = f"eval_hybrid_{uuid.uuid4().hex[:10]}"
    client = AsyncOpenSearch(hosts=[args.opensearch_url])
    openai_client = AsyncOpenAI(api_key=api_key)
    try:
        version = httpx.get(args.opensearch_url).json()["version"]["number"]
        embedder = MemoEmbedder(
            OpenAIEmbedder(openai_client, model=args.embedding_model, dimensions=DIMENSIONS)
        )
        indexer = OpenSearchIndexer(client, embedder, index)
        workflow = IngestionWorkflow(embedder, indexer, index, AppSchemaExtension())
        try:
            await workflow.ensure_index_exists()
            await workflow.ingest(DocumentContext(filename="eval_corpus.txt"), pages)

            dense = indexer.create_searcher(index)
            lexical = indexer.create_lexical_searcher(index)
            hybrid = HybridRetriever([dense, lexical], rrf_k=RRF_K, depth=DEPTH, top_n=max(KS))
            reports = {
                "dense": await evaluate(dense, golden, ks=KS),
                "lexical_bm25": await evaluate(lexical, golden, ks=KS),
                "hybrid_rrf": await evaluate(hybrid, golden, ks=KS),
            }
        finally:
            await client.indices.delete(index=index, ignore_unavailable=True)

        rows = {arm: rep.per_query() for arm, rep in reports.items()}
        drift = [
            r["query"] for r in rows["dense"] if baseline[r["query"]] != r["first_relevant_rank"]
        ]
        if drift:
            raise SystemExit(f"dense arm differs from dense_baseline.json on {len(drift)}: {drift}")

        record = {
            "date": date.today().isoformat(),
            "opensearch_version": version,
            "embedding_model": args.embedding_model,
            "embedding_dimensions": DIMENSIONS,
            "config": {
                "rrf_k": RRF_K,
                "depth": DEPTH,
                "lexical": ANALYZER,
                "final_k_values": list(KS),
            },
            "corpus_paragraphs": len(pages),
            "golden_queries": len(golden),
            "openai_embedding_inputs": embedder.texts_embedded,
            "dense_matches_recorded_baseline": True,
            "arms": {arm: rep.summary() for arm, rep in reports.items()},
            "hybrid_vs_dense_movement": movement(rows["dense"], rows["hybrid_rrf"]),
            "lexical_vs_dense_movement": movement(rows["dense"], rows["lexical_bm25"]),
            "hybrid_queries_with_any_metric_decrease": decreases(rows["dense"], rows["hybrid_rrf"]),
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
                f"  {arm:13} recall@1 {o['recall']['1']} recall@5 {o['recall']['5']}  "
                f"mrr@10 {o['mrr']['10']}  ndcg@10 {o['ndcg']['10']}"
            )
        print("  hybrid vs dense:", record["hybrid_vs_dense_movement"])
        print("  lexical vs dense:", record["lexical_vs_dense_movement"])
        print(
            "  hybrid queries with any decrease: "
            f"{len(record['hybrid_queries_with_any_metric_decrease'])}"
        )
        return 0
    finally:
        await client.close()
        await openai_client.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--opensearch-url", required=True, help="THROWAWAY OpenSearch node")
    p.add_argument("--corpus", type=Path, default=HERE / "corpus.txt")
    p.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    p.add_argument("--baseline", type=Path, default=HERE / "results" / "dense_baseline.json")
    p.add_argument("--out", type=Path, default=HERE / "results" / "hybrid_vs_dense.json")
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
