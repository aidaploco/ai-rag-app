"""Compare chunking strategies on long, blank-line-free documents, scored by anchor containment.

    OPENAI_API_KEY=... uv run python app/eval/run_chunking_eval.py

Documents are derived from corpus.txt by a fixed rule: consecutive groups of 18 paragraphs joined
with single spaces (so the default blank-line split finds no boundaries, like text pulled from a
PDF). Queries and anchors come from golden.jsonl; the runner checks every anchor occurs exactly
once. Retrieval uses in-memory Qdrant (nothing to start) and OpenAI embeddings of synthetic text.
Arms, token budgets and depth below are declared before the run; the result is recorded whichever
strategy wins.

Scoring (rag_sdk.eval.anchors): a chunk finds an anchor if it contains the anchor text, and
strategies are compared at equal token budgets, so huge chunks cannot win by returning a whole
document. An anchor cut by a chunk boundary is found by no chunk and is counted per arm.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import tiktoken
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from rag_app.parsing import split_pages
from rag_app.schema import AppSchemaExtension, DocumentContext
from rag_sdk.chunking import chunk_text
from rag_sdk.eval.anchors import anchor_recall_at_budget, first_hit_rank, split_anchors
from rag_sdk.providers.openai import OpenAIEmbedder
from rag_sdk.providers.qdrant import QdrantIndexer
from rag_sdk.tokenizer import openai_token_counter
from rag_sdk.workflow.ingestion import IngestionWorkflow
from run_backend_compare import MemoEmbedder
from run_eval import DIMENSIONS

HERE = Path(__file__).parent
MODEL = "text-embedding-3-small"
GROUP = 18  # paragraphs per derived document
DEPTH = 20  # chunks retrieved per query
BUDGETS = (256, 512, 1024)  # tokens of retrieved context allowed per query

Strategy = Callable[[str], list[str]]


def make_arms(count: Callable[[str], int]) -> dict[str, Strategy]:
    encoding = tiktoken.encoding_for_model(MODEL)

    def fixed(size: int) -> Strategy:
        def run(doc: str) -> list[str]:  # naive: cut every `size` tokens, mid-sentence if need be
            ids = encoding.encode(doc)
            return [encoding.decode(ids[i : i + size]).strip() for i in range(0, len(ids), size)]

        return run

    def recursive(size: int, overlap: int) -> Strategy:
        return lambda doc: chunk_text(doc, count, max_tokens=size, overlap_tokens=overlap)

    return {
        "paragraph_split (current default)": split_pages,
        "fixed_64": fixed(64),
        "fixed_128": fixed(128),
        "recursive_64": recursive(64, 0),
        "recursive_64_overlap32": recursive(64, 32),
        "recursive_128": recursive(128, 0),
        "recursive_128_overlap64": recursive(128, 64),
    }


def load_golden(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        {"query": r["query"], "kind": r["kind"], "anchors": r["relevant_anchors"]} for r in rows
    ]


async def main(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    paragraphs = split_pages(args.corpus.read_text())
    docs = [" ".join(paragraphs[i : i + GROUP]) for i in range(0, len(paragraphs), GROUP)]
    golden = load_golden(args.golden)
    everything = "\n".join(docs)
    for g in golden:
        for anchor in g["anchors"]:
            if everything.count(anchor) != 1:
                raise SystemExit(f"anchor {anchor!r} occurs {everything.count(anchor)}x, need 1")
    all_anchors = sorted({a for g in golden for a in g["anchors"]})

    count = openai_token_counter(MODEL)
    arms = make_arms(count)
    openai_client = AsyncOpenAI(api_key=api_key)
    embedder = MemoEmbedder(OpenAIEmbedder(openai_client, model=MODEL, dimensions=DIMENSIONS))
    record_arms: dict[str, Any] = {}
    try:
        for name, strategy in arms.items():
            chunks_by_doc = [strategy(doc) for doc in docs]
            chunks = list(dict.fromkeys(c for cs in chunks_by_doc for c in cs if c))
            tokens = [count(c) for c in chunks]

            client = AsyncQdrantClient(":memory:")
            index = "chunks"
            indexer = QdrantIndexer(client, embedder, index)
            workflow = IngestionWorkflow(embedder, indexer, index, AppSchemaExtension())
            await workflow.ensure_index_exists()
            await workflow.ingest(DocumentContext(filename="long_docs"), chunks)
            retriever = indexer.create_searcher(index, top_k=DEPTH)

            rows = []
            for g in golden:
                ranked = [r.content for r in await retriever.retrieve(g["query"])]
                rows.append(
                    {
                        "query": g["query"],
                        "kind": g["kind"],
                        "first_hit_rank": first_hit_rank(ranked, g["anchors"]),
                        "anchor_recall": {
                            str(b): round(
                                anchor_recall_at_budget(ranked, g["anchors"], count, b), 4
                            )
                            for b in BUDGETS
                        },
                    }
                )
            await client.close()

            def mean(kind: str | None, key: str) -> float:
                sel = [r for r in rows if kind is None or r["kind"] == kind]
                return round(statistics.fmean(r["anchor_recall"][key] for r in sel), 4)

            def mrr(kind: str | None) -> float:
                sel = [r for r in rows if kind is None or r["kind"] == kind]
                return round(
                    statistics.fmean(
                        1 / r["first_hit_rank"] if r["first_hit_rank"] else 0 for r in sel
                    ),
                    4,
                )

            kinds = sorted({g["kind"] for g in golden})
            cut = {
                a
                for doc, cs in zip(docs, chunks_by_doc)
                for a in split_anchors(doc, cs, all_anchors)
            }
            record_arms[name] = {
                "chunks": len(chunks),
                "tokens_mean": round(statistics.fmean(tokens), 1),
                "tokens_max": max(tokens),
                "anchors_cut_by_a_boundary": len(cut),
                "anchors_total": len(all_anchors),
                "overall": {
                    "anchor_recall_at_budget": {str(b): mean(None, str(b)) for b in BUDGETS},
                    "mrr": mrr(None),
                },
                "by_kind": {
                    k: {
                        "anchor_recall_at_budget": {str(b): mean(k, str(b)) for b in BUDGETS},
                        "mrr": mrr(k),
                    }
                    for k in kinds
                },
                "per_query": rows,
            }

        record = {
            "date": date.today().isoformat(),
            "embedding_model": MODEL,
            "documents": len(docs),
            "paragraphs_per_document": GROUP,
            "document_tokens": [count(d) for d in docs],
            "golden_queries": len(golden),
            "retrieval_depth": DEPTH,
            "token_budgets": list(BUDGETS),
            "openai_embedding_inputs": embedder.texts_embedded,
            "arms": record_arms,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")

        print(
            f"{len(docs)} docs ({record['document_tokens']} tokens), {len(golden)} queries, "
            f"embeddings sent: {embedder.texts_embedded}"
        )
        header = " ".join(f"R@{b:<4}" for b in BUDGETS)
        print(f"{'arm':36} {'chunks':>6} {'mean':>5} {'cut':>4}  {header} {'MRR':>6}")
        for name, a in record_arms.items():
            recalls = " ".join(
                f"{a['overall']['anchor_recall_at_budget'][str(b)]:<6}" for b in BUDGETS
            )
            cut = a["anchors_cut_by_a_boundary"]
            print(
                f"{name:36} {a['chunks']:>6} {a['tokens_mean']:>5} {cut:>4}  "
                f"{recalls} {a['overall']['mrr']:>6}"
            )
        return 0
    finally:
        await openai_client.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--corpus", type=Path, default=HERE / "corpus.txt")
    p.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    p.add_argument("--out", type=Path, default=HERE / "results" / "chunking.json")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
