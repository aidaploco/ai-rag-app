from pathlib import Path

from qdrant_client import AsyncQdrantClient
from rag_app.parsing import parse_document
from rag_app.schema import AppSchemaExtension, DocumentContext
from rag_app.services.ingestion_service import IngestionService
from rag_sdk.providers.base import Embedder
from rag_sdk.providers.qdrant import QdrantIndexer
from rag_sdk.workflow.ingestion import IngestionWorkflow

PDF = Path(__file__).parent / "fixtures" / "two_page.pdf"


class StubEmbedder(Embedder):
    @property
    def dimensions(self) -> int:
        return 3

    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    async def embed_chunks(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


async def _stack():
    client = AsyncQdrantClient(":memory:")
    embedder = StubEmbedder()
    indexer = QdrantIndexer(client, embedder, "col")
    workflow = IngestionWorkflow(embedder, indexer, "col", AppSchemaExtension())
    await workflow.ensure_index_exists()
    return IngestionService(workflow, indexer), indexer


async def _positions(indexer: QdrantIndexer) -> dict[str, tuple[int, int | None]]:
    results = await indexer.create_searcher("col", top_k=50).retrieve("q")
    return {r.content: (r.metadata["chunk_idx"], r.metadata.get("page_nr")) for r in results}


async def test_pdf_chunk_idx_and_page_nr_reach_the_stored_payload():
    service, indexer = await _stack()
    pages = parse_document("two_page.pdf", None, PDF.read_bytes())
    assert await service.ingest_with_dedup(DocumentContext("two_page.pdf"), pages) == (2, 0)

    stored = await _positions(indexer)
    assert sorted(stored.values()) == [(0, 1), (1, 2)]


async def test_pinned_limit_chunk_idx_is_a_first_ingest_position_after_edits():
    service, indexer = await _stack()
    ctx = DocumentContext("notes.txt")
    await service.ingest_with_dedup(ctx, parse_document("notes.txt", None, b"alpha\n\nbeta"))

    # New version: one paragraph inserted before the existing two, one appended. New chunks get
    # their true position among all chunks of the upload, whatever was skipped as already stored.
    v2 = parse_document("notes.txt", None, b"intro\n\nalpha\n\nbeta\n\ngamma")
    assert await service.ingest_with_dedup(ctx, v2) == (2, 2)

    stored = await _positions(indexer)
    assert stored["intro"] == (0, None)
    assert stored["gamma"] == (3, None)  # true position, not its index among the new pages
    # Known limit: existing chunks keep the position from their first ingest, so "alpha" (now
    # at index 1) still says 0 and shares it with "intro". Pinned here, not endorsed.
    assert stored["alpha"] == (0, None)
