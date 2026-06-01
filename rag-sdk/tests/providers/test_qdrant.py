from typing import Any
from unittest.mock import AsyncMock, MagicMock

from rag_sdk.providers.base import Document, Embedder, SearchResults
from rag_sdk.providers.qdrant import QdrantIndexer, QdrantRetriever


class StubEmbedder(Embedder):
    @property
    def dimensions(self) -> int:
        return 3

    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def embed_chunks(
        self, texts: list[str], batch_size: int = 2048
    ) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def _mock_qdrant() -> MagicMock:
    client = MagicMock()
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.search = AsyncMock(return_value=[])
    client.retrieve = AsyncMock(return_value=[])
    count = MagicMock()
    count.count = 0
    client.count = AsyncMock(return_value=count)
    return client


def _doc(content_id: str = "abc", content: str = "hello") -> Document:
    return Document(
        content_id=content_id,
        content=content,
        content_vector=[0.1, 0.2, 0.3],
        metadata={"page_nr": 0, "source": "test.pdf"},
    )


# --- QdrantIndexer ---

async def test_create_or_update_index_creates_collection():
    client = _mock_qdrant()
    indexer = QdrantIndexer(client=client, embedder=StubEmbedder(), collection="col")
    await indexer.create_or_update_index("col", {})
    client.create_collection.assert_called_once()


async def test_create_or_update_index_skips_existing():
    client = _mock_qdrant()
    client.collection_exists = AsyncMock(return_value=True)
    indexer = QdrantIndexer(client=client, embedder=StubEmbedder(), collection="col")
    await indexer.create_or_update_index("col", {})
    client.create_collection.assert_not_called()


async def test_upload_documents_calls_upsert():
    client = _mock_qdrant()
    indexer = QdrantIndexer(client=client, embedder=StubEmbedder(), collection="col")
    await indexer.upload_documents([_doc()])
    client.upsert.assert_called_once()


async def test_upload_document_point_payload():
    client = _mock_qdrant()
    indexer = QdrantIndexer(client=client, embedder=StubEmbedder(), collection="col")
    await indexer.upload_documents([_doc(content_id="uuid-1", content="the text")])
    point = client.upsert.call_args.kwargs["points"][0]
    assert point.id == "uuid-1"
    assert point.payload["content"] == "the text"
    assert point.payload["page_nr"] == 0
    assert point.payload["source"] == "test.pdf"


async def test_has_document_by_content_id_true():
    client = _mock_qdrant()
    client.retrieve = AsyncMock(return_value=[MagicMock()])
    indexer = QdrantIndexer(client=client, embedder=StubEmbedder(), collection="col")
    assert await indexer.has_document({"content_id": "abc"}) is True


async def test_has_document_by_content_id_false():
    client = _mock_qdrant()
    indexer = QdrantIndexer(client=client, embedder=StubEmbedder(), collection="col")
    assert await indexer.has_document({"content_id": "xyz"}) is False


async def test_create_searcher_returns_retriever():
    client = _mock_qdrant()
    indexer = QdrantIndexer(client=client, embedder=StubEmbedder(), collection="col")
    searcher = indexer.create_searcher("col")
    assert isinstance(searcher, QdrantRetriever)


# --- QdrantRetriever ---

async def test_retriever_maps_hits_to_search_results():
    client = _mock_qdrant()
    hit = MagicMock()
    hit.score = 0.95
    hit.payload = {"content": "relevant text", "page_nr": 1}
    client.search = AsyncMock(return_value=[hit])

    retriever = QdrantRetriever(
        client=client, collection="col", embedder=StubEmbedder()
    )
    results = await retriever.retrieve("query")

    assert len(results) == 1
    assert results[0].content == "relevant text"
    assert results[0].score == 0.95
    assert results[0].metadata["page_nr"] == 1
    assert "content" not in results[0].metadata


async def test_retriever_passes_filter_to_search():
    client = _mock_qdrant()
    retriever = QdrantRetriever(
        client=client, collection="col", embedder=StubEmbedder()
    )
    await retriever.retrieve("query", filters={"source": "file.pdf"})
    call_kwargs = client.search.call_args.kwargs
    assert call_kwargs["query_filter"] is not None


async def test_retriever_no_filter_when_none():
    client = _mock_qdrant()
    retriever = QdrantRetriever(
        client=client, collection="col", embedder=StubEmbedder()
    )
    await retriever.retrieve("query")
    call_kwargs = client.search.call_args.kwargs
    assert call_kwargs["query_filter"] is None
