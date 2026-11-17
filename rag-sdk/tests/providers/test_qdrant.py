from unittest.mock import AsyncMock, MagicMock

from qdrant_client import AsyncQdrantClient
from rag_sdk.providers.base import Document, Embedder
from rag_sdk.providers.qdrant import QdrantIndexer, QdrantRetriever


class StubEmbedder(Embedder):
    @property
    def dimensions(self) -> int:
        return 3

    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def embed_chunks(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def _mock_qdrant() -> MagicMock:
    # spec= makes calls to methods the real client lacks (e.g. the removed `search`) fail loudly.
    client = MagicMock(spec=AsyncQdrantClient)
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.query_points = AsyncMock(return_value=MagicMock(points=[]))
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


async def test_has_document_by_field_filter_uses_count():
    client = _mock_qdrant()
    client.count = AsyncMock(return_value=MagicMock(count=2))
    indexer = QdrantIndexer(client=client, embedder=StubEmbedder(), collection="col")
    assert await indexer.has_document({"filename": "a.txt"}) is True
    client.retrieve.assert_not_called()


async def test_existence_checks_do_not_fetch_payloads():
    client = _mock_qdrant()
    indexer = QdrantIndexer(client=client, embedder=StubEmbedder(), collection="col")
    await indexer.existing_content_ids(["a", "b"])
    await indexer.count_documents({"content_id": "a"})
    assert client.retrieve.call_count == 2
    assert all(c.kwargs["with_payload"] is False for c in client.retrieve.call_args_list)


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
    client.query_points = AsyncMock(return_value=MagicMock(points=[hit]))

    retriever = QdrantRetriever(client=client, collection="col", embedder=StubEmbedder())
    results = await retriever.retrieve("query")

    assert len(results) == 1
    assert results[0].content == "relevant text"
    assert results[0].score == 0.95
    assert results[0].metadata["page_nr"] == 1
    assert "content" not in results[0].metadata


async def test_retriever_tolerates_missing_payload():
    client = _mock_qdrant()
    hit = MagicMock()
    hit.score = 0.5
    hit.payload = None
    client.query_points = AsyncMock(return_value=MagicMock(points=[hit]))

    retriever = QdrantRetriever(client=client, collection="col", embedder=StubEmbedder())
    results = await retriever.retrieve("query")

    assert results[0].content == ""
    assert results[0].metadata == {}


async def test_retriever_queries_with_embedded_vector_and_top_k():
    client = _mock_qdrant()
    retriever = QdrantRetriever(client=client, collection="col", embedder=StubEmbedder(), top_k=7)
    await retriever.retrieve("query")
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "col"
    assert kwargs["query"] == [0.1, 0.2, 0.3]
    assert kwargs["limit"] == 7


async def test_retriever_passes_filter_to_search():
    client = _mock_qdrant()
    retriever = QdrantRetriever(client=client, collection="col", embedder=StubEmbedder())
    await retriever.retrieve("query", filters={"source": "file.pdf"})
    call_kwargs = client.query_points.call_args.kwargs
    assert call_kwargs["query_filter"] is not None


async def test_retriever_no_filter_when_none():
    client = _mock_qdrant()
    retriever = QdrantRetriever(client=client, collection="col", embedder=StubEmbedder())
    await retriever.retrieve("query")
    call_kwargs = client.query_points.call_args.kwargs
    assert call_kwargs["query_filter"] is None


# --- Against a real Qdrant client (in-memory local mode): no mocks of the client API ---

_ABSENT_ID = "00000000-0000-0000-0000-0000000000ff"  # well-formed UUID that is never upserted


async def test_roundtrip_against_in_memory_qdrant():
    await _roundtrip(AsyncQdrantClient(":memory:"))


async def _roundtrip(client: AsyncQdrantClient, collection: str = "col") -> None:
    embedder = StubEmbedder()
    indexer = QdrantIndexer(client=client, embedder=embedder, collection=collection)
    await indexer.create_or_update_index(collection, {})

    docs = [
        Document(
            content_id="00000000-0000-0000-0000-000000000001",
            content="alpha",
            content_vector=[1.0, 0.0, 0.0],
            metadata={"filename": "a.txt", "page_nr": 0},
        ),
        Document(
            content_id="00000000-0000-0000-0000-000000000002",
            content="beta",
            content_vector=[0.0, 1.0, 0.0],
            metadata={"filename": "b.txt", "page_nr": 0},
        ),
    ]
    await indexer.upload_documents(docs)

    assert await indexer.existing_content_ids([d.content_id for d in docs] + [_ABSENT_ID]) == {
        d.content_id for d in docs
    }
    assert await indexer.count_documents({"filename": "a.txt"}) == 1

    retriever = indexer.create_searcher(collection, top_k=5)
    results = await retriever.retrieve("anything")
    assert {r.content for r in results} == {"alpha", "beta"}
    assert all("content" not in r.metadata for r in results)

    filtered = await retriever.retrieve("anything", filters={"filename": "b.txt"})
    assert [r.content for r in filtered] == ["beta"]
    assert filtered[0].metadata["filename"] == "b.txt"
