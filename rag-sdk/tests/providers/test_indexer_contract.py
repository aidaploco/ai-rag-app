"""One behavioural contract, run against every vector-store provider.

pgvector needs a real Postgres (PGVECTOR_TEST_DSN) and OpenSearch a real node
(OPENSEARCH_TEST_URL). Point them at *throwaway* instances only: the tests create and drop their
own uniquely named tables/indexes. Without the variable the cases are skipped locally; in CI
(CI=true) a missing one is a failure so the tests cannot silently stop running.
"""

import hashlib
import math
import os
import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest
from opensearchpy import AsyncOpenSearch
from qdrant_client import AsyncQdrantClient
from rag_sdk.providers.base import Document, Embedder, Indexer
from rag_sdk.providers.opensearch import OpenSearchIndexer
from rag_sdk.providers.pgvector import PgVectorIndexer
from rag_sdk.providers.qdrant import QdrantIndexer
from rag_sdk.workflow.schema_extension import content_id_for

DIMS = 16


class HashEmbedder(Embedder):
    """Deterministic and text-dependent: equal text -> equal vector, different text -> different."""

    @property
    def dimensions(self) -> int:
        return DIMS

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 - 0.5 for b in digest[:DIMS]]

    async def embed_chunks(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


async def make_docs(*items: tuple[str, dict]) -> list[Document]:
    embedder = HashEmbedder()
    return [
        Document(content_id_for(text), text, await embedder.embed(text), metadata)
        for text, metadata in items
    ]


Backend = tuple[Indexer, str]  # indexer, index name


@pytest.fixture(params=["qdrant", "pgvector", "opensearch"])
async def backend(request: pytest.FixtureRequest) -> AsyncIterator[Backend]:
    name = f"contract_{uuid.uuid4().hex[:12]}"
    embedder = HashEmbedder()
    if request.param == "qdrant":
        client = AsyncQdrantClient(":memory:")
        indexer: Indexer = QdrantIndexer(client, embedder, name)
        await indexer.create_or_update_index(name, {})
        yield indexer, name
        await client.close()
        return

    if request.param == "opensearch":
        url = os.environ.get("OPENSEARCH_TEST_URL")
        if not url:
            if os.environ.get("CI"):
                pytest.fail("OPENSEARCH_TEST_URL is not set in CI; opensearch tests would not run")
            pytest.skip("OPENSEARCH_TEST_URL not set")
        os_client = AsyncOpenSearch(hosts=[url])
        indexer = OpenSearchIndexer(os_client, embedder, name)
        try:
            await indexer.create_or_update_index(name, {})
            yield indexer, name
        finally:
            await os_client.indices.delete(index=name, ignore_unavailable=True)
            await os_client.close()
        return

    dsn = os.environ.get("PGVECTOR_TEST_DSN")
    if not dsn:
        if os.environ.get("CI"):
            pytest.fail("PGVECTOR_TEST_DSN is not set in CI; pgvector tests would silently not run")
        pytest.skip("PGVECTOR_TEST_DSN not set")
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4)
    indexer = PgVectorIndexer(pool, embedder, name)
    try:
        await indexer.create_or_update_index(name, {})
        yield indexer, name
    finally:
        async with pool.acquire() as conn:
            await conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        await pool.close()


async def test_create_or_update_index_is_idempotent(backend: Backend):
    indexer, name = backend
    await indexer.create_or_update_index(name, {})
    await indexer.create_or_update_index(name, {})


async def test_existing_content_ids(backend: Backend):
    indexer, _ = backend
    docs = await make_docs(("alpha", {}), ("beta", {}))
    await indexer.upload_documents(docs)
    absent = content_id_for("never uploaded")
    found = await indexer.existing_content_ids([d.content_id for d in docs] + [absent])
    assert found == {d.content_id for d in docs}
    assert await indexer.existing_content_ids([]) == set()


async def test_count_and_has_document(backend: Backend):
    indexer, _ = backend
    docs = await make_docs(
        ("a", {"filename": "x.txt"}), ("b", {"filename": "x.txt"}), ("c", {"filename": "y.txt"})
    )
    await indexer.upload_documents(docs)
    assert await indexer.count_documents({"filename": "x.txt"}) == 2
    assert await indexer.count_documents({"filename": "nope"}) == 0
    assert await indexer.count_documents({"content_id": docs[0].content_id}) == 1
    assert await indexer.count_documents({"content_id": content_id_for("zzz")}) == 0
    assert await indexer.has_document({"filename": "y.txt"}) is True
    assert await indexer.has_document({"filename": "nope"}) is False
    assert await indexer.has_document({"content_id": docs[2].content_id}) is True


async def test_retrieve_ranks_exact_match_first_with_similarity_scores(backend: Backend):
    indexer, name = backend
    await indexer.upload_documents(
        await make_docs(*[(f"doc {i}", {"n": i, "filename": "f.txt"}) for i in range(10)])
    )
    results = await indexer.create_searcher(name, top_k=5).retrieve("doc 3")
    assert len(results) == 5
    assert results[0].content == "doc 3"
    assert results[0].content_id == content_id_for("doc 3")
    assert results[0].score == pytest.approx(1.0, abs=1e-5)  # identical vector: cosine similarity
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert results[0].metadata == {"n": 3, "filename": "f.txt"}  # payload minus content


async def test_scores_equal_cosine_similarity_of_the_vectors(backend: Backend):
    indexer, name = backend
    texts = [f"passage {i}" for i in range(12)]
    await indexer.upload_documents(await make_docs(*[(t, {}) for t in texts]))
    embedder = HashEmbedder()
    query = await embedder.embed("passage 4")

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

    expected = {content_id_for(t): cosine(query, await embedder.embed(t)) for t in texts}
    results = await indexer.create_searcher(name, top_k=12).retrieve("passage 4")
    assert len(results) == 12
    for r in results:
        assert r.content_id is not None
        assert r.score == pytest.approx(expected[r.content_id], abs=1e-4)


async def test_top_k_argument_overrides_default(backend: Backend):
    indexer, name = backend
    await indexer.upload_documents(await make_docs(*[(f"d{i}", {}) for i in range(8)]))
    searcher = indexer.create_searcher(name, top_k=3)
    assert len(await searcher.retrieve("d1")) == 3
    assert len(await searcher.retrieve("d1", top_k=6)) == 6


async def test_filters_are_and_equality(backend: Backend):
    indexer, name = backend
    await indexer.upload_documents(
        await make_docs(
            ("one", {"filename": "a.txt", "page_nr": 1}),
            ("two", {"filename": "a.txt", "page_nr": 2}),
            ("three", {"filename": "b.txt", "page_nr": 1}),
        )
    )
    searcher = indexer.create_searcher(name, top_k=10)
    by_file = await searcher.retrieve("one", filters={"filename": "a.txt"})
    assert {r.content for r in by_file} == {"one", "two"}
    both = await searcher.retrieve("one", filters={"filename": "a.txt", "page_nr": 2})
    assert [r.content for r in both] == ["two"]
    assert await searcher.retrieve("one", filters={"filename": "missing"}) == []


async def test_selective_filter_still_returns_top_k(backend: Backend):
    indexer, name = backend
    await indexer.upload_documents(
        await make_docs(
            *[(f"item {i}", {"group": "keep" if i % 5 == 0 else "drop"}) for i in range(60)]
        )
    )
    results = await indexer.create_searcher(name, top_k=10).retrieve(
        "item 1", filters={"group": "keep"}
    )
    assert len(results) == 10 and all(r.metadata["group"] == "keep" for r in results)


async def test_upsert_overwrites_existing_id(backend: Backend):
    indexer, name = backend
    first = (await make_docs(("same text", {"filename": "old.txt"})))[0]
    await indexer.upload_documents([first])
    second = Document(first.content_id, "same text", first.content_vector, {"filename": "new.txt"})
    await indexer.upload_documents([second])
    assert await indexer.count_documents({"content_id": first.content_id}) == 1
    assert await indexer.count_documents({"filename": "old.txt"}) == 0
    assert await indexer.count_documents({"filename": "new.txt"}) == 1


async def test_repeated_id_in_one_batch_last_write_wins(backend: Backend):
    indexer, _ = backend
    doc = (await make_docs(("dup", {"v": 1})))[0]
    later = Document(doc.content_id, "dup", doc.content_vector, {"v": 2})
    await indexer.upload_documents([doc, later])
    assert await indexer.count_documents({"v": 2}) == 1
    assert await indexer.count_documents({"v": 1}) == 0


async def test_empty_upload_is_a_noop(backend: Backend):
    indexer, _ = backend
    await indexer.upload_documents([])
    assert await indexer.count_documents({"filename": "anything"}) == 0
