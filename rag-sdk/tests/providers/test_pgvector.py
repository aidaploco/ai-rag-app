"""pgvector-specific behaviour. The shared contract lives in test_indexer_contract.py."""

import os
import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest
from rag_sdk.providers.pgvector import PgVectorIndexer, PgVectorRetriever

from tests.providers.test_indexer_contract import DIMS, HashEmbedder, make_docs


class OtherDimsEmbedder(HashEmbedder):
    @property
    def dimensions(self) -> int:
        return DIMS // 2


@pytest.fixture
async def pool() -> AsyncIterator[asyncpg.Pool]:
    dsn = os.environ.get("PGVECTOR_TEST_DSN")
    if not dsn:
        if os.environ.get("CI"):
            pytest.fail("PGVECTOR_TEST_DSN is not set in CI; pgvector tests would silently not run")
        pytest.skip("PGVECTOR_TEST_DSN not set")
    created = await asyncpg.create_pool(dsn, min_size=1, max_size=4)
    yield created
    await created.close()


@pytest.fixture
async def table(pool: asyncpg.Pool) -> AsyncIterator[str]:
    name = f"pg_{uuid.uuid4().hex[:12]}"
    yield name
    async with pool.acquire() as conn:
        await conn.execute(f'DROP TABLE IF EXISTS "{name}"')


async def test_existing_table_with_other_dimensions_is_rejected(pool, table):
    await PgVectorIndexer(pool, HashEmbedder(), table).create_or_update_index(table, {})
    with pytest.raises(ValueError, match="vector\\(16\\).*produces 8"):
        await PgVectorIndexer(pool, OtherDimsEmbedder(), table).create_or_update_index(table, {})


@pytest.mark.parametrize("bad", ["", "1abc", "has space", 'q"uote', "semi;colon", "a" * 41, "x-y"])
def test_invalid_table_names_are_rejected(bad):
    with pytest.raises(ValueError, match="invalid table name"):
        PgVectorRetriever(None, bad, HashEmbedder())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid table name"):
        PgVectorIndexer(None, HashEmbedder(), bad)  # type: ignore[arg-type]


def test_table_name_at_the_length_cap_is_accepted():
    PgVectorRetriever(None, "a" * 40, HashEmbedder())  # type: ignore[arg-type]


def test_unknown_iterative_scan_mode_is_rejected():
    with pytest.raises(ValueError):
        PgVectorRetriever(None, "t", HashEmbedder(), hnsw_iterative_scan="off; drop")  # type: ignore[arg-type]


async def test_hnsw_is_opt_in(pool, table):
    await PgVectorIndexer(pool, HashEmbedder(), table).create_or_update_index(table, {})
    async with pool.acquire() as conn:
        indexes = {
            r["indexname"]
            for r in await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE tablename = $1", table
            )
        }
    assert f"{table}_embedding_hnsw_idx" not in indexes
    assert f"{table}_metadata_idx" in indexes


async def test_hnsw_without_iterative_scan_underfills_and_with_it_fills_top_k(pool, table):
    indexer = PgVectorIndexer(pool, HashEmbedder(), table, create_hnsw_index=True)
    await indexer.create_or_update_index(table, {})
    # 30% of rows match the filter: the case where post-filtering an HNSW scan falls short.
    docs = await make_docs(
        *[(f"item {i}", {"group": "g1" if i % 10 in (1, 2, 3) else "g2"}) for i in range(3000)]
    )
    await indexer.upload_documents(docs)
    async with pool.acquire() as conn:
        await conn.execute(f'ANALYZE "{table}"')
        indexes = {
            r["indexname"]
            for r in await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE tablename = $1", table
            )
        }
    assert f"{table}_embedding_hnsw_idx" in indexes

    plain = PgVectorRetriever(pool, table, HashEmbedder(), top_k=40)  # no iterative scan
    assert len(await plain.retrieve("item 7", filters={"group": "g1"})) < 40  # the pitfall

    fixed = indexer.create_searcher(table, top_k=40)  # strict_order iterative scan
    results = await fixed.retrieve("item 7", filters={"group": "g1"})
    assert len(results) == 40 and all(r.metadata["group"] == "g1" for r in results)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
