"""The real app lifespan against a throwaway Postgres (see PGVECTOR_TEST_DSN in the SDK tests)."""

import os
import uuid

import asyncpg
import pytest
from asgi_lifespan import LifespanManager
from rag_app.main import create_app


@pytest.fixture
def dsn() -> str:
    value = os.environ.get("PGVECTOR_TEST_DSN")
    if not value:
        if os.environ.get("CI"):
            pytest.fail("PGVECTOR_TEST_DSN is not set in CI; pgvector tests would silently not run")
        pytest.skip("PGVECTOR_TEST_DSN not set")
    return value


async def test_lifespan_creates_the_table_and_closes_the_pool(monkeypatch, dsn):
    table = f"lifespan_{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")  # never used: no embedding happens here
    monkeypatch.setenv("VECTOR_STORE", "pgvector")
    monkeypatch.setenv("POSTGRES_DSN", dsn)
    monkeypatch.setenv("PGVECTOR_TABLE", table)

    app = create_app()
    verify = await asyncpg.connect(dsn)
    try:
        async with LifespanManager(app):
            assert await verify.fetchval("SELECT to_regclass($1)", f'"{table}"') is not None
            indexer = app.state.ingestion_service._indexer
            pool = indexer._pool
            assert await indexer.count_documents({"filename": "none"}) == 0
        with pytest.raises(asyncpg.InterfaceError):  # pool was closed on shutdown
            await pool.acquire()
    finally:
        await verify.execute(f'DROP TABLE IF EXISTS "{table}"')
        await verify.close()


async def test_lifespan_builds_the_chunker_when_recursive_is_configured(monkeypatch, dsn):
    pytest.importorskip("tiktoken")
    table = f"lifespan_{uuid.uuid4().hex[:10]}"
    for key, value in {
        "OPENAI_API_KEY": "test-key",
        "VECTOR_STORE": "pgvector",
        "POSTGRES_DSN": dsn,
        "PGVECTOR_TABLE": table,
        "CHUNKING": "recursive",
        "CHUNK_MAX_TOKENS": "30",
        "CHUNK_OVERLAP_TOKENS": "4",
    }.items():
        monkeypatch.setenv(key, value)

    app = create_app()
    verify = await asyncpg.connect(dsn)
    try:
        async with LifespanManager(app):
            chunker = app.state.chunker
            assert chunker is not None
            chunks = chunker("A fairly ordinary sentence about retrieval. " * 40)
            assert len(chunks) > 3
    finally:
        await verify.execute(f'DROP TABLE IF EXISTS "{table}"')
        await verify.close()
