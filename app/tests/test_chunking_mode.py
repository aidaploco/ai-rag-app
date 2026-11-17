from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from rag_app.config import Config
from rag_app.main import build_chunker, create_app
from rag_app.parsing import parse_document
from rag_app.services.ingestion_service import IngestionService
from rag_sdk.chunking import chunk_text

FIXTURES = Path(__file__).parent / "fixtures"
WALL = " ".join(f"Sentence number {i} says something distinct about topic {i}." for i in range(80))


def words(text: str) -> int:
    return len(text.split())


def word_chunker(text: str) -> list[str]:
    return chunk_text(text, words, max_tokens=20, overlap_tokens=0)


def test_default_is_one_huge_chunk_for_text_without_blank_lines():
    pages = parse_document("wall.txt", None, WALL.encode())
    assert len(pages) == 1 and words(pages[0].text) > 500  # the problem the chunker solves


def test_chunker_keeps_text_without_blank_lines_under_the_cap():
    pages = parse_document("wall.txt", None, WALL.encode(), chunker=word_chunker)
    assert len(pages) > 20
    assert all(words(p.text) <= 20 for p in pages)
    assert [p.chunk_idx for p in pages] == list(range(len(pages)))
    assert all(p.page_nr is None for p in pages)


def test_pdf_chunks_keep_their_real_page_number_and_a_global_chunk_idx():
    raw = (FIXTURES / "two_page.pdf").read_bytes()
    pages = parse_document("two_page.pdf", None, raw, chunker=lambda t: chunk_text(t, words, 4))
    assert {p.page_nr for p in pages} == {1, 2}
    assert [p.chunk_idx for p in pages] == list(range(len(pages)))
    assert [p.page_nr for p in pages] == sorted(p.page_nr for p in pages)  # type: ignore[type-var]


def test_default_chunker_is_none_and_validation_rejects_bad_overlap(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("CHUNKING", raising=False)
    config = Config(_env_file=None)
    assert config.chunking == "paragraph" and build_chunker(config) is None

    monkeypatch.setenv("CHUNKING", "recursive")
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "50")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "50")
    with pytest.raises(ValueError, match="CHUNK_OVERLAP_TOKENS"):
        Config(_env_file=None)


def test_recursive_mode_builds_a_real_token_capped_chunker(monkeypatch):
    pytest.importorskip("tiktoken")
    from rag_sdk.tokenizer import openai_token_counter

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("CHUNKING", "recursive")
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "40")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "8")
    config = Config(_env_file=None)
    chunker = build_chunker(config)
    assert chunker is not None
    count = openai_token_counter(config.embedding_model)
    chunks = chunker(WALL)
    assert len(chunks) > 5 and all(count(c) <= 40 for c in chunks)


def test_unknown_embedding_model_fails_at_startup_not_at_first_upload(monkeypatch):
    pytest.importorskip("tiktoken")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("CHUNKING", "recursive")
    monkeypatch.setenv("EMBEDDING_MODEL", "not-a-real-model")
    with pytest.raises(ValueError, match="no known tokenizer"):
        build_chunker(Config(_env_file=None))


@pytest.fixture
async def client_with_chunker() -> AsyncIterator[tuple[AsyncClient, IngestionService]]:
    service = MagicMock(spec=IngestionService)
    service.ingest_with_dedup = AsyncMock(return_value=(1, 0))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ingestion_service = service
        app.state.chunker = word_chunker
        yield

    async with LifespanManager(create_app(lifespan=lifespan)) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c, service


async def test_ingest_route_uses_the_configured_chunker(client_with_chunker):
    client, service = client_with_chunker
    response = await client.post(
        "/ingest", files={"file": ("wall.txt", BytesIO(WALL.encode()), "text/plain")}
    )
    assert response.status_code == 200
    pages = service.ingest_with_dedup.await_args.args[1]
    assert len(pages) > 20 and all(words(p.text) <= 20 for p in pages)
