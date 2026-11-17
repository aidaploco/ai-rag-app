from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from rag_app.main import create_app
from rag_app.services.ingestion_service import IngestionService
from rag_app.services.rag_service import RagService
from rag_sdk.providers.base import SearchResult, TokenUsage
from rag_sdk.workflow.rag import RagResult


def _rag_result(response: str = "test answer") -> RagResult:
    return RagResult(
        response=response,
        context=[SearchResult(content="chunk", score=0.9, metadata={"page_nr": 0})],
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.fixture
def ingestion_service() -> IngestionService:
    svc = MagicMock(spec=IngestionService)
    svc.ingest_with_dedup = AsyncMock(return_value=(2, 0))
    return svc


@pytest.fixture
def rag_service() -> RagService:
    svc = MagicMock(spec=RagService)
    svc.answer = AsyncMock(return_value=_rag_result())
    return svc


@pytest.fixture
def app(ingestion_service: IngestionService, rag_service: RagService) -> FastAPI:
    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ingestion_service = ingestion_service
        app.state.rag_service = rag_service
        yield

    return create_app(lifespan=test_lifespan)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with LifespanManager(app) as manager:
        async with AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as c:
            yield c


class TestIngestRoute:
    async def test_returns_counts(
        self, client: AsyncClient, ingestion_service: IngestionService
    ) -> None:
        ingestion_service.ingest_with_dedup = AsyncMock(return_value=(3, 1))
        content = b"page one\n\npage two\n\npage three\n\nskipped"
        response = await client.post(
            "/ingest",
            files={"file": ("doc.txt", BytesIO(content), "text/plain")},
            data={"filename": "doc.txt"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["pages_ingested"] == 3
        assert body["pages_skipped"] == 1

    async def test_falls_back_to_upload_filename(
        self, client: AsyncClient, ingestion_service: IngestionService
    ) -> None:
        ingestion_service.ingest_with_dedup = AsyncMock(return_value=(1, 0))
        response = await client.post(
            "/ingest",
            files={"file": ("fallback.txt", BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 200

    async def test_rejects_non_utf8_upload(
        self, client: AsyncClient, ingestion_service: IngestionService
    ) -> None:
        response = await client.post(
            "/ingest",
            files={"file": ("bin.dat", BytesIO(b"\xff\xfe\x00bad"), "application/octet-stream")},
        )
        assert response.status_code == 415
        ingestion_service.ingest_with_dedup.assert_not_awaited()


class TestGenerateRoute:
    async def test_returns_answer_and_context(self, client: AsyncClient) -> None:
        response = await client.post("/generate", json={"query": "what is rag?"})
        assert response.status_code == 200
        body = response.json()
        assert body["response"] == "test answer"
        assert len(body["context"]) == 1
        assert body["context"][0]["score"] == 0.9
        assert body["token_usage"]["total_tokens"] == 15

    async def test_passes_filters(self, client: AsyncClient, rag_service: RagService) -> None:
        await client.post(
            "/generate",
            json={"query": "filtered query", "filters": {"filename": "doc.txt"}},
        )
        rag_service.answer.assert_awaited_once_with("filtered query", {"filename": "doc.txt"})


class TestE2ERoute:
    async def test_ingest_and_answer(
        self,
        client: AsyncClient,
        ingestion_service: IngestionService,
        rag_service: RagService,
    ) -> None:
        ingestion_service.ingest_with_dedup = AsyncMock(return_value=(2, 0))
        content = b"first paragraph\n\nsecond paragraph"
        response = await client.post(
            "/",
            files={"file": ("test.txt", BytesIO(content), "text/plain")},
            data={"query": "summarise"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["response"] == "test answer"
        assert body["pages_ingested"] == 2
        ingestion_service.ingest_with_dedup.assert_awaited_once()
        rag_service.answer.assert_awaited_once_with("summarise")

    async def test_rejects_non_utf8_upload(
        self, client: AsyncClient, ingestion_service: IngestionService
    ) -> None:
        response = await client.post(
            "/",
            files={"file": ("bin.dat", BytesIO(b"\xff\xfe\x00bad"), "application/octet-stream")},
            data={"query": "q"},
        )
        assert response.status_code == 415
        ingestion_service.ingest_with_dedup.assert_not_awaited()
