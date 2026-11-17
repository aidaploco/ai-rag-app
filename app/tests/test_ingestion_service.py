from unittest.mock import AsyncMock, MagicMock

from rag_app.schema import DocumentContext
from rag_app.services.ingestion_service import IngestionService
from rag_sdk.providers.base import Indexer
from rag_sdk.workflow.ingestion import IngestionWorkflow
from rag_sdk.workflow.schema_extension import content_id_for


def _service(existing: set[str] | None = None) -> tuple[IngestionService, AsyncMock, AsyncMock]:
    indexer = MagicMock(spec=Indexer)
    indexer.existing_content_ids = AsyncMock(return_value=existing or set())
    workflow = MagicMock(spec=IngestionWorkflow)
    workflow.ingest = AsyncMock()
    return (
        IngestionService(workflow=workflow, indexer=indexer),
        workflow.ingest,
        indexer.existing_content_ids,
    )


async def test_ingests_all_pages_when_none_exist():
    service, ingest, _ = _service()
    ctx = DocumentContext(filename="a.txt")
    assert await service.ingest_with_dedup(ctx, ["one", "two"]) == (2, 0)
    ingest.assert_awaited_once_with(ctx, ["one", "two"])


async def test_skips_pages_that_already_exist():
    service, ingest, _ = _service(existing={content_id_for("one")})
    ctx = DocumentContext(filename="a.txt")
    assert await service.ingest_with_dedup(ctx, ["one", "two"]) == (1, 1)
    ingest.assert_awaited_once_with(ctx, ["two"])


async def test_does_not_call_workflow_when_everything_exists():
    service, ingest, _ = _service(existing={content_id_for("one")})
    assert await service.ingest_with_dedup(DocumentContext("a.txt"), ["one"]) == (0, 1)
    ingest.assert_not_awaited()


async def test_duplicate_pages_within_one_upload_are_ingested_once():
    service, ingest, existing = _service()
    ctx = DocumentContext(filename="a.txt")
    assert await service.ingest_with_dedup(ctx, ["same", "other", "same"]) == (2, 1)
    ingest.assert_awaited_once_with(ctx, ["same", "other"])
    existing.assert_awaited_once_with([content_id_for("same"), content_id_for("other")])
