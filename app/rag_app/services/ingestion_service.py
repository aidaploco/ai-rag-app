import uuid

from rag_sdk.providers.base import Indexer
from rag_sdk.workflow.ingestion import IngestionWorkflow

from rag_app.schema import DocumentContext


class IngestionService:
    def __init__(
        self,
        workflow: IngestionWorkflow[DocumentContext],
        indexer: Indexer,
    ) -> None:
        self._workflow = workflow
        self._indexer = indexer

    async def ingest_with_dedup(
        self, context: DocumentContext, pages: list[str]
    ) -> tuple[int, int]:
        new_pages = []
        for page in pages:
            content_id = str(uuid.uuid5(uuid.NAMESPACE_OID, page))
            if not await self._indexer.has_document({"content_id": content_id}):
                new_pages.append(page)

        if new_pages:
            await self._workflow.ingest(context, new_pages)

        return len(new_pages), len(pages) - len(new_pages)
