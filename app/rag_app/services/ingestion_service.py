from rag_sdk.providers.base import Indexer
from rag_sdk.workflow.ingestion import IngestionWorkflow
from rag_sdk.workflow.schema_extension import content_id_for

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
        all_ids = [content_id_for(p) for p in pages]
        existing = await self._indexer.existing_content_ids(all_ids)
        new_pages = [p for p, cid in zip(pages, all_ids) if cid not in existing]

        if new_pages:
            await self._workflow.ingest(context, new_pages)

        return len(new_pages), len(pages) - len(new_pages)
