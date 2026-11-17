from typing import Any

from rag_sdk.providers.base import Indexer
from rag_sdk.workflow.ingestion import IngestionWorkflow
from rag_sdk.workflow.schema_extension import content_id_for

from rag_app.parsing import Page
from rag_app.schema import DocumentContext


def _metadata(page: Page) -> dict[str, Any]:
    meta: dict[str, Any] = {"chunk_idx": page.chunk_idx}
    if page.page_nr is not None:
        meta["page_nr"] = page.page_nr
    return meta


class IngestionService:
    def __init__(
        self,
        workflow: IngestionWorkflow[DocumentContext],
        indexer: Indexer,
    ) -> None:
        self._workflow = workflow
        self._indexer = indexer

    async def ingest_with_dedup(
        self, context: DocumentContext, pages: list[Page]
    ) -> tuple[int, int]:
        # Order-preserving in-batch dedup (first occurrence wins, keeping its chunk_idx):
        # identical paragraphs share one content ID, so embedding and counting them twice
        # would be wasted work and an overstated total.
        unique: dict[str, Page] = {}
        for page in pages:
            unique.setdefault(page.text, page)
        unique_pages = list(unique.values())
        all_ids = [content_id_for(p.text) for p in unique_pages]
        existing = await self._indexer.existing_content_ids(all_ids)
        new_pages = [p for p, cid in zip(unique_pages, all_ids) if cid not in existing]

        if new_pages:
            await self._workflow.ingest(
                context,
                [p.text for p in new_pages],
                page_metadata=[_metadata(p) for p in new_pages],
            )

        return len(new_pages), len(pages) - len(new_pages)
