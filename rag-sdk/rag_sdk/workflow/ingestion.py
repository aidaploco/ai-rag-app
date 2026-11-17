from collections.abc import Sequence
from typing import Any, Generic

from rag_sdk.core.tracing import trace
from rag_sdk.providers.base import Documents, Embedder, Indexer
from rag_sdk.workflow.schema_extension import DocumentProcessor, RagSchemaExtension, TContext


class IngestionWorkflow(Generic[TContext]):
    def __init__(
        self,
        embedder: Embedder,
        indexer: Indexer,
        index: str,
        schema_extension: RagSchemaExtension[TContext],
    ) -> None:
        self._embedder = embedder
        self._indexer = indexer
        self._index = index
        self._schema_extension = schema_extension
        self._processor = DocumentProcessor(schema_extension)

    async def ensure_index_exists(self) -> None:
        schema = self._schema_extension.get_schema_extension()
        await self._indexer.create_or_update_index(self._index, schema)

    @trace()
    async def ingest(
        self,
        context: TContext,
        pages: list[str],
        page_metadata: Sequence[dict[str, Any]] | None = None,
    ) -> Documents:
        """Embed and index ``pages``.

        ``page_metadata`` is an optional list parallel to ``pages``. When given, each entry is the
        document's base metadata (the SDK adds no position of its own, and extensions cannot
        override it). When omitted, each document gets ``page_nr`` = its index in ``pages``.
        """
        if page_metadata is not None and len(page_metadata) != len(pages):
            raise ValueError("page_metadata must have one entry per page")
        if not pages:
            return []
        vectors = await self._embedder.embed_chunks(pages)
        documents = [
            self._processor.process(
                content=page,
                content_vector=vector,
                context=context,
                page_nr=i,
                base_metadata=None if page_metadata is None else page_metadata[i],
            )
            for i, (page, vector) in enumerate(zip(pages, vectors))
        ]
        await self._indexer.upload_documents(documents)
        return documents
